// ============================================================================
//  lgn/viz.hpp  —  OpenGL/GLFW/ImGui debug viewer
//
//  Renders:
//    ● circle        revolute/continuous joint  + θ label [rad]
//    ■ square        prismatic joint            + d label [m]
//    ◆ diamond       fixed joint
//    ◈ tetrahedron   link CoM + world-coord label
//    ★ asterisk      tip (leaf) links
//    → red arrow     applied force (via apply_wrench)
//    ↺ amber arc     applied torque (via apply_wrench)
//    -- yellow       Hv velocity frame (ω, v) at each link
//    ● cyan sphere   contact point
//    → cyan arrow    contact normal
//    → magenta arrow contact force magnitude + direction
//    ○ green wire    FreeSphere world object (radius to scale)
//    XYZ gizmo at world origin
//
//  Usage:
//    lgn::Viz viz(tree);                           // create window
//    lgn::ContactWorld world; ...                  // set up contacts
//    while (viz.is_open()) {
//        world.step_free_spheres(dt);
//        tree.fk(q);
//        auto contacts = world.detect(tree);
//        world.resolve_soft(contacts, tree, q, dq);
//        viz.set_q(q);
//        viz.set_contacts(contacts);               // ← pass contacts
//        viz.set_contact_world(&world);            // ← pass world (for spheres)
//        viz.frame();
//    }
//
//  Dependencies (apt): libglfw3-dev libglew-dev
//  ImGui: vendored, point cmake at it with -DIMGUI_DIR=...
// ============================================================================
#pragma once
#include "kinematic_tree.hpp"
#include "contacts.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lgn {

struct Wrench {
    Vec3 force {Vec3::Zero()};
    Vec3 torque{Vec3::Zero()};
};

// ============================================================================
class Viz {
public:
    explicit Viz(KinematicTree& tree,
                 const std::string& title = "lgn debug viewer",
                 int win_w = 1280, int win_h = 800)
        : tree_(tree)
    {
        if (!glfwInit())
            throw std::runtime_error("Viz: glfwInit failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        win_ = glfwCreateWindow(win_w, win_h, title.c_str(), nullptr, nullptr);
        if (!win_) { glfwTerminate(); throw std::runtime_error("Viz: window failed"); }
        glfwMakeContextCurrent(win_);
        glfwSwapInterval(1);
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK)
            throw std::runtime_error("Viz: glewInit failed");
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_MULTISAMPLE);
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(win_, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        cam_.yaw=-90.f; cam_.pitch=-25.f; cam_.dist=5.f;
        cam_.target={0.f,1.f,0.f};
        build_gl();
        q_  = Eigen::VectorXd::Zero(tree_.n_dof());
        dq_ = Eigen::VectorXd::Zero(tree_.n_dof());
        build_parent_map();
    }

    ~Viz() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glDeleteVertexArrays(1,&vao_); glDeleteBuffers(1,&vbo_);
        glDeleteProgram(shader_);
        glfwDestroyWindow(win_); glfwTerminate();
    }

    bool is_open() const { return !glfwWindowShouldClose(win_); }

    void set_q (const Eigen::VectorXd& q)  { q_  = q; }
    void set_dq(const Eigen::VectorXd& dq) { dq_ = dq; }

    void apply_wrench(const std::string& joint_name, const Wrench& w) {
        wrenches_[joint_name] = w;
    }
    void clear_wrenches() { wrenches_.clear(); }

    /// Pass the latest resolved contact points so they are visualised.
    void set_contacts(const std::vector<ContactPoint>& contacts) {
        contacts_ = contacts;
    }

    /// Pass the ContactWorld so FreeSpheres are rendered.
    void set_contact_world(const ContactWorld* world) { cworld_ = world; }

    void frame() {
        glfwPollEvents();

        // ── Camera input — polled directly so we don't fight ImGui callbacks ──
        if (!ImGui::GetIO().WantCaptureMouse) {
            double mx, my;
            glfwGetCursorPos(win_, &mx, &my);
            bool lmb = glfwGetMouseButton(win_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            if (lmb && !cam_.dragging) {
                cam_.dragging = true;
                cam_.last_x = mx; cam_.last_y = my;
            } else if (!lmb) {
                cam_.dragging = false;
            }
            if (cam_.dragging && lmb) {
                cam_.yaw   += (float)(mx - cam_.last_x) * 0.4f;
                cam_.pitch  = std::clamp(
                    cam_.pitch - (float)(my - cam_.last_y) * 0.4f, -89.f, 89.f);
                cam_.last_x = mx; cam_.last_y = my;
            }
        } else {
            cam_.dragging = false;
        }
        // Scroll zoom via ImGui (works even without our own scroll callback)
        if (!ImGui::GetIO().WantCaptureMouse) {
            float scroll = ImGui::GetIO().MouseWheel;
            if (scroll != 0.f)
                cam_.dist = std::clamp(cam_.dist - scroll * 0.3f, 0.2f, 50.f);
        }
        tree_.fk(q_);
        if (dq_.size() == tree_.n_dof())
            tree_.velocity_propagation(q_, dq_);

        int fw, fh;
        glfwGetFramebufferSize(win_, &fw, &fh);
        glViewport(0,0,fw,fh);
        glClearColor(0.12f,0.12f,0.14f,1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        float asp = fh>0 ? (float)fw/fh : 1.f;
        Eigen::Matrix4f proj = perspective(60.f*(float)M_PI/180.f, asp, 0.01f, 200.f);
        Eigen::Matrix4f view = look_at();

        glUseProgram(shader_);
        set_mat4("uProj", proj); set_mat4("uView", view);
        set_mat4("uModel", Eigen::Matrix4f::Identity());

        draw_tree(proj, view);
        draw_contacts(proj, view);
        draw_free_spheres(proj, view);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw_imgui();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win_);
    }

private:
    struct Vert { float x,y,z,r,g,b; };

    GLFWwindow* win_{nullptr};
    GLuint vao_{0}, vbo_{0}, shader_{0};
    static constexpr int VBO_MAX = 8192;

    KinematicTree& tree_;
    Eigen::VectorXd q_, dq_;
    std::unordered_map<std::string,Wrench> wrenches_;
    std::vector<ContactPoint>              contacts_;
    const ContactWorld*                    cworld_{nullptr};

    // Prebuilt: child link index → parent link index
    std::vector<int> parent_link_;

    struct Camera {
        float yaw,pitch,dist;
        Eigen::Vector3f target;
        bool dragging{false};
        double last_x{0},last_y{0};
    } cam_;

    // ── Colours ───────────────────────────────────────────────────────────────
    static constexpr float C_LINK[]    = {0.55f,0.62f,0.70f};
    static constexpr float C_REV[]     = {0.22f,0.54f,0.87f};
    static constexpr float C_PRIS[]    = {0.11f,0.72f,0.46f};
    static constexpr float C_FIX[]     = {0.50f,0.50f,0.48f};
    static constexpr float C_COM[]     = {0.11f,0.72f,0.46f};
    static constexpr float C_TIP[]     = {0.85f,0.33f,0.49f};
    static constexpr float C_FORCE[]   = {0.85f,0.30f,0.18f};
    static constexpr float C_TORQUE[]  = {0.85f,0.60f,0.08f};
    static constexpr float C_VEL[]     = {0.90f,0.90f,0.30f};
    static constexpr float C_CONTACT[] = {0.20f,0.85f,0.85f};  // cyan contact point/normal
    static constexpr float C_CFORCE[]  = {0.85f,0.20f,0.85f};  // magenta contact force
    static constexpr float C_SPHERE[]  = {0.20f,0.80f,0.35f};  // green free sphere

    // ── GL setup ──────────────────────────────────────────────────────────────
    void build_gl() {
        glGenVertexArrays(1,&vao_); glGenBuffers(1,&vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER,vbo_);
        glBufferData(GL_ARRAY_BUFFER, VBO_MAX*sizeof(Vert), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vert),(void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vert),(void*)(3*sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);

        shader_ = build_shader(
            "#version 330 core\n"
            "layout(location=0) in vec3 aPos;\n"
            "layout(location=1) in vec3 aCol;\n"
            "uniform mat4 uProj,uView,uModel;\n"
            "out vec3 vCol;\n"
            "void main(){\n"
            "  gl_Position=uProj*uView*uModel*vec4(aPos,1.0);\n"
            "  vCol=aCol;}\n",
            "#version 330 core\n"
            "in vec3 vCol; out vec4 FragColor;\n"
            "void main(){FragColor=vec4(vCol,1.0);}\n");
    }

    void build_parent_map() {
        int nl = tree_.n_links();
        parent_link_.assign(nl, -1);
        for (int li = 0; li < nl; ++li)
            for (int ji : tree_.link(li).child_joints)
                parent_link_[tree_.joint(ji).child_link] = li;
    }

    // ── Drawing ───────────────────────────────────────────────────────────────
    void draw_verts(const std::vector<Vert>& v, GLenum mode, float lw=1.f) {
        if (v.empty()) return;
        size_t n = std::min(v.size(), (size_t)VBO_MAX);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER,vbo_);
        glBufferSubData(GL_ARRAY_BUFFER,0,n*sizeof(Vert),v.data());
        glLineWidth(lw);
        glDrawArrays(mode,0,(GLsizei)n);
        glBindVertexArray(0);
    }

    Vert V(const Vec3& p, const float* c) const {
        return {(float)p.x(),(float)p.y(),(float)p.z(),c[0],c[1],c[2]};
    }
    Vert Vf(float x,float y,float z,const float* c) const {
        return {x,y,z,c[0],c[1],c[2]};
    }

    void draw_line(const Vec3& a, const Vec3& b, const float* col, float w=1.f) {
        std::vector<Vert> v={V(a,col),V(b,col)};
        draw_verts(v,GL_LINES,w);
    }

    void draw_arrow(const Vec3& from, const Vec3& to, const float* col, float lw=1.5f) {
        Vec3 d=to-from; float len=(float)d.norm();
        if (len<1e-6f) return;
        draw_line(from,to,col,lw);
        Vec3 dn=d/len;
        Vec3 perp=dn.cross(Vec3::UnitY());
        if (perp.norm()<0.1) perp=dn.cross(Vec3::UnitX());
        perp.normalize();
        float hs=std::min(len*0.25f,0.08f);
        draw_line(to, to-dn*hs+perp*hs*0.4, col, lw);
        draw_line(to, to-dn*hs-perp*hs*0.4, col, lw);
    }

    void draw_circle(const Vec3& c, const Mat3& R, double r, const float* col) {
        const int N=24;
        std::vector<Vert> v;
        for (int i=0;i<N;++i) {
            double a0=2*M_PI*i/N, a1=2*M_PI*(i+1)/N;
            v.push_back(V(c+R.col(0)*r*cos(a0)+R.col(1)*r*sin(a0),col));
            v.push_back(V(c+R.col(0)*r*cos(a1)+R.col(1)*r*sin(a1),col));
        }
        v.push_back(V(c+R.col(0)*r*0.5,col)); v.push_back(V(c-R.col(0)*r*0.5,col));
        v.push_back(V(c+R.col(1)*r*0.5,col)); v.push_back(V(c-R.col(1)*r*0.5,col));
        draw_verts(v,GL_LINES,2.f);
    }

    void draw_square(const Vec3& c, const Mat3& R, double r, const float* col) {
        Vec3 corners[4]={c+R.col(0)*r+R.col(1)*r,c+R.col(0)*r-R.col(1)*r,
                         c-R.col(0)*r-R.col(1)*r,c-R.col(0)*r+R.col(1)*r};
        std::vector<Vert> v;
        for (int i=0;i<4;++i) {
            v.push_back(V(corners[i],col));
            v.push_back(V(corners[(i+1)%4],col));
        }
        draw_verts(v,GL_LINES,2.f);
    }

    void draw_diamond(const Vec3& c, const Mat3& R, double r, const float* col) {
        Vec3 u=c+R.col(1)*r,d=c-R.col(1)*r,rt=c+R.col(0)*r,lt=c-R.col(0)*r;
        std::vector<Vert> v={V(u,col),V(rt,col),V(rt,col),V(d,col),
                              V(d,col),V(lt,col),V(lt,col),V(u,col)};
        draw_verts(v,GL_LINES,2.f);
    }

    void draw_tetra(const Eigen::Vector3f& c, float r, const float* col) {
        Eigen::Vector3f v0=c+Eigen::Vector3f(0,r,0);
        Eigen::Vector3f v1=c+Eigen::Vector3f(r,-r*.5f,r*.5f);
        Eigen::Vector3f v2=c+Eigen::Vector3f(-r,-r*.5f,r*.5f);
        Eigen::Vector3f v3=c+Eigen::Vector3f(0,-r*.5f,-r);
        std::vector<Vert> v;
        auto edge=[&](Eigen::Vector3f a,Eigen::Vector3f b){
            v.push_back(Vf(a.x(),a.y(),a.z(),col));
            v.push_back(Vf(b.x(),b.y(),b.z(),col));
        };
        edge(v0,v1);edge(v0,v2);edge(v0,v3);
        edge(v1,v2);edge(v1,v3);edge(v2,v3);
        draw_verts(v,GL_LINES,1.5f);
    }

    void draw_star(const Vec3& c, double r, const float* col) {
        std::vector<Vert> v;
        for (int i=0;i<6;++i) {
            double a=i*M_PI/3.0;
            v.push_back(V(c,col)); v.push_back(V(c+Vec3(r*cos(a),r*sin(a),0),col));
        }
        v.push_back(V(c,col)); v.push_back(V(c+Vec3(0,0, r),col));
        v.push_back(V(c,col)); v.push_back(V(c+Vec3(0,0,-r),col));
        draw_verts(v,GL_LINES,2.f);
    }

    /// Wireframe sphere (3 latitude circles + 3 longitude circles)
    void draw_wire_sphere(const Vec3& centre, double r, const float* col) {
        const int N=32;
        // 3 great circles: XY, XZ, YZ planes
        Eigen::Vector3f axes[3][2]={
            {{1,0,0},{0,1,0}},
            {{1,0,0},{0,0,1}},
            {{0,1,0},{0,0,1}}
        };
        for (auto& ax : axes) {
            std::vector<Vert> v;
            for (int i=0;i<N;++i) {
                double a0=2*M_PI*i/N, a1=2*M_PI*(i+1)/N;
                Vec3 p0=centre+ax[0].cast<double>()*r*cos(a0)
                              +ax[1].cast<double>()*r*sin(a0);
                Vec3 p1=centre+ax[0].cast<double>()*r*cos(a1)
                              +ax[1].cast<double>()*r*sin(a1);
                v.push_back(V(p0,col)); v.push_back(V(p1,col));
            }
            draw_verts(v,GL_LINES,1.5f);
        }
    }

    void draw_torque_arc(const Vec3& c, const Vec3& torque) {
        double tl=torque.norm(); if (tl<0.01) return;
        Vec3 ax=torque/tl;
        Vec3 perp=ax.cross(Vec3::UnitY());
        if (perp.norm()<0.1) perp=ax.cross(Vec3::UnitX());
        perp.normalize(); Vec3 perp2=ax.cross(perp);
        double r=0.10; const int N=20;
        std::vector<Vert> v;
        for (int i=0;i<N;++i) {
            double a0=2*M_PI*i/N, a1=2*M_PI*(i+1)/N;
            v.push_back(V(c+perp*r*cos(a0)+perp2*r*sin(a0),C_TORQUE));
            v.push_back(V(c+perp*r*cos(a1)+perp2*r*sin(a1),C_TORQUE));
        }
        draw_verts(v,GL_LINES,2.f);
        draw_arrow(c+perp*r-perp2*0.02, c+perp*r+perp2*0.04, C_TORQUE, 2.f);
    }

    // ── Main tree draw ────────────────────────────────────────────────────────
    void draw_tree(const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view) {
        int nl=tree_.n_links();
        for (int li=0; li<nl; ++li) {
            const Link& lk=tree_.link(li);
            Vec3 pw=p_of(lk.T_world);

            // Link line to parent
            if (parent_link_[li]>=0)
                draw_line(p_of(tree_.link(parent_link_[li]).T_world), pw, C_LINK, 4.f);

            // Joint symbol
            if (lk.parent_joint>=0) {
                const Joint& pj=tree_.joint(lk.parent_joint);
                Mat3 R=R_of(lk.T_world).cast<double>();
                switch (pj.type) {
                    case JointType::Revolute: case JointType::Continuous:
                        draw_circle(pw,R,0.06,C_REV);
                        if (pj.dof_index>=0)
                            label(pw,proj,view,
                                pj.name+"\n\xce\xb8="+fmt(q_[pj.dof_index],3)+" rad",
                                {0.4f,0.7f,1.f});
                        break;
                    case JointType::Prismatic:
                        draw_square(pw,R,0.06,C_PRIS);
                        if (pj.dof_index>=0)
                            label(pw,proj,view,
                                pj.name+"\nd="+fmt(q_[pj.dof_index],3)+" m",
                                {0.2f,0.9f,0.6f});
                        break;
                    case JointType::Fixed:
                        draw_diamond(pw,R,0.05,C_FIX);
                        break;
                    default:
                        draw_diamond(pw,R,0.05,C_PRIS);
                }
            }

            // CoM
            if (lk.inertial.mass>1e-9) {
                Vec3 cw=p_of(lk.T_world)+R_of(lk.T_world)*lk.inertial.com;
                draw_tetra(cw.cast<float>(),0.045f,C_COM);
                label(cw,proj,view,lk.name+"\n["+fmt(cw.x(),2)+" "+
                      fmt(cw.y(),2)+" "+fmt(cw.z(),2)+"]",{0.2f,0.9f,0.6f});
            }

            // Tip
            if (lk.child_joints.empty()) {
                draw_star(pw,0.07,C_TIP);
                label(pw,proj,view,"tip:"+lk.name+"\n["+
                      fmt(pw.x(),3)+" "+fmt(pw.y(),3)+" "+fmt(pw.z(),3)+"]",
                      {1.f,0.5f,0.7f});
            }

            // Hv velocity arrows
            if (dq_.size()==tree_.n_dof()) {
                Vec3 om=omega_of(lk.Hv_world), vl=v_of(lk.Hv_world);
                if (om.norm()>0.01) draw_arrow(pw,pw+om*0.20,C_VEL,1.5f);
                if (vl.norm()>0.01) draw_arrow(pw,pw+vl*0.15,C_VEL,1.5f);
            }

            // Applied wrench
            if (lk.parent_joint>=0) {
                const std::string& jn=tree_.joint(lk.parent_joint).name;
                auto it=wrenches_.find(jn);
                if (it!=wrenches_.end()) {
                    if (it->second.force.norm()>0.01)
                        draw_arrow(pw,pw+it->second.force*0.05,C_FORCE,2.5f);
                    if (it->second.torque.norm()>0.01)
                        draw_torque_arc(pw,it->second.torque);
                }
            }
        }

        // XYZ gizmo
        float r=0.3f;
        float cx[]={0.9f,0.2f,0.2f},cy[]={0.2f,0.9f,0.2f},cz[]={0.2f,0.5f,0.9f};
        draw_line({0,0,0},{r,0,0},cx,2.f);
        draw_line({0,0,0},{0,r,0},cy,2.f);
        draw_line({0,0,0},{0,0,r},cz,2.f);
        label({(double)r+0.03,0,0},proj,view,"X",{0.9f,0.2f,0.2f});
        label({0,(double)r+0.03,0},proj,view,"Y",{0.2f,0.9f,0.2f});
        label({0,0,(double)r+0.03},proj,view,"Z",{0.2f,0.5f,0.9f});
    }

    // ── Contact visualisation ─────────────────────────────────────────────────
    void draw_contacts(const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view) {
        for (const auto& cp : contacts_) {
            // Contact point: small cyan cross
            double cr=0.015;
            Vec3 ex=cp.point_w+Vec3::UnitX()*cr, exn=cp.point_w-Vec3::UnitX()*cr;
            Vec3 ey=cp.point_w+Vec3::UnitY()*cr, eyn=cp.point_w-Vec3::UnitY()*cr;
            Vec3 ez=cp.point_w+Vec3::UnitZ()*cr, ezn=cp.point_w-Vec3::UnitZ()*cr;
            draw_line(exn,ex,C_CONTACT,2.f);
            draw_line(eyn,ey,C_CONTACT,2.f);
            draw_line(ezn,ez,C_CONTACT,2.f);

            // Contact normal arrow (cyan, 0.08 m long)
            draw_arrow(cp.point_w,
                       cp.point_w + cp.normal_w * 0.08,
                       C_CONTACT, 2.f);

            // Contact force arrow (magenta, scaled by magnitude)
            double fmag = cp.force_w.norm();
            if (fmag > 0.1) {
                // Scale: 1 N → 0.01 m arrow length, capped at 0.15 m
                double scale = std::min(fmag * 0.01, 0.15);
                draw_arrow(cp.point_w,
                           cp.point_w + cp.force_w.normalized() * scale,
                           C_CFORCE, 2.5f);
                // Force magnitude label
                label(cp.point_w + cp.force_w.normalized()*scale*1.1,
                      proj, view, fmt(fmag,1)+" N", {0.85f,0.20f,0.85f});
            }

            // Penetration depth label
            if (cp.penetration > 1e-4)
                label(cp.point_w, proj, view,
                      "d="+fmt(cp.penetration*1000,1)+"mm",
                      {0.20f,0.85f,0.85f});
        }
    }

    // ── Free sphere visualisation ─────────────────────────────────────────────
    void draw_free_spheres(const Eigen::Matrix4f& proj,
                            const Eigen::Matrix4f& view) {
        if (!cworld_) return;
        for (int i=0; i<cworld_->n_free_spheres(); ++i) {
            const FreeSphere& fs = cworld_->free_sphere(i);
            draw_wire_sphere(fs.pos, fs.radius, C_SPHERE);
            // Velocity arrow (green, scaled)
            if (fs.vel.norm() > 0.01)
                draw_arrow(fs.pos, fs.pos+fs.vel*0.05, C_SPHERE, 1.5f);
            // Label: name + speed
            label(fs.pos+Vec3(0,fs.radius+0.02,0), proj, view,
                  fs.name+"\n"+fmt(fs.vel.norm(),2)+" m/s",
                  {0.20f,0.80f,0.35f});
        }
    }

    // ── ImGui panel ───────────────────────────────────────────────────────────
    void draw_imgui() {
        ImGui::SetNextWindowPos({10,10},ImGuiCond_Always);
        ImGui::SetNextWindowSize({310,0},ImGuiCond_Always);
        ImGui::Begin("lgn debug",nullptr,
            ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|
            ImGuiWindowFlags_NoCollapse);

        ImGui::TextColored({0.5f,0.8f,1.f,1.f},
            "%d links  %d DOFs  %zu tips",
            tree_.n_links(), tree_.n_dof(), tree_.tips().size());

        if (!contacts_.empty()) {
            ImGui::TextColored({0.20f,0.85f,0.85f,1.f},
                "%zu contact(s) active", contacts_.size());
            for (size_t i=0; i<contacts_.size(); ++i) {
                const auto& cp=contacts_[i];
                ImGui::TextColored({0.85f,0.20f,0.85f,0.9f},
                    "  [%zu] pen=%.2fmm F=%.1fN",
                    i, cp.penetration*1000.0, cp.force_w.norm());
            }
        }

        if (cworld_ && cworld_->n_free_spheres()>0) {
            ImGui::Separator();
            ImGui::TextColored({0.20f,0.80f,0.35f,1.f},"Free spheres");
            for (int i=0; i<cworld_->n_free_spheres(); ++i) {
                const auto& fs=cworld_->free_sphere(i);
                ImGui::Text("  %s  pos=[%.2f %.2f %.2f]  v=%.2fm/s",
                    fs.name.c_str(),
                    fs.pos.x(),fs.pos.y(),fs.pos.z(),
                    fs.vel.norm());
            }
        }

        ImGui::Separator();
        const auto& dnames=tree_.dof_names();
        bool changed=false;
        for (int i=0; i<(int)dnames.size(); ++i) {
            float lo=-3.14f,hi=3.14f;
            for (const auto& jt:tree_.joints_ref())
                if (jt.dof_index==i&&jt.limits.has_limits)
                    {lo=(float)jt.limits.lower;hi=(float)jt.limits.upper;break;}
            float v=(float)q_[i];
            if (ImGui::SliderFloat(dnames[i].c_str(),&v,lo,hi,"%.3f"))
                {q_[i]=v;changed=true;}
        }
        if (changed) tree_.fk(q_);

        ImGui::Separator();
        ImGui::TextColored({0.9f,0.6f,0.1f,1.f},"Wrenches");
        for (const auto& jt:tree_.joints_ref()) {
            if (jt.type==JointType::Fixed) continue;
            ImGui::PushID(jt.dof_index);
            if (ImGui::TreeNode(("@ "+jt.name).c_str())) {
                Wrench& w=wrenches_[jt.name];
                float f[3]={(float)w.force.x(),(float)w.force.y(),(float)w.force.z()};
                float t[3]={(float)w.torque.x(),(float)w.torque.y(),(float)w.torque.z()};
                if (ImGui::DragFloat3("F [N]", f,0.1f)) w.force={f[0],f[1],f[2]};
                if (ImGui::DragFloat3("t [Nm]",t,0.1f)) w.torque={t[0],t[1],t[2]};
                if (ImGui::Button("Clear")) w={};
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        ImGui::Separator();
        ImGui::TextColored({0.6f,0.6f,0.6f,1.f},
            "Drag=orbit  Scroll=zoom");
        ImGui::TextColored({0.20f,0.85f,0.85f,1.f},
            "Cyan = contact point/normal");
        ImGui::TextColored({0.85f,0.20f,0.85f,1.f},
            "Magenta = contact force");
        ImGui::TextColored({0.20f,0.80f,0.35f,1.f},
            "Green wire = free sphere");
        ImGui::End();
    }

    // ── World → screen label ──────────────────────────────────────────────────
    void label(const Vec3& p,
                const Eigen::Matrix4f& proj,
                const Eigen::Matrix4f& view,
                const std::string& text,
                Eigen::Vector3f col={1,1,1}) const
    {
        int fw,fh; glfwGetFramebufferSize(win_,&fw,&fh);
        Eigen::Vector4f wp((float)p.x(),(float)p.y(),(float)p.z(),1.f);
        Eigen::Vector4f clip=proj*view*wp;
        if (clip.w()<=0.f) return;
        Eigen::Vector3f ndc=clip.head<3>()/clip.w();
        if (std::abs(ndc.x())>1.1f||std::abs(ndc.y())>1.1f) return;
        float sx=( ndc.x()*0.5f+0.5f)*fw+4.f;
        float sy=(-ndc.y()*0.5f+0.5f)*fh;
        ImGui::GetForegroundDrawList()->AddText(
            {sx,sy},
            ImGui::GetColorU32({col.x(),col.y(),col.z(),1.f}),
            text.c_str());
    }

    // ── Camera ────────────────────────────────────────────────────────────────
    Eigen::Matrix4f look_at() const {
        float yr=cam_.yaw*(float)M_PI/180.f;
        float pr=cam_.pitch*(float)M_PI/180.f;
        Eigen::Vector3f dir(cosf(pr)*cosf(yr),sinf(pr),cosf(pr)*sinf(yr));
        Eigen::Vector3f eye=cam_.target+dir*cam_.dist;
        Eigen::Vector3f f=(cam_.target-eye).normalized();
        Eigen::Vector3f r=f.cross(Eigen::Vector3f::UnitY()).normalized();
        Eigen::Vector3f u=r.cross(f);
        Eigen::Matrix4f V=Eigen::Matrix4f::Zero();
        V(0,0)=r.x();V(0,1)=r.y();V(0,2)=r.z();V(0,3)=-r.dot(eye);
        V(1,0)=u.x();V(1,1)=u.y();V(1,2)=u.z();V(1,3)=-u.dot(eye);
        V(2,0)=-f.x();V(2,1)=-f.y();V(2,2)=-f.z();V(2,3)=f.dot(eye);
        V(3,3)=1.f; return V;
    }

    Eigen::Matrix4f perspective(float fov,float asp,float zn,float zf) const {
        float t=tanf(fov*0.5f);
        Eigen::Matrix4f P=Eigen::Matrix4f::Zero();
        P(0,0)=1.f/(asp*t);P(1,1)=1.f/t;
        P(2,2)=-(zf+zn)/(zf-zn);P(2,3)=-2.f*zf*zn/(zf-zn);
        P(3,2)=-1.f; return P;
    }

    // ── Shader helpers ────────────────────────────────────────────────────────
    GLuint build_shader(const char* vs, const char* fs) {
        auto compile=[](GLenum type,const char* src)->GLuint{
            GLuint s=glCreateShader(type);
            glShaderSource(s,1,&src,nullptr);glCompileShader(s);
            GLint ok;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
            if(!ok){char buf[512];glGetShaderInfoLog(s,512,nullptr,buf);
                    throw std::runtime_error(std::string("Shader:")+buf);}
            return s;};
        GLuint v=compile(GL_VERTEX_SHADER,vs);
        GLuint f=compile(GL_FRAGMENT_SHADER,fs);
        GLuint p=glCreateProgram();
        glAttachShader(p,v);glAttachShader(p,f);
        glLinkProgram(p);
        glDeleteShader(v);glDeleteShader(f);
        return p;
    }
    void set_mat4(const char* nm,const Eigen::Matrix4f& m){
        GLint loc=glGetUniformLocation(shader_,nm);
        if(loc>=0) glUniformMatrix4fv(loc,1,GL_FALSE,m.data());
    }
    static std::string fmt(double v,int d){
        char buf[32];snprintf(buf,sizeof(buf),"%.*f",d,v);return buf;
    }
};

} // namespace lgn
