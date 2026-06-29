// ============================================================================
//  lgn/mjcf_loader.hpp
//  Parses a MuJoCo MJCF file into a KinematicTree + loop constraints.
//  Dependency: TinyXML2. No MuJoCo dependency.
// ============================================================================
#pragma once
#include "kinematic_tree.hpp"
#include "contacts.hpp"        // ContactWorld, Sphere, Capsule, Box
#include <tinyxml2.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <cmath>

namespace lgn {

// ============================================================================
//  MJCFLoader result
// ============================================================================
struct MJCFResult {
    KinematicTree                tree;
    std::vector<LoopConstraint>  loop_constraints;
};

// ============================================================================
//  collect_visual_classes — MUST be defined before load_collision_geometry_mjcf
// ============================================================================
inline void collect_visual_classes(
    tinyxml2::XMLElement* def_el,
    const std::string& /*parent*/,
    bool parent_contype_zero,
    bool parent_conaffinity_zero,
    std::unordered_set<std::string>& visual_classes)
{
    for (auto* child = def_el->FirstChildElement("default"); child;
         child = child->NextSiblingElement("default"))
    {
        const char* cn = child->Attribute("class");
        std::string cls = cn ? std::string(cn) : "";

        bool ct0 = parent_contype_zero;
        bool ca0 = parent_conaffinity_zero;

        if (auto* gel = child->FirstChildElement("geom")) {
            const char* ct = gel->Attribute("contype");
            const char* ca = gel->Attribute("conaffinity");
            if (ct && std::string(ct) == "0") ct0 = true;
            if (ca && std::string(ca) == "0") ca0 = true;
        }

        if (ct0 && ca0) visual_classes.insert(cls);
        collect_visual_classes(child, cls, ct0, ca0, visual_classes);
    }
}

// ============================================================================
//  load_collision_geometry_mjcf
// ============================================================================
inline void load_collision_geometry_mjcf(
    const std::string& mjcf_path,
    const KinematicTree& tree,
    ContactWorld& world)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(mjcf_path.c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "[lgn] collision_loader_mjcf: cannot open " << mjcf_path << "\n";
        return;
    }
    auto* mujoco = doc.FirstChildElement("mujoco");
    if (!mujoco) return;

    std::unordered_set<std::string> visual_classes;
    if (auto* def = mujoco->FirstChildElement("default"))
        collect_visual_classes(def, "", false, false, visual_classes);

    int n_loaded = 0, n_skipped = 0;

    auto* wb = mujoco->FirstChildElement("worldbody");
    if (!wb) return;

    std::function<void(tinyxml2::XMLElement*, const std::string&)> walk;
    walk = [&](tinyxml2::XMLElement* body_el, const std::string& body_name) {
        for (auto* gel = body_el->FirstChildElement("geom"); gel;
             gel = gel->NextSiblingElement("geom"))
        {
            const char* cls = gel->Attribute("class");
            std::string cls_str = cls ? std::string(cls) : "";
            if (visual_classes.count(cls_str)) continue;

            const char* ct = gel->Attribute("contype");
            if (ct && std::string(ct) == "0") continue;

            T_mat T_link_geom = T_identity();
            if (auto* pv = gel->Attribute("pos")) {
                std::istringstream ss(pv);
                double x,y,z; ss >> x >> y >> z;
                T_link_geom.topRightCorner<3,1>() = Vec3(x,y,z);
            }
            if (auto* qv = gel->Attribute("quat")) {
                std::istringstream ss(qv);
                double w,x,y,z; ss >> w >> x >> y >> z;
                Eigen::Quaterniond q(w,x,y,z); q.normalize();
                T_link_geom.topLeftCorner<3,3>() = q.toRotationMatrix();
            }

            const char* gtype = gel->Attribute("type");
            std::string type_str = gtype ? std::string(gtype) : "sphere";
            const char* size_attr = gel->Attribute("size");

            try {
                if (type_str == "sphere") {
                    std::string sz = size_attr ? std::string(size_attr) : "0.01";
                    double r = std::stod(sz.substr(0, sz.find(' ')));
                    world.add_collider(tree, body_name, Sphere{r}, T_link_geom);
                    ++n_loaded;
                } else if (type_str == "capsule" || type_str == "cylinder") {
                    double r = 0.01, hl = 0.05;
                    if (size_attr) { std::istringstream ss(size_attr); ss >> r >> hl; }
                    world.add_collider(tree, body_name, Capsule{r, hl}, T_link_geom);
                    ++n_loaded;
                } else if (type_str == "box") {
                    Vec3 half(0.05, 0.05, 0.05);
                    if (size_attr) {
                        std::istringstream ss(size_attr);
                        ss >> half.x() >> half.y() >> half.z();
                    }
                    world.add_collider(tree, body_name, Box{half}, T_link_geom);
                    ++n_loaded;
                } else {
                    ++n_skipped;
                }
            } catch (...) {}
        }

        for (auto* cb = body_el->FirstChildElement("body"); cb;
             cb = cb->NextSiblingElement("body"))
        {
            const char* cname = cb->Attribute("name");
            if (cname) walk(cb, std::string(cname));
        }
    };

    for (auto* body_el = wb->FirstChildElement("body"); body_el;
         body_el = body_el->NextSiblingElement("body"))
    {
        const char* bn = body_el->Attribute("name");
        if (bn) walk(body_el, std::string(bn));
    }

    std::cout << "[lgn] collision_loader_mjcf: " << n_loaded << " primitives loaded";
    if (n_skipped > 0)
        std::cout << ", " << n_skipped << " shapes skipped";
    std::cout << "\n";
}

// ============================================================================
//  MJCFLoader
// ============================================================================
class MJCFLoader {
public:
    static MJCFResult from_file(const std::string& path) {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
            throw std::runtime_error("MJCFLoader: cannot open " + path);
        return parse(doc);
    }

    static MJCFResult from_string(const std::string& xml) {
        tinyxml2::XMLDocument doc;
        if (doc.Parse(xml.c_str()) != tinyxml2::XML_SUCCESS)
            throw std::runtime_error("MJCFLoader: XML parse failed");
        return parse(doc);
    }

private:
    struct DefaultProps {
        std::optional<std::string> joint_type;
        std::optional<std::string> joint_axis;
        std::optional<std::string> joint_range;
        std::optional<double>      joint_damping;
        std::optional<double>      joint_frictionloss;
        std::optional<double>      joint_armature;
        std::optional<double>      joint_stiffness;
        std::optional<double>      joint_kp;
    };

    using ClassMap = std::unordered_map<std::string, DefaultProps>;

    struct SiteRecord {
        std::string name;
        int         link_idx{-1};
        T_mat       T_link_site{Mat4::Identity()};
    };

    struct PendingLC {
        std::string name;
        std::string site1;
        std::string site2;
    };

    struct State {
        KinematicTree&               tree;
        ClassMap&                    classes;
        std::unordered_map<std::string, int>& link_map;
        std::unordered_map<std::string, SiteRecord>& sites;
        std::vector<PendingLC>&      pending_lcs;
        bool                         degrees;
        int&                         dof_count;
        std::vector<std::string>&    dof_names;
        std::vector<std::string>&    actuated_joints;
    };

    // ── Top-level parse ───────────────────────────────────────────────────────
    static MJCFResult parse(tinyxml2::XMLDocument& doc) {
        auto* mujoco = doc.FirstChildElement("mujoco");
        if (!mujoco)
            throw std::runtime_error("MJCFLoader: no <mujoco> element");

        bool degrees = false;
        if (auto* comp = mujoco->FirstChildElement("compiler")) {
            const char* ang = comp->Attribute("angle");
            if (ang && std::string(ang) == "degree") degrees = true;
        }

        ClassMap classes;
        classes[""] = DefaultProps{};
        if (auto* def = mujoco->FirstChildElement("default"))
            parse_defaults(def, "", classes);

        std::vector<std::string> actuated_joints;
        if (auto* act = mujoco->FirstChildElement("actuator")) {
            for (auto* el = act->FirstChildElement(); el;
                 el = el->NextSiblingElement()) {
                const char* jn = el->Attribute("joint");
                if (jn) actuated_joints.push_back(jn);
            }
        }

        KinematicTree tree;
        std::unordered_map<std::string, int> link_map;
        std::unordered_map<std::string, SiteRecord> sites;
        std::vector<PendingLC> pending_lcs;
        int dof_count = 0;
        std::vector<std::string> dof_names;

        State st{tree, classes, link_map, sites, pending_lcs,
                 degrees, dof_count, dof_names, actuated_joints};

        auto* wb = mujoco->FirstChildElement("worldbody");
        if (!wb)
            throw std::runtime_error("MJCFLoader: no <worldbody> element");

        {
            Link world_link; world_link.name = "world";
            int wi = tree.add_link(world_link);
            link_map["world"] = wi;
            tree.set_root(wi);
        }

        for (auto* body_el = wb->FirstChildElement("body"); body_el;
             body_el = body_el->NextSiblingElement("body"))
            parse_body(body_el, link_map["world"], T_identity(), st);

        tree.set_dof_count(dof_count);
        tree.set_dof_names(dof_names);

        // ── Equality constraints ──────────────────────────────────────────────
        if (auto* eq = mujoco->FirstChildElement("equality")) {
            for (auto* el = eq->FirstChildElement("connect"); el;
                 el = el->NextSiblingElement("connect"))
            {
                const char* s1 = el->Attribute("site1");
                const char* s2 = el->Attribute("site2");
                const char* nm = el->Attribute("name");
                if (!s1 || !s2) continue;
                pending_lcs.push_back({
                    nm ? std::string(nm) : (std::string(s1) + "_" + s2),
                    std::string(s1), std::string(s2)
                });
            }
        }

        std::vector<LoopConstraint> lcs;
        for (const auto& plc : pending_lcs) {
            auto it1 = sites.find(plc.site1);
            auto it2 = sites.find(plc.site2);
            if (it1 == sites.end()) {
                std::cerr << "[lgn] MJCFLoader: site not found: " << plc.site1 << "\n";
                continue;
            }
            if (it2 == sites.end()) {
                std::cerr << "[lgn] MJCFLoader: site not found: " << plc.site2 << "\n";
                continue;
            }
            LoopConstraint lc;
            lc.name     = plc.name;
            lc.link_a   = it1->second.link_idx;
            lc.T_link_a = it1->second.T_link_site;
            lc.link_b   = it2->second.link_idx;
            lc.T_link_b = it2->second.T_link_site;
            tree.add_loop_constraint(lc);
            lcs.push_back(lc);
        }

        tree.finalize();

        std::cout << "[lgn] MJCF loaded: "
                  << tree.n_links()       << " links, "
                  << tree.n_dof()         << " DOFs, "
                  << tree.tips().size()   << " tip(s), "
                  << lcs.size()           << " loop constraint(s)\n";

        return {std::move(tree), std::move(lcs)};
    }

    // ── Default class parsing ─────────────────────────────────────────────────
    static void parse_defaults(tinyxml2::XMLElement* def_el,
                                const std::string& parent_class,
                                ClassMap& classes)
    {
        // Parse top-level <joint> inside this <default> element
        if (auto* jel = def_el->FirstChildElement("joint")) {
            DefaultProps& gp = classes[parent_class];
            merge_joint_defaults(jel, gp);
        }

        for (auto* child = def_el->FirstChildElement("default"); child;
             child = child->NextSiblingElement("default"))
        {
            const char* cn = child->Attribute("class");
            std::string cls_name = cn ? std::string(cn) : "";

            DefaultProps props = classes.count(parent_class)
                                ? classes.at(parent_class)
                                : DefaultProps{};

            if (auto* jel = child->FirstChildElement("joint"))
                merge_joint_defaults(jel, props);

            classes[cls_name] = props;
            parse_defaults(child, cls_name, classes);
        }
    }

    static void merge_joint_defaults(tinyxml2::XMLElement* jel, DefaultProps& p) {
        if (auto* v = jel->Attribute("type"))       p.joint_type = v;
        if (auto* v = jel->Attribute("axis"))       p.joint_axis = v;
        if (auto* v = jel->Attribute("range"))      p.joint_range = v;
        auto get_d = [&](const char* a) -> std::optional<double> {
            const char* v = jel->Attribute(a);
            return v ? std::optional<double>(std::stod(v)) : std::nullopt;
        };
        auto ov = [](auto& opt, auto val) { if (val) opt = val; };
        ov(p.joint_damping,      get_d("damping"));
        ov(p.joint_frictionloss, get_d("frictionloss"));
        ov(p.joint_armature,     get_d("armature"));
        ov(p.joint_stiffness,    get_d("stiffness"));
        ov(p.joint_kp,           get_d("kp"));
    }

    // ── Body traversal ────────────────────────────────────────────────────────
    static void parse_body(tinyxml2::XMLElement* body_el,
                            int parent_link_idx,
                            const T_mat& T_parent_world,
                            State& st)
    {
        const char* bname = body_el->Attribute("name");
        std::string body_name = bname
            ? std::string(bname)
            : ("_body_" + std::to_string(st.link_map.size()));

        T_mat T_body_in_parent = parse_body_transform(body_el, st.degrees);
        T_mat T_body_world     = T_parent_world * T_body_in_parent;

        Link lk;
        lk.name = body_name;
        if (auto* in = body_el->FirstChildElement("inertial"))
            parse_inertial(in, lk.inertial);

        int link_idx = st.tree.add_link(lk);
        st.link_map[body_name] = link_idx;

        // Collect joints for this body
        std::vector<tinyxml2::XMLElement*> joint_els;
        for (auto* jel = body_el->FirstChildElement("joint"); jel;
             jel = jel->NextSiblingElement("joint"))
            joint_els.push_back(jel);

        if (joint_els.empty()) {
            Joint jt;
            jt.name           = body_name + "_fixed";
            jt.type           = JointType::Fixed;
            jt.dof_count      = 0;
            jt.T_parent_joint = T_body_in_parent;
            jt.child_link     = link_idx;
            int ji = st.tree.add_joint(jt);
            st.tree.link(parent_link_idx).child_joints.push_back(ji);
            st.tree.link(link_idx).parent_joint = ji;
        } else {
            int prev_parent = parent_link_idx;
            T_mat T_accum   = T_body_in_parent;

            for (int ji_idx = 0; ji_idx < (int)joint_els.size(); ++ji_idx) {
                auto* jel = joint_els[ji_idx];

                int child_of_this_joint;
                if (ji_idx == (int)joint_els.size() - 1) {
                    child_of_this_joint = link_idx;
                } else {
                    Link intermediate;
                    intermediate.name = body_name + "_jlink" + std::to_string(ji_idx);
                    child_of_this_joint = st.tree.add_link(intermediate);
                    st.link_map[intermediate.name] = child_of_this_joint;
                }

                const char* cls_attr = jel->Attribute("class");
                std::string cls = cls_attr ? std::string(cls_attr) : "";
                const DefaultProps& defs = st.classes.count(cls)
                    ? st.classes.at(cls)
                    : st.classes.at("");

                Joint jt = build_joint(jel, defs, T_accum,
                                       child_of_this_joint,
                                       st.degrees, st.dof_count,
                                       st.dof_names,
                                       st.actuated_joints);

                int ji = st.tree.add_joint(jt);
                st.tree.link(prev_parent).child_joints.push_back(ji);
                st.tree.link(child_of_this_joint).parent_joint = ji;

                T_accum     = T_identity();
                prev_parent = child_of_this_joint;
            }
        }

        // Sites
        for (auto* sel = body_el->FirstChildElement("site"); sel;
             sel = sel->NextSiblingElement("site"))
        {
            const char* sn = sel->Attribute("name");
            if (!sn) continue;
            SiteRecord sr;
            sr.name        = sn;
            sr.link_idx    = link_idx;
            sr.T_link_site = parse_body_transform(sel, st.degrees);
            st.sites[sr.name] = sr;
        }

        // Recurse
        for (auto* child_body = body_el->FirstChildElement("body"); child_body;
             child_body = child_body->NextSiblingElement("body"))
            parse_body(child_body, link_idx, T_body_world, st);
    }

    // ── Build Joint from <joint> element ─────────────────────────────────────
    static Joint build_joint(tinyxml2::XMLElement* jel,
                               const DefaultProps& defs,
                               const T_mat& T_parent_joint,
                               int child_link,
                               bool degrees,
                               int& dof_count,
                               std::vector<std::string>& dof_names,
                               const std::vector<std::string>& /*actuated*/)
    {
        Joint jt;

        const char* jname = jel->Attribute("name");
        jt.name = jname ? std::string(jname)
                        : ("_joint_" + std::to_string(dof_count));

        const char* type_attr = jel->Attribute("type");
        std::string type_str  = type_attr ? std::string(type_attr)
                                          : defs.joint_type.value_or("hinge");
        if      (type_str == "hinge")  jt.type = JointType::Revolute;
        else if (type_str == "slide")  jt.type = JointType::Prismatic;
        else if (type_str == "ball")   jt.type = JointType::Ball;
        else if (type_str == "free")   jt.type = JointType::Floating;
        else                           jt.type = JointType::Fixed;

        jt.dof_count = joint_dof_count(jt.type);
        if (jt.dof_count > 0) {
            jt.dof_index  = dof_count;
            dof_count    += jt.dof_count;
            if (jt.dof_count == 1) {
                dof_names.push_back(jt.name);
            } else {
                for (int d = 0; d < jt.dof_count; ++d)
                    dof_names.push_back(jt.name + "_" + std::to_string(d));
            }
        }

        {
            const char* ax = jel->Attribute("axis");
            std::string ax_str = ax ? std::string(ax)
                                    : defs.joint_axis.value_or("0 0 1");
            jt.axis = parse_vec3_str(ax_str).normalized();
        }

        {
            const char* rng = jel->Attribute("range");
            std::string rng_str = rng ? std::string(rng)
                                      : defs.joint_range.value_or("");
            if (!rng_str.empty()) {
                std::istringstream ss(rng_str);
                double lo, hi; ss >> lo >> hi;
                if (degrees) { lo *= M_PI/180.0; hi *= M_PI/180.0; }
                jt.limits.lower      = lo;
                jt.limits.upper      = hi;
                jt.limits.has_limits = true;
            } else {
                jt.limits.has_limits = false;
            }
        }

        {
            auto get_attr_d = [&](const char* a, std::optional<double> def_val) -> double {
                const char* v = jel->Attribute(a);
                if (v) return std::stod(v);
                return def_val.value_or(0.0);
            };
            double damping  = get_attr_d("damping",      defs.joint_damping);
            double friction = get_attr_d("frictionloss", defs.joint_frictionloss);
            double stiff    = get_attr_d("stiffness",    defs.joint_stiffness);
            jt.limits.limit_stiffness = stiff;
            jt.limits.limit_damping   = damping + friction;
        }

        jt.T_parent_joint = T_parent_joint;
        jt.child_link     = child_link;
        return jt;
    }

    // ── Inertial ──────────────────────────────────────────────────────────────
    static void parse_inertial(tinyxml2::XMLElement* in, LinkInertial& li) {
        if (auto* v = in->Attribute("mass"))
            li.mass = std::stod(v);
        if (auto* v = in->Attribute("pos"))
            li.com = parse_vec3_str(v);
        if (auto* v = in->Attribute("fullinertia")) {
            std::istringstream ss(v);
            double ixx,iyy,izz,ixy,ixz,iyz;
            ss >> ixx >> iyy >> izz >> ixy >> ixz >> iyz;
            li.inertia << ixx, ixy, ixz,
                          ixy, iyy, iyz,
                          ixz, iyz, izz;
        } else if (auto* v = in->Attribute("diaginertia")) {
            std::istringstream ss(v);
            double ixx, iyy, izz; ss >> ixx >> iyy >> izz;
            li.inertia = Vec3(ixx, iyy, izz).asDiagonal();
        }
    }

    // ── Body transform: pos + quat/euler/axisangle ────────────────────────────
    static T_mat parse_body_transform(tinyxml2::XMLElement* el, bool degrees) {
        Vec3 pos = Vec3::Zero();
        Mat3 R   = Mat3::Identity();

        if (auto* v = el->Attribute("pos"))
            pos = parse_vec3_str(v);

        if (auto* v = el->Attribute("quat")) {
            std::istringstream ss(v);
            double w, x, y, z; ss >> w >> x >> y >> z;
            Eigen::Quaterniond q(w, x, y, z); q.normalize();
            R = q.toRotationMatrix();
        } else if (auto* v = el->Attribute("euler")) {
            Vec3 rpy = parse_vec3_str(v);
            if (degrees) rpy *= M_PI / 180.0;
            R = rpy_to_rot(rpy);
        } else if (auto* v = el->Attribute("axisangle")) {
            std::istringstream ss(v);
            double ax, ay, az, ang; ss >> ax >> ay >> az >> ang;
            if (degrees) ang *= M_PI / 180.0;
            R = Eigen::AngleAxisd(ang, Vec3(ax,ay,az).normalized())
                    .toRotationMatrix();
        }

        return T_from_Rp(R, pos);
    }

    // ── Vec3 string helper ────────────────────────────────────────────────────
    static Vec3 parse_vec3_str(const std::string& s) {
        std::istringstream ss(s);
        double x=0, y=0, z=0; ss >> x >> y >> z;
        return {x, y, z};
    }
};

} // namespace lgn
