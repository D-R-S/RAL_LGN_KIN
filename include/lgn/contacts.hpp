// ============================================================================
//  lgn/contacts.hpp  —  Rigid-body contact detection and resolution
//
//  General — works for any KinematicTree (hands, humanoids, quadrupeds).
//
//  Collision geometry primitives: Sphere, Capsule, Box (on robot links)
//  Static world geometry:         StaticPlane
//  Free rigid bodies:             FreeSphere  ← NEW
//
//  Two contact solvers:
//    resolve_soft    — penalty spring, no LCP. Fast, for training rollouts.
//    resolve_lcp     — Gauss-Seidel LCP with friction. Physically correct.
//
//  FreeSphere:
//    A dynamic sphere not attached to the robot tree. Has its own position,
//    velocity, mass. Integrated by step() each timestep. Collides with
//    robot link colliders and static planes. Designed for:
//      - dropping a ball onto the hand / ground (demo, test)
//      - object manipulation in the sim tool
//      - training rollout scenes without full rigid-body object support
// ============================================================================
#pragma once
#include "kinematic_tree.hpp"
#include <variant>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>

namespace lgn {

// ============================================================================
//  Collision geometry primitives
// ============================================================================

struct Sphere  { double radius; };
struct Capsule { double radius; double half_length; };  // axis = local z
struct Box     { Vec3 half_extents; };

using CollisionShape = std::variant<Sphere, Capsule, Box>;

// A collision element attached to a robot link
struct Collider {
    std::string    link_name;
    int            link_idx{-1};
    CollisionShape shape;
    T_mat          T_link_shape{Mat4::Identity()};
};

// Static world plane:  normal · x = offset
struct StaticPlane {
    Vec3   normal{Vec3::UnitY()};
    double offset{0.0};
};

// ============================================================================
//  FreeSphere — a dynamic sphere not attached to the robot tree
//  Integrates its own position/velocity, collides with everything.
// ============================================================================
struct FreeSphere {
    Vec3   pos   {Vec3::Zero()};    // world position of centre
    Vec3   vel   {Vec3::Zero()};    // world velocity
    double radius{0.05};
    double mass  {0.1};             // kg
    double restitution{0.3};        // coefficient of restitution (bounciness)
    std::string name{"ball"};

    /// Simple symplectic Euler integration with gravity.
    /// Call once per timestep before contact detection.
    void step(double dt, const Vec3& g = {0, -9.81, 0}) {
        vel += dt * g;              // gravity
        pos += dt * vel;            // position update
    }

    /// Apply an impulse (collision response). Modifies vel directly.
    void apply_impulse(const Vec3& j) { vel += j / mass; }
};

// ============================================================================
//  Contact point (result of detection)
// ============================================================================
struct ContactPoint {
    int    link_a{-1};        // robot link index (-1 = static or free sphere)
    int    link_b{-1};        // robot link index (-1 = static or free sphere)
    int    free_sphere_idx{-1}; // index into ContactWorld::free_spheres_ (-1 = N/A)
    Vec3   point_w;           // contact point in world frame
    Vec3   normal_w;          // contact normal pointing from A toward B
    double penetration{0.0};  // > 0 means overlap

    // Filled by resolve_*
    Vec3   force_w{Vec3::Zero()};  // estimated contact force at point_w
    Eigen::MatrixXd Jc_a;
    Eigen::MatrixXd Jc_b;
};

// ============================================================================
//  Distance primitives (world frame)
// ============================================================================
namespace detail {

inline double sphere_plane(const Vec3& c, double r,
                            const Vec3& n, double d,
                            Vec3& cp, Vec3& nm) {
    nm = n;
    cp = c - n * r;
    return n.dot(c) - d - r;
}

inline double sphere_sphere(const Vec3& ca, double ra,
                             const Vec3& cb, double rb,
                             Vec3& cp, Vec3& nm) {
    Vec3 diff = cb - ca;
    double dist = diff.norm();
    if (dist < 1e-9) { nm = Vec3::UnitY(); cp = ca; return -(ra+rb); }
    nm = diff / dist;
    cp = ca + nm * ra;
    return dist - ra - rb;
}

inline double capsule_plane(const Vec3& c, const Vec3& ax,
                             double r, double hl,
                             const Vec3& n, double d,
                             Vec3& cp, Vec3& nm) {
    Vec3 p1 = c + ax*hl, p2 = c - ax*hl;
    double d1 = n.dot(p1)-d-r, d2 = n.dot(p2)-d-r;
    nm = n;
    if (d1 < d2) { cp = p1-n*r; return d1; }
    else         { cp = p2-n*r; return d2; }
}

inline double box_plane(const Vec3& c, const Mat3& R, const Vec3& he,
                         const Vec3& n, double d,
                         Vec3& cp, Vec3& nm) {
    Vec3 ln = R.transpose() * n;
    Vec3 sl(ln.x()>0?-he.x():he.x(),
            ln.y()>0?-he.y():he.y(),
            ln.z()>0?-he.z():he.z());
    cp = c + R*sl; nm = n;
    return n.dot(cp) - d;
}

} // namespace detail

// ============================================================================
//  ContactWorld
// ============================================================================
class ContactWorld {
public:

    // ── Setup ─────────────────────────────────────────────────────────────────
    // Set the simulation timestep used for free-sphere impulse in resolve_soft.
    // Default 1/60 s (60 Hz). Set to your actual sim dt before calling resolve_soft.
    void set_timestep(double dt) { dt_ = dt; }
    double timestep() const { return dt_; }
    void add_collider(const KinematicTree& tree,
                       const std::string& link_name,
                       CollisionShape shape,
                       const T_mat& T_link_shape = Mat4::Identity()) {
        Collider c;
        c.link_name    = link_name;
        c.link_idx     = tree.link_index(link_name);
        c.shape        = std::move(shape);
        c.T_link_shape = T_link_shape;
        colliders_.push_back(std::move(c));
    }

    void add_static_plane(const Vec3& normal, double offset) {
        planes_.push_back({ normal.normalized(), offset });
    }

    /// Add a free dynamic sphere to the world.
    /// Returns index into free_spheres_ for later access.
    int add_free_sphere(const FreeSphere& s) {
        free_spheres_.push_back(s);
        return (int)free_spheres_.size() - 1;
    }

    FreeSphere&       free_sphere(int i)       { return free_spheres_[i]; }
    const FreeSphere& free_sphere(int i) const { return free_spheres_[i]; }
    int n_free_spheres() const { return (int)free_spheres_.size(); }

    /// Step all free spheres (gravity integration).
    /// Call this BEFORE detect() each timestep.
    void step_free_spheres(double dt, const Vec3& g = {0, -9.81, 0}) {
        for (auto& s : free_spheres_) s.step(dt, g);
    }

    // ── Detection ─────────────────────────────────────────────────────────────

    /// Detect all contacts. tree.fk(q) must have been called first.
    /// Also detects free sphere collisions with robot links and planes.
    std::vector<ContactPoint> detect(const KinematicTree& tree,
                                      double tolerance = 0.0) const {
        std::vector<ContactPoint> contacts;

        // ── Robot colliders vs. static planes ────────────────────────────────
        for (const auto& col : colliders_) {
            T_mat Tw = tree.link(col.link_idx).T_world * col.T_link_shape;
            Vec3 ctr = p_of(Tw); Mat3 R = R_of(Tw);
            for (const auto& pl : planes_) {
                Vec3 cp, nm; double dist = 1e9;
                std::visit([&](auto&& s) {
                    using S = std::decay_t<decltype(s)>;
                    if constexpr (std::is_same_v<S,Sphere>)
                        dist = detail::sphere_plane(ctr,s.radius,pl.normal,pl.offset,cp,nm);
                    else if constexpr (std::is_same_v<S,Capsule>)
                        dist = detail::capsule_plane(ctr,R*Vec3::UnitZ(),s.radius,
                                                     s.half_length,pl.normal,pl.offset,cp,nm);
                    else if constexpr (std::is_same_v<S,Box>)
                        dist = detail::box_plane(ctr,R,s.half_extents,pl.normal,pl.offset,cp,nm);
                }, col.shape);
                if (dist <= tolerance) {
                    ContactPoint c;
                    c.link_a      = col.link_idx; c.link_b = -1;
                    c.point_w     = cp; c.normal_w = nm;
                    c.penetration = -dist;
                    contacts.push_back(c);
                }
            }
        }

        // ── Robot colliders vs. robot colliders (sphere-sphere only) ──────────
        for (size_t a = 0; a < colliders_.size(); ++a) {
            for (size_t b = a+1; b < colliders_.size(); ++b) {
                const auto& ca = colliders_[a];
                const auto& cb = colliders_[b];
                if (ca.link_idx == cb.link_idx) continue;
                // A3e — skip excluded pairs (parent-child and explicit exclusions)
                if (tree.is_excluded(ca.link_idx, cb.link_idx)) continue;
                
                if (!std::holds_alternative<Sphere>(ca.shape)) continue;
                if (!std::holds_alternative<Sphere>(cb.shape)) continue;
                T_mat Ta = tree.link(ca.link_idx).T_world * ca.T_link_shape;
                T_mat Tb = tree.link(cb.link_idx).T_world * cb.T_link_shape;
                double ra = std::get<Sphere>(ca.shape).radius;
                double rb = std::get<Sphere>(cb.shape).radius;
                Vec3 cp, nm;
                double dist = detail::sphere_sphere(p_of(Ta),ra,p_of(Tb),rb,cp,nm);
                if (dist <= tolerance) {
                    ContactPoint c;
                    c.link_a=ca.link_idx; c.link_b=cb.link_idx;
                    c.point_w=cp; c.normal_w=nm; c.penetration=-dist;
                    contacts.push_back(c);
                }
            }
        }

        // ── Free spheres vs. static planes ────────────────────────────────────
        for (int si = 0; si < (int)free_spheres_.size(); ++si) {
            const FreeSphere& fs = free_spheres_[si];
            for (const auto& pl : planes_) {
                Vec3 cp, nm;
                double dist = detail::sphere_plane(
                    fs.pos, fs.radius, pl.normal, pl.offset, cp, nm);
                if (dist <= tolerance) {
                    ContactPoint c;
                    c.link_a=-1; c.link_b=-1;
                    c.free_sphere_idx = si;
                    c.point_w=cp; c.normal_w=nm; c.penetration=-dist;
                    contacts.push_back(c);
                }
            }
        }

        // ── Free spheres vs. robot link colliders ─────────────────────────────
        for (int si = 0; si < (int)free_spheres_.size(); ++si) {
            const FreeSphere& fs = free_spheres_[si];
            for (const auto& col : colliders_) {
                if (!std::holds_alternative<Sphere>(col.shape)) continue;
                T_mat Tw = tree.link(col.link_idx).T_world * col.T_link_shape;
                double rr = std::get<Sphere>(col.shape).radius;
                Vec3 cp, nm;
                double dist = detail::sphere_sphere(
                    fs.pos, fs.radius, p_of(Tw), rr, cp, nm);
                if (dist <= tolerance) {
                    ContactPoint c;
                    c.link_a      = col.link_idx;
                    c.link_b      = -1;
                    c.free_sphere_idx = si;
                    c.point_w     = cp;
                    c.normal_w    = nm;   // points from robot link toward ball
                    c.penetration = -dist;
                    contacts.push_back(c);
                }
            }
        }

        return contacts;
    }

    // ── Contact Jacobian ──────────────────────────────────────────────────────

    Eigen::MatrixXd contact_jacobian(const KinematicTree& tree,
                                      const Eigen::VectorXd& q,
                                      int link_idx,
                                      const Vec3& point_w) const {
        if (link_idx < 0) return Eigen::MatrixXd::Zero(3, tree.n_dof());
        std::vector<int> dof_idx;
        Eigen::MatrixXd Jc_chain = tree.jacobian_chain(q, link_idx, dof_idx);
        Vec3 p_link = p_of(tree.link(link_idx).T_world);
        Vec3 delta  = point_w - p_link;
        Eigen::MatrixXd Jpt = Eigen::MatrixXd::Zero(3, tree.n_dof());
        for (int c = 0; c < (int)dof_idx.size(); ++c) {
            int di = dof_idx[c];
            Vec3 z_w = Jc_chain.col(c).tail<3>();
            Vec3 jl  = Jc_chain.col(c).head<3>();
            Jpt.col(di) = (z_w.norm()>1e-9) ? jl + z_w.cross(delta) : jl;
        }
        return Jpt;
    }

    // ── Soft contact resolution ───────────────────────────────────────────────

    /// Returns generalised contact force Jcᵀ·λ for the robot DOFs.
    /// Also resolves free-sphere collisions via impulse (modifies free_spheres_).
    Eigen::VectorXd resolve_soft(
        std::vector<ContactPoint>& contacts,
        const KinematicTree& tree,
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& dq,
        double kn = 5e4,
        double kd = 100.0)
    {
        Eigen::VectorXd lambda_Jc = Eigen::VectorXd::Zero(tree.n_dof());

        for (auto& cp : contacts) {
            // ── Free sphere vs. plane (impulse-based) ────────────────────────
            if (cp.free_sphere_idx >= 0 && cp.link_a < 0) {
                FreeSphere& fs = free_spheres_[cp.free_sphere_idx];
                double v_n = fs.vel.dot(cp.normal_w);
                if (v_n < 0) {
                    // Restitution impulse
                    double j = -(1.0 + fs.restitution) * v_n * fs.mass;
                    fs.apply_impulse(j * cp.normal_w);
                }
                // Positional correction (push out of plane)
                fs.pos += cp.penetration * cp.normal_w;
                cp.force_w = kn * cp.penetration * cp.normal_w;
                continue;
            }

            // ── Free sphere vs. robot link ────────────────────────────────────
            if (cp.free_sphere_idx >= 0 && cp.link_a >= 0) {
                FreeSphere& fs = free_spheres_[cp.free_sphere_idx];
                // Contact Jacobian for robot link at contact point
                Eigen::MatrixXd Jca = contact_jacobian(tree, q, cp.link_a, cp.point_w);
                cp.Jc_a = Jca;

                // Relative velocity at contact: v_ball - v_link
                Vec3 v_link(0,0,0);
                if (q.size() == tree.n_dof() && dq.size() == tree.n_dof())
                    v_link = Jca * dq;  // linear vel of contact point on link

                Vec3 v_rel = fs.vel - v_link;
                double v_n = v_rel.dot(cp.normal_w);

                // Penalty spring on ball
                Vec3 f_contact = kn * cp.penetration * cp.normal_w;
                if (v_n < 0) f_contact -= kd * v_n * cp.normal_w;

                // Impulse on ball (Newton 3rd: reaction on link)
                fs.apply_impulse(f_contact * dt_);  // configurable timestep

                // Reaction generalised force on robot
                lambda_Jc += Jca.transpose() * (-f_contact);
                cp.force_w = f_contact;
                continue;
            }

            // ── Robot vs. plane / robot vs. robot ─────────────────────────────
            Eigen::MatrixXd Jca = contact_jacobian(tree, q, cp.link_a, cp.point_w);
            Eigen::MatrixXd Jcb = contact_jacobian(tree, q, cp.link_b, cp.point_w);
            Eigen::MatrixXd Jrel = Jca - Jcb;
            cp.Jc_a = Jca; cp.Jc_b = Jcb;

            double f_n = kn * cp.penetration;
            double v_n = (Jrel * dq).dot(cp.normal_w);
            if (v_n < 0) f_n -= kd * v_n;

            Vec3 f_world = f_n * cp.normal_w;
            lambda_Jc += Jrel.transpose() * f_world;
            cp.force_w = f_world;
        }
        return lambda_Jc;
    }

    // ── Gauss-Seidel LCP ─────────────────────────────────────────────────────
    // (robot DOFs only — free spheres are handled via impulse in resolve_soft)

    Eigen::VectorXd resolve_lcp(
        std::vector<ContactPoint>& contacts,
        const KinematicTree& tree,
        const Eigen::VectorXd& q,
        const Eigen::VectorXd& dq,
        const Eigen::MatrixXd& M,
        const Eigen::VectorXd& tau_minus_CG,
        double dt,
        double mu = 0.5,
        int max_iter = 50,
        double tol = 1e-6) const
    {
        int n = tree.n_dof();
        // Filter to robot-only contacts (free sphere contacts handled elsewhere)
        std::vector<ContactPoint*> robot_contacts;
        for (auto& cp : contacts)
            if (cp.free_sphere_idx < 0 && cp.link_a >= 0)
                robot_contacts.push_back(&cp);

        int nc = (int)robot_contacts.size();
        if (nc == 0) return Eigen::VectorXd::Zero(n);

        for (auto* cp : robot_contacts) {
            cp->Jc_a = contact_jacobian(tree, q, cp->link_a, cp->point_w);
            cp->Jc_b = contact_jacobian(tree, q, cp->link_b, cp->point_w);
        }

        Eigen::MatrixXd Jc(nc, n);
        for (int i = 0; i < nc; ++i) {
            Eigen::MatrixXd Jrel = robot_contacts[i]->Jc_a
                                 - robot_contacts[i]->Jc_b;
            Jc.row(i) = robot_contacts[i]->normal_w.transpose() * Jrel;
        }

        Eigen::MatrixXd Minv = M.ldlt().solve(
            Eigen::MatrixXd::Identity(n,n));
        Eigen::MatrixXd A = Jc * Minv * Jc.transpose();
        Eigen::VectorXd dq_free = dq + dt * Minv * tau_minus_CG;
        Eigen::VectorXd b = Jc * dq_free;

        Eigen::VectorXd lambda = Eigen::VectorXd::Zero(nc);
        for (int iter = 0; iter < max_iter; ++iter) {
            double max_delta = 0.0;
            for (int i = 0; i < nc; ++i) {
                double Aii = A(i,i);
                if (std::abs(Aii) < 1e-12) continue;
                double wi = b[i] + (A.row(i)*lambda)(0) - A(i,i)*lambda[i];
                double ln = std::max(0.0, -wi/Aii);
                max_delta = std::max(max_delta, std::abs(ln-lambda[i]));
                lambda[i] = ln;
            }
            if (max_delta < tol) break;
        }
        return Jc.transpose() * lambda;
    }

    // ── Virtual sensor readings ───────────────────────────────────────────────

    struct VirtualContactReading {
        std::string link_name;
        Vec3        force_world;
        Vec3        point_world;
        double      normal_force;
    };

    std::vector<VirtualContactReading> virtual_sensor_readings(
        const std::vector<ContactPoint>& contacts,
        const KinematicTree& tree,
        const Eigen::VectorXd& /*lambda_Jc*/,
        const Eigen::VectorXd& /*q*/) const
    {
        std::vector<VirtualContactReading> out;
        for (const auto& cp : contacts) {
            if (cp.link_a < 0) continue;
            VirtualContactReading r;
            r.link_name    = tree.link(cp.link_a).name;
            r.point_world  = cp.point_w;
            r.force_world  = cp.force_w;
            r.normal_force = cp.force_w.norm();
            out.push_back(r);
        }
        return out;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    const std::vector<Collider>&    colliders()    const { return colliders_; }
    const std::vector<StaticPlane>& planes()       const { return planes_; }
    const std::vector<FreeSphere>&  free_spheres() const { return free_spheres_; }

private:
    std::vector<Collider>    colliders_;
    std::vector<StaticPlane> planes_;
    std::vector<FreeSphere>  free_spheres_;
    double dt_{1.0/60.0};
};

} // namespace lgn
// Note: load_collision_geometry() — URDF <collision> parser — lives in
// collision_loader.hpp, which is the only file that needs tinyxml2.
// contacts.hpp has no XML dependency.
