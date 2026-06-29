// ============================================================================
//  lgn/kinematic_tree.hpp  —  Patch 1.4
//
//  Patch 1.4 changes vs 1.2:
//   - Added chain_dof_indices(tip_link, dof_indices): walks the joint path
//     and emits the DOF index list WITHOUT touching the Jacobian matrix.
//     Used by IKSolver/HandIKSolver result-merge to avoid the wasteful
//     "build full Jacobian to throw it away" pattern.
//
//  Patch 1.2 (preserved):
//   - jacobian_chain_cached(tip, dof_idx): reads T_world from the FK cache.
//   - q_cached_ and cache_valid_for(q) for debug-build contract checks.
// ============================================================================
#pragma once
#include "core.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace lgn {

class KinematicTree;

enum class JointType {
    Fixed, Revolute, Continuous, Prismatic, Floating, Planar, Ball
};

// CPP magic ... well not. Prealocation makes cpu go brrrrrrr ... E.S. <3
using JacobianMatrix = Eigen::Matrix<double, 6, Eigen::Dynamic, 0, 6, 256>;

inline int joint_dof_count(JointType t) {
    switch (t) {
        case JointType::Fixed:      return 0;
        case JointType::Revolute:
        case JointType::Continuous:
        case JointType::Prismatic:  return 1;
        case JointType::Floating:   return 6;
        case JointType::Planar:     return 3;
        case JointType::Ball:       return 3;
    }
    return 0;
}

struct JointLimits {
    double lower   {-M_PI};
    double upper   { M_PI};
    double velocity{10.0};
    double effort  {10.0};
    bool   has_limits{false};
    double limit_stiffness{0.0};
    double limit_damping  {0.0};
};

struct LinkInertial {
    double   mass   {0.0};
    Vec3     com    {Vec3::Zero()};
    Mat3     inertia{Mat3::Zero()};
    Gam_mat  gamma  {Mat4::Zero()};
};

struct Joint {
    std::string  name;
    JointType    type       {JointType::Fixed};
    int          dof_index  {-1};
    int          dof_count  {0};
    T_mat        T_parent_joint{Mat4::Identity()};
    Vec3         axis       {Vec3::UnitZ()};
    JointLimits  limits;
    int          child_link {-1};
};

struct Link {
    std::string       name;
    int               parent_joint{-1};
    std::vector<int>  child_joints;
    LinkInertial      inertial;

    T_mat   T_world {Mat4::Identity()};
    Hv_mat  Hv_world{Mat4::Zero()};
    Ha_mat  Ha_world{Mat4::Zero()};
};

struct LoopConstraint {
    std::string name;
    int   link_a    {-1};
    T_mat T_link_a  {Mat4::Identity()};
    int   link_b    {-1};
    T_mat T_link_b  {Mat4::Identity()};
    Vec3 eval_error(const KinematicTree& tree) const;
};

class KinematicTree {
public:
    int  add_link (const Link&  l) { links_.push_back(l);  return (int)links_.size()-1; }
    int  add_joint(const Joint& j) { joints_.push_back(j); return (int)joints_.size()-1; }
    void set_root      (int i) { root_  = i; }
    void set_dof_count (int n) { n_dof_ = n; }
    void set_dof_names (const std::vector<std::string>& n) { dof_names_ = n; }

    void add_loop_constraint(const LoopConstraint& lc) { loop_constraints_.push_back(lc); }
    const std::vector<LoopConstraint>& loop_constraints() const { return loop_constraints_; }

    void finalize() {
        link_map_.clear(); joint_map_.clear(); tips_.clear();
        exclusion_set_.clear();
        for (int i=0;i<(int)links_.size(); ++i)  link_map_[links_[i].name]  = i;
        for (int i=0;i<(int)joints_.size();++i)  joint_map_[joints_[i].name]= i;
        for (int i=0;i<(int)links_.size(); ++i)
            if (links_[i].child_joints.empty()) tips_.push_back(i);
        for (auto& lk : links_)
            if (lk.inertial.mass > 1e-9)
                lk.inertial.gamma = Gamma(lk.inertial.inertia,
                                           lk.inertial.com,
                                           lk.inertial.mass);
        for (int ji = 0; ji < (int)joints_.size(); ++ji) {
            const Joint& jt = joints_[ji];
            if (jt.child_link < 0) continue;
            for (int li = 0; li < (int)links_.size(); ++li) {
                for (int owned_ji : links_[li].child_joints) {
                    if (owned_ji == ji) { add_exclusion(li, jt.child_link); break; }
                }
            }
        }
    }

    int n_dof()   const { return n_dof_; }
    int n_links() const { return (int)links_.size(); }
    int n_joints()const { return (int)joints_.size(); }
    int root()    const { return root_; }

    const Link&  link (int i) const { return links_[i]; }
          Link&  link (int i)       { return links_[i]; }
    const Joint& joint(int i) const { return joints_[i]; }
          Joint& joint(int i)       { return joints_[i]; }

    const std::vector<Link>&  links_ref()  const { return links_; }
    const std::vector<Joint>& joints_ref() const { return joints_; }

    int link_index(const std::string& name) const {
        auto it = link_map_.find(name);
        if (it == link_map_.end()) throw std::runtime_error("Link not found: " + name);
        return it->second;
    }
    int joint_index(const std::string& name) const {
        auto it = joint_map_.find(name);
        if (it == joint_map_.end()) throw std::runtime_error("Joint not found: " + name);
        return it->second;
    }

    const std::vector<std::string>& dof_names() const { return dof_names_; }
    const std::vector<int>&         tips()      const { return tips_; }

    void add_exclusion(int link_a, int link_b) {
        int lo = std::min(link_a, link_b);
        int hi = std::max(link_a, link_b);
        exclusion_set_.insert((static_cast<int64_t>(lo) << 32) | hi);
    }
    void add_exclusion(const std::string& a, const std::string& b) {
        add_exclusion(link_index(a), link_index(b));
    }
    bool is_excluded(int link_a, int link_b) const {
        int lo = std::min(link_a, link_b);
        int hi = std::max(link_a, link_b);
        return exclusion_set_.count((static_cast<int64_t>(lo) << 32) | hi) > 0;
    }

    T_mat joint_T(const Joint& j, const Eigen::VectorXd& q) const {
        switch (j.type) {
            case JointType::Fixed: return j.T_parent_joint;
            case JointType::Revolute:
            case JointType::Continuous: {
                double theta = (j.dof_index >= 0) ? q[j.dof_index] : 0.0;
                return j.T_parent_joint * T_rot(j.axis, theta);
            }
            case JointType::Prismatic: {
                double d = (j.dof_index >= 0) ? q[j.dof_index] : 0.0;
                return j.T_parent_joint * T_trans(d * j.axis);
            }
            case JointType::Floating: {
                if (j.dof_index < 0) return j.T_parent_joint;
                Vec3 t   = q.segment<3>(j.dof_index);
                Vec3 rpy = q.segment<3>(j.dof_index + 3);
                Mat3 R = (Eigen::AngleAxisd(rpy.z(), Vec3::UnitZ())
                        * Eigen::AngleAxisd(rpy.y(), Vec3::UnitY())
                        * Eigen::AngleAxisd(rpy.x(), Vec3::UnitX()))
                         .toRotationMatrix();
                return j.T_parent_joint * T_from_Rp(R, t);
            }
            case JointType::Planar: {
                if (j.dof_index < 0) return j.T_parent_joint;
                Vec3 t(q[j.dof_index], q[j.dof_index+1], 0.0);
                double rz = q[j.dof_index + 2];
                Mat3 R = Eigen::AngleAxisd(rz, Vec3::UnitZ()).toRotationMatrix();
                return j.T_parent_joint * T_from_Rp(R, t);
            }
            case JointType::Ball: {
                if (j.dof_index < 0) return j.T_parent_joint;
                double rz = q[j.dof_index + 0];
                double ry = q[j.dof_index + 1];
                double rx = q[j.dof_index + 2];
                Mat3 R = (Eigen::AngleAxisd(rz, Vec3::UnitZ())
                        * Eigen::AngleAxisd(ry, Vec3::UnitY())
                        * Eigen::AngleAxisd(rx, Vec3::UnitX()))
                         .toRotationMatrix();
                return j.T_parent_joint * T_from_Rp(R, Vec3::Zero());
            }
        }
        return j.T_parent_joint;
    }

    Hv_mat joint_Hv_rel(const Joint& j, const Eigen::VectorXd& dq) const {
        switch (j.type) {
            case JointType::Fixed: return Mat4::Zero();
            case JointType::Revolute:
            case JointType::Continuous: {
                double dqv = (j.dof_index >= 0) ? dq[j.dof_index] : 0.0;
                return Hv_revolute(j.axis, dqv);
            }
            case JointType::Prismatic: {
                double dqv = (j.dof_index >= 0) ? dq[j.dof_index] : 0.0;
                return Hv_prismatic(j.axis, dqv);
            }
            case JointType::Floating: {
                if (j.dof_index < 0) return Mat4::Zero();
                Vec3 v = dq.segment<3>(j.dof_index);
                Vec3 w = dq.segment<3>(j.dof_index + 3);
                return Hv_from(w, v);
            }
            case JointType::Planar: {
                if (j.dof_index < 0) return Mat4::Zero();
                Vec3 v(dq[j.dof_index], dq[j.dof_index+1], 0.0);
                Vec3 w(0.0, 0.0, dq[j.dof_index+2]);
                return Hv_from(w, v);
            }
            case JointType::Ball: {
                if (j.dof_index < 0) return Mat4::Zero();
                Vec3 omega = dq[j.dof_index+0] * Vec3::UnitZ()
                           + dq[j.dof_index+1] * Vec3::UnitY()
                           + dq[j.dof_index+2] * Vec3::UnitX();
                return Hv_from(omega, Vec3::Zero());
            }
        }
        return Mat4::Zero();
    }

    void fk(const Eigen::VectorXd& q) {
        links_[root_].T_world = T_identity();
        std::vector<int> stack = {root_};
        while (!stack.empty()) {
            int li = stack.back(); stack.pop_back();
            for (int ji : links_[li].child_joints) {
                const Joint& jt = joints_[ji];
                links_[jt.child_link].T_world =
                    links_[li].T_world * joint_T(jt, q);
                stack.push_back(jt.child_link);
            }
        }
        q_cached_ = q;
        cache_valid_ = true;
    }

    T_mat fk_tip(const Eigen::VectorXd& q, int tip_link) const {
        std::vector<int> jp;
        path_to(tip_link, jp);
        T_mat T = T_identity();
        for (int ji : jp) T = T * joint_T(joints_[ji], q);
        return T;
    }

    void velocity_propagation(const Eigen::VectorXd& q,
                               const Eigen::VectorXd& dq) {
        links_[root_].Hv_world = Mat4::Zero();
        std::vector<int> stack = {root_};
        while (!stack.empty()) {
            int li = stack.back(); stack.pop_back();
            for (int ji : links_[li].child_joints) {
                const Joint& jt = joints_[ji];
                int cl = jt.child_link;
                const T_mat& T_child = links_[cl].T_world;
                links_[cl].Hv_world = Hv_propagate(
                    links_[li].Hv_world, T_child, joint_Hv_rel(jt, dq));
                stack.push_back(cl);
            }
        }
    }

    JacobianMatrix jacobian(const Eigen::VectorXd& q, int tip_link) const {
        std::vector<int> dof_idx;
        JacobianMatrix Jc = jacobian_chain(q, tip_link, dof_idx);
        JacobianMatrix J  = JacobianMatrix::Zero(6, n_dof_);
        for (int c=0; c<(int)dof_idx.size(); ++c)
            J.col(dof_idx[c]) = Jc.col(c);
        return J;
    }

    JacobianMatrix jacobian_chain(const Eigen::VectorXd& q, int tip_link,
                                   std::vector<int>& dof_indices) const {
        std::vector<int> joint_path;
        path_to(tip_link, joint_path);

        T_mat T_tip = fk_tip(q, tip_link);
        Vec3  p_tip = p_of(T_tip);

        dof_indices.clear();
        for (int ji : joint_path) {
            const Joint& jt = joints_[ji];
            for (int d = 0; d < jt.dof_count; ++d)
                if (jt.dof_index >= 0) dof_indices.push_back(jt.dof_index + d);
        }

        int nc = (int)dof_indices.size();
        JacobianMatrix J = JacobianMatrix::Zero(6, nc);

        T_mat T_cur = T_identity();
        int col = 0;
        for (int ji : joint_path) {
            const Joint& jt = joints_[ji];
            T_cur = T_cur * joint_T(jt, q);
            if (jt.dof_index < 0) continue;

            Vec3 p_i = p_of(T_cur);
            Mat3 R_i = R_of(T_cur);
            fill_jacobian_columns(jt, p_i, R_i, p_tip, q, J, col);
        }
        return J;
    }

    JacobianMatrix jacobian_chain_cached(int tip_link,
                                          std::vector<int>& dof_indices) const {
        std::vector<int> joint_path;
        path_to(tip_link, joint_path);

        Vec3 p_tip = p_of(links_[tip_link].T_world);

        dof_indices.clear();
        for (int ji : joint_path) {
            const Joint& jt = joints_[ji];
            for (int d = 0; d < jt.dof_count; ++d)
                if (jt.dof_index >= 0) dof_indices.push_back(jt.dof_index + d);
        }

        int nc = (int)dof_indices.size();
        JacobianMatrix J = JacobianMatrix::Zero(6, nc);

        int col = 0;
        for (int ji : joint_path) {
            const Joint& jt = joints_[ji];
            if (jt.dof_index < 0) continue;

            const T_mat& T_i = links_[jt.child_link].T_world;
            Vec3 p_i = p_of(T_i);
            Mat3 R_i = R_of(T_i);

            fill_jacobian_columns(jt, p_i, R_i, p_tip, q_cached_, J, col);
        }
        return J;
    }

    // ── Patch 1.4 — DOF-only chain walk (no Jacobian matrix work) ────────────
    //
    //  Walks the joint path from root to tip_link and emits the DOF indices.
    //  Identical traversal to jacobian_chain but skips the matrix fill —
    //  for callers (e.g. HandIKSolver::solve_all merge) that only need to
    //  know WHICH DOFs lie on the chain, not their geometric Jacobian.
    //
    //  Cost: O(path_length) integer pushes. No Eigen allocation.
    //
    void chain_dof_indices(int tip_link, std::vector<int>& dof_indices) const {
        std::vector<int> joint_path;
        path_to(tip_link, joint_path);
        dof_indices.clear();
        for (int ji : joint_path) {
            const Joint& jt = joints_[ji];
            for (int d = 0; d < jt.dof_count; ++d)
                if (jt.dof_index >= 0) dof_indices.push_back(jt.dof_index + d);
        }
    }

    bool cache_valid_for(const Eigen::VectorXd& q) const {
        if (!cache_valid_) return false;
        if (q_cached_.size() != q.size()) return false;
        return (q_cached_ - q).cwiseAbs().maxCoeff() < 1e-15;
    }

    Eigen::VectorXd joint_limit_torque(const Eigen::VectorXd& q,
                                        const Eigen::VectorXd& dq) const {
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(n_dof_);
        for (const auto& jt : joints_) {
            if (jt.dof_index < 0) continue;
            if (!jt.limits.has_limits) continue;
            double k = jt.limits.limit_stiffness;
            double b = jt.limits.limit_damping;
            if (k < 1e-9 && b < 1e-9) continue;
            for (int d = 0; d < jt.dof_count; ++d) {
                int i = jt.dof_index + d;
                if (i >= (int)q.size()) break;
                double qi  = q[i];
                double dqi = (i < (int)dq.size()) ? dq[i] : 0.0;
                double lo  = jt.limits.lower;
                double hi  = jt.limits.upper;
                if (qi > hi)      tau[i] = -k * (qi - hi) - b * dqi;
                else if (qi < lo) tau[i] =  k * (lo - qi) - b * dqi;
            }
        }
        return tau;
    }

    Eigen::VectorXd clamp(const Eigen::VectorXd& q) const {
        Eigen::VectorXd qc = q;
        for (const auto& jt : joints_) {
            if (jt.dof_index < 0 || !jt.limits.has_limits) continue;
            for (int d = 0; d < jt.dof_count; ++d) {
                int i = jt.dof_index + d;
                if (i < (int)qc.size())
                    qc[i] = std::max(jt.limits.lower,
                                     std::min(jt.limits.upper, qc[i]));
            }
        }
        return qc;
    }

private:
    std::vector<Link>            links_;
    std::vector<Joint>           joints_;
    std::vector<LoopConstraint>  loop_constraints_;
    std::unordered_set<int64_t>  exclusion_set_;
    int  root_  {0};
    int  n_dof_ {0};
    std::vector<std::string> dof_names_;
    std::vector<int>         tips_;
    std::unordered_map<std::string,int> link_map_;
    std::unordered_map<std::string,int> joint_map_;

    Eigen::VectorXd q_cached_;
    bool            cache_valid_{false};

    bool path_to(int target, std::vector<int>& path) const {
        path.clear();
        return path_impl(root_, target, path);
    }
    bool path_impl(int cur, int target, std::vector<int>& path) const {
        if (cur == target) return true;
        for (int ji : links_[cur].child_joints) {
            path.push_back(ji);
            if (path_impl(joints_[ji].child_link, target, path)) return true;
            path.pop_back();
        }
        return false;
    }

    void fill_jacobian_columns(const Joint& jt,
                                 const Vec3& p_i, const Mat3& R_i,
                                 const Vec3& p_tip,
                                 const Eigen::VectorXd& q,
                                 JacobianMatrix& J, int& col) const
    {
        switch (jt.type) {
            case JointType::Revolute:
            case JointType::Continuous: {
                Vec3 z_w = R_i * jt.axis;
                J.col(col).head<3>() = z_w.cross(p_tip - p_i);
                J.col(col).tail<3>() = z_w;
                ++col;
                break;
            }
            case JointType::Prismatic: {
                Vec3 z_w = R_i * jt.axis;
                J.col(col).head<3>() = z_w;
                J.col(col).tail<3>() = Vec3::Zero();
                ++col;
                break;
            }
            case JointType::Floating: {
                J.col(col).head<3>() = Vec3::UnitX(); ++col;
                J.col(col).head<3>() = Vec3::UnitY(); ++col;
                J.col(col).head<3>() = Vec3::UnitZ(); ++col;
                for (int ax = 0; ax < 3; ++ax) {
                    Vec3 e = Vec3::Zero(); e[ax] = 1.0;
                    J.col(col).head<3>() = e.cross(p_tip - p_i);
                    J.col(col).tail<3>() = e;
                    ++col;
                }
                break;
            }
            case JointType::Planar: {
                J.col(col).head<3>() = Vec3::UnitX(); ++col;
                J.col(col).head<3>() = Vec3::UnitY(); ++col;
                Vec3 z_w = R_i * Vec3::UnitZ();
                J.col(col).head<3>() = z_w.cross(p_tip - p_i);
                J.col(col).tail<3>() = z_w; ++col;
                break;
            }
            case JointType::Ball: {
                double rz_a = 0.0, ry_a = 0.0, rx_a = 0.0;
                if (jt.dof_index >= 0 && (int)q.size() > jt.dof_index + 2) {
                    rz_a = q[jt.dof_index + 0];
                    ry_a = q[jt.dof_index + 1];
                    rx_a = q[jt.dof_index + 2];
                }
                Mat3 Rball = (Eigen::AngleAxisd(rz_a, Vec3::UnitZ())
                            * Eigen::AngleAxisd(ry_a, Vec3::UnitY())
                            * Eigen::AngleAxisd(rx_a, Vec3::UnitX()))
                             .toRotationMatrix();
                Mat3 R_before = R_i * Rball.transpose();
                Mat3 Rz_only  = Eigen::AngleAxisd(rz_a, Vec3::UnitZ()).toRotationMatrix();
                Mat3 Ry_only  = Eigen::AngleAxisd(ry_a, Vec3::UnitY()).toRotationMatrix();

                Vec3 axis_rz = R_before * Vec3::UnitZ();
                Vec3 axis_ry = R_before * Rz_only * Vec3::UnitY();
                Vec3 axis_rx = R_before * Rz_only * Ry_only * Vec3::UnitX();

                for (const Vec3& ax : {axis_rz, axis_ry, axis_rx}) {
                    J.col(col).head<3>() = ax.cross(p_tip - p_i);
                    J.col(col).tail<3>() = ax;
                    ++col;
                }
                break;
            }
            case JointType::Fixed: break;
        }
    }

    friend struct LoopConstraint;
};

inline Vec3 LoopConstraint::eval_error(const KinematicTree& tree) const {
    const T_mat& Twa = tree.links_[link_a].T_world;
    const T_mat& Twb = tree.links_[link_b].T_world;
    Vec3 pa = p_of(Twa * T_link_a);
    Vec3 pb = p_of(Twb * T_link_b);
    return pb - pa;
}

} // namespace lgn
