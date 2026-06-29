// ============================================================================
//  lgn/ik_solver.hpp  —  Patch 1.4
//
//  Per-tip IK solver using the Legnani geometric Jacobian.
//  Method: Damped Least Squares (DLS) with joint-limit null-space.
//
//  Patch 1.4 changes:
//   - IKSolver::solve now uses fk(q) + jacobian_chain_cached. One FK walk
//     per iteration; the Jacobian is read off the FK cache. Matches the
//     §2.3 claim in the paper.
//   - HandIKSolver::solve_all uses tree.chain_dof_indices() instead of
//     calling jacobian_chain just to throw the matrix away.
// ============================================================================
#pragma once
#include "kinematic_tree.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace lgn {

struct IKParams {
    double lambda_sq    = 1e-4;
    double pos_tol      = 1e-4;
    double rot_tol      = 1e-3;
    int    max_iter     = 50;
    double step_limit   = 0.2;
    bool   use_nullspace= true;
    double w_nullspace  = 0.05;
};

struct TaskError {
    Vec3 dp;
    Vec3 dR;
    double pos_norm() const { return dp.norm(); }
    double rot_norm() const { return dR.norm(); }
    bool   converged(const IKParams& p) const {
        return pos_norm() < p.pos_tol && rot_norm() < p.rot_tol;
    }
    Eigen::Matrix<double,6,1> as_vec() const {
        Eigen::Matrix<double,6,1> v;
        v << dp, dR;
        return v;
    }
};

inline TaskError se3_error(const T_mat& T_cur, const T_mat& T_des) {
    TaskError e;
    e.dp = p_of(T_des) - p_of(T_cur);
    Mat3 R_err = R_of(T_des) * R_of(T_cur).transpose();
    Eigen::AngleAxisd aa(R_err);
    e.dR = aa.angle() * aa.axis();
    return e;
}

// ============================================================================
//  IKSolver — one kinematic chain tip
// ============================================================================
class IKSolver {
public:
    IKSolver(KinematicTree& tree, const std::string& tip_name,
              IKParams params = {})
        : tree_(tree)
        , tip_link_(tree.link_index(tip_name))
        , p_(params)
    {}

    bool solve(const Eigen::VectorXd& q0,
               const T_mat& T_des,
               Eigen::VectorXd& q_out,
               int* iters_out = nullptr) const
    {
        Eigen::VectorXd q = q0;
        std::vector<int> dof_idx;

        for (int iter = 0; iter < p_.max_iter; ++iter) {
            // Patch 1.4: ONE FK walk per iteration. T_cur and the Jacobian
            // columns both come from the FK cache.
            tree_.fk(q);
            const T_mat& T_cur = tree_.link(tip_link_).T_world;
            TaskError err = se3_error(T_cur, T_des);

            if (err.converged(p_)) {
                if (iters_out) *iters_out = iter;
                q_out = q;
                return true;
            }

            //Eigen::MatrixXd Jc = tree_.jacobian_chain_cached(tip_link_, dof_idx);
            auto Jc = tree_.jacobian_chain_cached(tip_link_, dof_idx);
            int nc = (int)dof_idx.size();
            if (nc == 0) break;

            Eigen::Matrix<double,6,1> ex = err.as_vec();
            Eigen::VectorXd dqc = dls_step(Jc, ex, nc);

            double nrm = dqc.norm();
            if (nrm > p_.step_limit) dqc *= p_.step_limit / nrm;

            if (p_.use_nullspace && nc > 6) {
                Eigen::MatrixXd Jp = pseudoinverse(Jc);
                Eigen::MatrixXd N  = Eigen::MatrixXd::Identity(nc,nc) - Jp*Jc;
                dqc += p_.w_nullspace * N * jl_gradient(q, dof_idx);
            }

            for (int c = 0; c < nc; ++c)
                q[dof_idx[c]] += dqc[c];

            q = tree_.clamp(q);
        }

        if (iters_out) *iters_out = p_.max_iter;
        q_out = q;
        return false;
    }

    int             tip_link() const { return tip_link_; }
    const IKParams& params()   const { return p_; }
    void set_params(const IKParams& p) { p_ = p; }

private:
    KinematicTree& tree_;
    int            tip_link_;
    IKParams       p_;

    Eigen::VectorXd dls_step(const JacobianMatrix& J,
                             const Eigen::Matrix<double,6,1>& e,
                             int nc) const {
        if (nc > 6) { // Redundant manipulator: J is fat. J*J^T is 6x6.
            Eigen::Matrix<double,6,6> A = J * J.transpose();
            A.diagonal().array() += p_.lambda_sq;
            return J.transpose() * A.ldlt().solve(e);
        } else {      // Under-actuated/Square: J is tall. J^T*J is nc x nc.
            Eigen::MatrixXd A = J.transpose() * J;
            A.diagonal().array() += p_.lambda_sq;
            return A.ldlt().solve(J.transpose() * e);
        }
    }

    Eigen::MatrixXd pseudoinverse(const Eigen::MatrixXd& J) const {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const double eps = 1e-9;
        Eigen::VectorXd sinv = svd.singularValues().unaryExpr(
            [eps](double s){ return s > eps ? 1.0/s : 0.0; });
        return svd.matrixV() * sinv.asDiagonal() * svd.matrixU().transpose();
    }

    Eigen::VectorXd jl_gradient(const Eigen::VectorXd& q,
                                  const std::vector<int>& dof_idx) const {
        int nc = (int)dof_idx.size();
        Eigen::VectorXd g(nc);
        const auto& joints = tree_.joints_ref();
        for (int c = 0; c < nc; ++c) {
            int i = dof_idx[c];
            double lo = -M_PI, hi = M_PI;
            for (const auto& jt : joints) {
                if (jt.dof_index == i && jt.limits.has_limits) {
                    lo = jt.limits.lower;
                    hi = jt.limits.upper;
                    break;
                }
            }
            double mid   = 0.5*(lo + hi);
            double range = hi - lo;
            g[c] = (range < 1e-6) ? 0.0
                                   : -2.0*(q[i] - mid) / (range*range);
        }
        return g;
    }
};

// ============================================================================
//  HandIKSolver — all tips solved in parallel (OpenMP)
// ============================================================================
class HandIKSolver {
public:
    explicit HandIKSolver(KinematicTree& tree, IKParams params = {})
        : tree_(tree), params_(params)
    {
        for (int ti : tree.tips())
            tip_links_.push_back(ti);
    }

    int n_tips() const { return (int)tip_links_.size(); }

    Eigen::VectorXd solve_all(
        const Eigen::VectorXd& q0,
        const std::unordered_map<std::string, T_mat>& targets,
        std::unordered_map<std::string, bool>* converged_out = nullptr) const
    {
        int n = (int)tip_links_.size();
        std::vector<Eigen::VectorXd> q_results(n, q0);
        std::vector<bool> conv(n, false);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const std::string& tname = tree_.link(tip_links_[i]).name;
            auto it = targets.find(tname);
            if (it == targets.end()) { conv[i] = true; continue; }

            IKSolver solver(
                const_cast<KinematicTree&>(tree_), tname, params_);
            conv[i] = solver.solve(q0, it->second, q_results[i]);
        }

        // Patch 1.4: merge using chain_dof_indices — no Jacobian work.
        Eigen::VectorXd q_out = q0;
        std::vector<int> dof_idx;
        for (int i = 0; i < n; ++i) {
            tree_.chain_dof_indices(tip_links_[i], dof_idx);
            for (int di : dof_idx) q_out[di] = q_results[i][di];

            if (converged_out) {
                const std::string& tname = tree_.link(tip_links_[i]).name;
                (*converged_out)[tname] = conv[i];
            }
        }
        return q_out;
    }

    void set_params(const IKParams& p) { params_ = p; }
    const IKParams& params() const     { return params_; }

private:
    KinematicTree&   tree_;
    IKParams         params_;
    std::vector<int> tip_links_;
};

} // namespace lgn
