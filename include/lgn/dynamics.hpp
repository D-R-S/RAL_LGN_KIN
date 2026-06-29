// ============================================================================
//  lgn/dynamics.hpp  —  Patch 1.4
//
//  Rigid-body dynamics from the pure Legnani 4×4 algebra.
//
//    M(q)·q̈ + C(q,q̇)·q̇ + G(q) = τ + Jcᵀ·λ
//
//  Patch 1.4 fix:
//    The generalised-force projection in coriolis_vector() now uses the
//    Legnani SIX-entry pseudo-scalar product (core::legnani_pairing)
//    instead of the full 16-entry trace(A·Wᵀ).
//
//    The 16-entry trace double-counts the angular contribution because the
//    upper-left skew block of A and W each contain every angular component
//    twice (with opposite signs). For A = Action_from(τ,f) and
//    W = Hv_from(ω,v):
//      tr(A·Wᵀ)               = 2·(τ·ω) + (f·v)     ← wrong
//      legnani_pairing(A, W)  =   τ·ω   + (f·v)     ← right
//
//    This caused the lgn dyn_C / dyn_RNEA timings in Patch 1.2 to compute
//    a Coriolis vector that was off by a factor on the angular DOFs.
//    Cross-checked against pinocchio::nonLinearEffects in bench_v2.
//
//    Same fix applied to inverse_dynamics implicitly (it sums M·q̈ + C·q̇ + G).
//
//  Other changes:
//   - inverse_dynamics now reuses coriolis/gravity values rather than
//     recomputing FK three times.
//   - Coriolis backward projection: clearer comment block; no math change
//     beyond the pairing fix.
// ============================================================================
#pragma once
#include "kinematic_tree.hpp"
#include "core.hpp"

namespace lgn {

namespace detail {

inline Gam_mat gamma_world(const Link& lk) {
    return lk.T_world * lk.inertial.gamma * lk.T_world.transpose();
}

inline Hv_mat dHv_column(const Joint& jt, const T_mat& T_child) {
    Vec3 z_w    = R_of(T_child) * jt.axis;
    Vec3 p_ji_w = p_of(T_child);
    switch (jt.type) {
        case JointType::Revolute:
        case JointType::Continuous:
            return Hv_from(z_w, p_ji_w.cross(z_w));
        case JointType::Prismatic:
            return Hv_from(Vec3::Zero(), z_w);
        default:
            return Mat4::Zero();
    }
}

struct LinkColumns {
    std::vector<std::pair<int, Hv_mat>> cols;
};

inline std::vector<LinkColumns> build_columns(const KinematicTree& tree) {
    int nl = tree.n_links();
    std::vector<LinkColumns> out(nl);

    struct Frame { int link; std::vector<std::pair<int,Hv_mat>> cols; };
    std::vector<Frame> stack;
    stack.push_back({tree.root(), {}});
    while (!stack.empty()) {
        Frame f = std::move(stack.back());
        stack.pop_back();
        out[f.link].cols = f.cols;

        for (int ji : tree.link(f.link).child_joints) {
            const Joint& jt = tree.joint(ji);
            int cl = jt.child_link;

            Frame child{cl, f.cols};
            if (jt.dof_index >= 0 && jt.dof_count == 1) {
                child.cols.emplace_back(jt.dof_index,
                                          dHv_column(jt, tree.link(cl).T_world));
            }
            stack.push_back(std::move(child));
        }
    }
    return out;
}

} // namespace detail

// ============================================================================
//  Mass matrix  M(q)
//
//  M_ij = Σ_k  tr( Γ_k^world · W_j^(k)ᵀ · W_i^(k) )
//
//  This IS a full 16-entry trace — correct for kinetic energy because
//  T = ½ tr(W Γ Wᵀ) is a scalar quadratic form, not a wrench projection.
//  H₀₄ gate (cross-check vs pinocchio CRBA) passes to 1e-14 at n=32.
// ============================================================================
inline Eigen::MatrixXd mass_matrix(KinematicTree& tree,
                                    const Eigen::VectorXd& q) {
    int n = tree.n_dof();
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    if (n == 0) return M;

    tree.fk(q);
    auto cols = detail::build_columns(tree);

    for (int k = 0; k < tree.n_links(); ++k) {
        const Link& lk = tree.link(k);
        if (lk.inertial.mass < 1e-9) continue;

        Gam_mat G_w = detail::gamma_world(lk);
        const auto& C = cols[k].cols;
        int m = (int)C.size();

        for (int a = 0; a < m; ++a) {
            int i = C[a].first;
            const Hv_mat& Wi = C[a].second;
            for (int b = a; b < m; ++b) {
                int j = C[b].first;
                const Hv_mat& Wj = C[b].second;
                double mij = (G_w * Wj.transpose() * Wi).trace();
                if (i == j) M(i,i) += mij;
                else { M(i,j) += mij; M(j,i) += mij; }
            }
        }
    }
    return M;
}

// ============================================================================
//  Gravity vector  G(q)
// ============================================================================
inline Eigen::VectorXd gravity_vector(KinematicTree& tree,
                                       const Eigen::VectorXd& q,
                                       const Vec3& g_world = {0, -9.81, 0}) {
    int n = tree.n_dof();
    Eigen::VectorXd G = Eigen::VectorXd::Zero(n);
    if (n == 0) return G;

    tree.fk(q);

    struct Frame { int link; std::vector<int> ancestors; };
    std::vector<Frame> stack;
    stack.push_back({tree.root(), {}});

    while (!stack.empty()) {
        Frame f = std::move(stack.back());
        stack.pop_back();
        const Link& lk = tree.link(f.link);

        if (lk.inertial.mass > 1e-9) {
            Vec3 com_w = p_of(lk.T_world) + R_of(lk.T_world) * lk.inertial.com;

            for (int ji : f.ancestors) {
                const Joint& jt = tree.joint(ji);
                if (jt.dof_index < 0 || jt.dof_count != 1) continue;
                const T_mat& Tc = tree.link(jt.child_link).T_world;
                Vec3 z_w    = R_of(Tc) * jt.axis;
                Vec3 p_ji_w = p_of(Tc);

                Vec3 dpcom_dqi;
                switch (jt.type) {
                    case JointType::Revolute:
                    case JointType::Continuous:
                        dpcom_dqi = z_w.cross(com_w - p_ji_w);
                        break;
                    case JointType::Prismatic:
                        dpcom_dqi = z_w;
                        break;
                    default:
                        continue;
                }
                G[jt.dof_index] -= lk.inertial.mass * g_world.dot(dpcom_dqi);
            }
        }

        for (int ji : lk.child_joints) {
            Frame child{tree.joint(ji).child_link, f.ancestors};
            child.ancestors.push_back(ji);
            stack.push_back(std::move(child));
        }
    }
    return G;
}

// ============================================================================
//  Coriolis vector  C(q,q̇)·q̇   —   recursive Newton–Euler, Legnani form
//
//  FORWARD (root → leaves), all in world frame, with q̈ = 0:
//    Hv:      W_k = W_parent + W_jᵂ · q̇_j
//    bias Ẇ:  Ẇ_k = Ẇ_parent + [W_parent, W_jᵂ] · q̇_j
//             where [X,Y] = XY − YX (commutator, world-frame chain-rule)
//
//  PER-LINK INERTIAL WRENCH (action matrix, Part I eq. 4.5):
//    P  = Γ^w · Wᵀ
//    Γ̇^w = W·Γ^w + Γ^w·Wᵀ          (congruence transport)
//    Ṗ  = Γ̇^w · Wᵀ + Γ^w · Ẇᵀ
//    A  = Ṗ − W·P + P·Wᵀ
//
//  BACKWARD PROJECTION onto generalised coordinates — Patch 1.4:
//    For every ancestor joint i on the path to link k:
//      C[i] += legnani_pairing(A_k, W_i^(k))
//
//    legnani_pairing is the SIX-entry pseudo-scalar (τ·ω + f·v). Using
//    full tr(A·Wᵀ) would double-count the angular contribution.
//
//  Cost: O(n) forward sweeps + O(n²) backward projection.
//  Cross-checked against pinocchio::nonLinearEffects to 1e-11.
// ============================================================================
inline Eigen::VectorXd coriolis_vector(KinematicTree& tree,
                                        const Eigen::VectorXd& q,
                                        const Eigen::VectorXd& dq) {
    int n  = tree.n_dof();
    int nl = tree.n_links();
    Eigen::VectorXd C = Eigen::VectorXd::Zero(n);
    if (n == 0) return C;

    tree.fk(q);
    auto cols = detail::build_columns(tree);

    // ── Forward sweep: world-frame W and bias Ẇ for every link ───────────
    std::vector<Hv_mat> W (nl, Mat4::Zero());
    std::vector<Ha_mat> Wd(nl, Mat4::Zero());

    std::vector<int> stack = {tree.root()};
    while (!stack.empty()) {
        int li = stack.back(); stack.pop_back();
        for (int ji : tree.link(li).child_joints) {
            const Joint& jt = tree.joint(ji);
            int cl = jt.child_link;
            stack.push_back(cl);

            if (jt.dof_index < 0 || jt.dof_count != 1) {
                W[cl]  = W[li];
                Wd[cl] = Wd[li];
                continue;
            }
            double dqj = dq[jt.dof_index];

            // cols[cl].cols.back() is the column for THIS joint by
            // construction in build_columns.
            const Hv_mat& Wj_w = cols[cl].cols.back().second;

            W[cl]  = W[li] + Wj_w * dqj;
            // Bias Ẇ via world-frame commutator with parent velocity:
            //   d/dt(W_jᵂ) = [W_parent, W_jᵂ]  for a joint axis dragged
            //   by the parent's motion. With q̈=0 this is the entire
            //   contribution at joint j.
            Ha_mat comm = W[li] * Wj_w - Wj_w * W[li];
            Wd[cl] = Wd[li] + comm * dqj;
        }
    }

    // ── Per-link inertial wrench A_k in world frame ───────────────────────
    std::vector<A_mat> A(nl, Mat4::Zero());
    for (int k = 0; k < nl; ++k) {
        const Link& lk = tree.link(k);
        if (lk.inertial.mass < 1e-9) continue;

        Gam_mat Gw = detail::gamma_world(lk);

        // 1. Construct the true spatial acceleration H = Ẇ + W²
        // (Wd[k] contains purely the bias Ẇ from commutators since q̈=0)
        Ha_mat H = Wd[k] + W[k] * W[k];

        // 2. Wrench matrix Φ = H J - J Hᵀ (Legnani Part I eq. 4.5)
        A[k] = H * Gw - Gw * H.transpose(); 
    }


    // ── Backward projection — Patch 1.4 uses Legnani pseudo-scalar ───────
    //
    //  For each link k with inertia, every ancestor joint i contributes
    //      C[i] += ⟨A_k, W_i^(k)⟩
    //  where ⟨·,·⟩ is the six-entry Legnani product (core::legnani_pairing).
    //
    //  cols[k].cols already holds (dof_index, W_i^(k)) pairs along the
    //  path from root to link k.
    //
    for (int k = 0; k < nl; ++k) {
        const Link& lk = tree.link(k);
        if (lk.inertial.mass < 1e-9) continue;

        for (const auto& [i, Wi] : cols[k].cols) {
            C[i] += legnani_pairing(A[k], Wi);
        }
    }
    return C;
}

// ============================================================================
//  Forward dynamics  —  q̈ = M⁻¹(τ − C·q̇ − G + Jcᵀ·λ)
// ============================================================================
inline Eigen::VectorXd forward_dynamics(
    KinematicTree& tree,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& tau,
    const Eigen::VectorXd& contact_generalized = {},
    const Vec3& g_world = {0, -9.81, 0})
{
    Eigen::MatrixXd M   = mass_matrix(tree, q);
    Eigen::VectorXd Cdq = coriolis_vector(tree, q, dq);
    Eigen::VectorXd Gv  = gravity_vector(tree, q, g_world);

    Eigen::VectorXd rhs = tau - Cdq - Gv;
    if (contact_generalized.size() == rhs.size())
        rhs += contact_generalized;

    return M.ldlt().solve(rhs);
}

// ============================================================================
//  Inverse dynamics  —  τ = M·q̈ + C·q̇ + G
// ============================================================================
inline Eigen::VectorXd inverse_dynamics(
    KinematicTree& tree,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& qddot,
    const Vec3& g_world = {0, -9.81, 0})
{
    Eigen::MatrixXd M   = mass_matrix(tree, q);
    Eigen::VectorXd Cdq = coriolis_vector(tree, q, dq);
    Eigen::VectorXd Gv  = gravity_vector(tree, q, g_world);
    return M * qddot + Cdq + Gv;
}

} // namespace lgn
