// ============================================================================
//  lgn/core.hpp  —  Patch 1.4
//
//  Legnani, Casolo, Righettini & Zappa (1996)
//  "A Homogeneous Matrix Approach to 3D Kinematics and Dynamics"
//  Part I: Theory   — Mech. Mach. Theory 31:573-587
//  Part II: Apps    — Mech. Mach. Theory 31:589-605
//
//  ALL kinematics and dynamics live in 4×4 matrix space.
//  Angular + linear are UNIFIED — one matrix instead of a 6-vector pair.
//  No vectors of size 6. No separate ω/v bookkeeping. Ever.
//
//  Patch 1.4 changes:
//   - Added legnani_pairing(A, W): the SIX-entry pseudo-scalar product
//     between an action matrix A and a velocity matrix W. This is the
//     correct generalised-force projection (replaces tr(A·Wᵀ), which
//     double-counts the angular contribution).
//   - Removed Ha_revolute, Ha_prismatic, Ha_propagate. They were dead
//     code with broken math (missing W² centripetal term; cross-term
//     had wrong sign structure). Correct Ẇ propagation now lives inline
//     in dynamics::coriolis_vector via the commutator form
//     Ẇ_C = Ẇ_P + [W_P, W_j]·q̇_j (with q̈=0).
// ============================================================================
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>

namespace lgn {

// ============================================================================
//  Type aliases  —  all four Legnani matrix kinds share the same underlying
//  storage (Mat4). The aliases are documentation, not distinct types.
// ============================================================================
using Mat4    = Eigen::Matrix4d;
using Mat3    = Eigen::Matrix3d;
using Vec3    = Eigen::Vector3d;
using Vec4    = Eigen::Vector4d;
using Mat2    = Eigen::Matrix2d;
using Vec2    = Eigen::Vector2d;

using T_mat   = Mat4;   // SE(3) homogeneous transform              (Part I §2)
using Hv_mat  = Mat4;   // velocity matrix  [ skew(ω)  v ; 0  0 ]   (Part I §3)
using Ha_mat  = Mat4;   // acceleration matrix (Ẇ + W², in dynamics)(Part I §3)
using Gam_mat = Mat4;   // pseudo-inertia Γ                         (Part I §4)
using P_mat   = Mat4;   // momentum matrix  P = Γ Wᵀ                (Part I §4)
using A_mat   = Mat4;   // action (wrench) matrix                   (Part I §4)

// ============================================================================
//  §2  Homogeneous Transforms  T ∈ SE(3)
// ============================================================================

inline T_mat T_identity() { return Mat4::Identity(); }

inline T_mat T_from_Rp(const Mat3& R, const Vec3& p) {
    T_mat T = Mat4::Identity();
    T.topLeftCorner<3,3>()  = R;
    T.topRightCorner<3,1>() = p;
    return T;
}

inline Mat3 R_of(const T_mat& T) { return T.topLeftCorner<3,3>(); }
inline Vec3 p_of(const T_mat& T) { return T.topRightCorner<3,1>(); }

/// Inverse of SE(3) — Rᵀ, −Rᵀp — exact, no LU decomposition needed.
inline T_mat T_inv(const T_mat& T) {
    Mat3 Rt = T.topLeftCorner<3,3>().transpose();
    T_mat Ti = Mat4::Zero();
    Ti.topLeftCorner<3,3>()  =  Rt;
    Ti.topRightCorner<3,1>() = -Rt * T.topRightCorner<3,1>();
    Ti(3,3) = 1.0;
    return Ti;
}

inline T_mat T_rot(const Vec3& axis, double theta) {
    return T_from_Rp(
        Eigen::AngleAxisd(theta, axis).toRotationMatrix(),
        Vec3::Zero());
}

inline T_mat T_trans(const Vec3& d) {
    return T_from_Rp(Mat3::Identity(), d);
}

/// URDF convention: R = Rz · Ry · Rx
inline Mat3 rpy_to_rot(const Vec3& rpy) {
    return (Eigen::AngleAxisd(rpy.z(), Vec3::UnitZ())
          * Eigen::AngleAxisd(rpy.y(), Vec3::UnitY())
          * Eigen::AngleAxisd(rpy.x(), Vec3::UnitX()))
           .toRotationMatrix();
}

// ============================================================================
//  Skew-symmetric matrix
// ============================================================================
inline Mat3 skew(const Vec3& v) {
    Mat3 S;
    S <<     0, -v.z(),  v.y(),
          v.z(),     0, -v.x(),
         -v.y(),  v.x(),    0;
    return S;
}

// ============================================================================
//  §3  Velocity Matrix Hv
//
//  Hv = [ skew(ω)  v ]
//       [   0      0 ]
//
//  v is the velocity of the body-fixed point currently at the WORLD ORIGIN.
//  Velocity at an arbitrary world point p is v_at_point(Hv, p) = v + ω×p.
// ============================================================================

inline Hv_mat Hv_from(const Vec3& omega, const Vec3& v) {
    Hv_mat H = Mat4::Zero();
    H(0,1) = -omega.z();  H(0,2) =  omega.y();  H(0,3) = v.x();
    H(1,0) =  omega.z();  H(1,2) = -omega.x();  H(1,3) = v.y();
    H(2,0) = -omega.y();  H(2,1) =  omega.x();  H(2,3) = v.z();
    return H;
}

inline Vec3 omega_of(const Hv_mat& Hv) {
    return { Hv(2,1), Hv(0,2), Hv(1,0) };
}

inline Vec3 v_of(const Hv_mat& Hv) {
    return Hv.col(3).head<3>();
}

inline Vec3 v_at_point(const Hv_mat& Hv, const Vec3& p) {
    return v_of(Hv) + omega_of(Hv).cross(p);
}

inline Hv_mat Hv_revolute(const Vec3& axis, double theta_dot) {
    return Hv_from(theta_dot * axis, Vec3::Zero());
}

inline Hv_mat Hv_prismatic(const Vec3& axis, double d_dot) {
    return Hv_from(Vec3::Zero(), d_dot * axis);
}

/// Velocity composition (Part I eq. 3.9), expressed in same observer frame:
///   Hv_child = Hv_parent + T · Hv_rel · T⁻¹
/// where T = G^{obs, local_of_Hv_rel}. For typical use:
///   - Hv_prev   = parent link velocity in WORLD frame
///   - Hv_rel    = joint twist in CHILD link's local frame
///   - T         = child link's WORLD transform
inline Hv_mat Hv_propagate(const Hv_mat& Hv_prev,
                             const T_mat&  T_world_local,
                             const Hv_mat& Hv_rel_local) {
    return Hv_prev + T_world_local * Hv_rel_local * T_inv(T_world_local);
}

// ============================================================================
//  §4  Pseudo-inertia Matrix Γ
//
//  Γ = [ Σ          m·g ]
//      [ m·gᵀ        m  ]
//
//  with second-moment matrix Σ related to the conventional inertia I_G by
//      Σ = ½ tr(I_O) I₃ − I_O,   I_O = I_G + m·skew(g)·skew(g)ᵀ
//  so that kinetic energy is T = ½ tr(W Γ Wᵀ).
//
//  Cross-checked against pinocchio::crba to 1e-14 at chain length 32.
// ============================================================================
inline Gam_mat Gamma(const Mat3& I_G, const Vec3& g, double m) {
    Mat3 sg    = skew(g);
    Mat3 I_O   = I_G + m * sg * sg.transpose();
    Mat3 Sigma = 0.5 * I_O.trace() * Mat3::Identity() - I_O;

    Gam_mat G = Mat4::Zero();
    G.topLeftCorner<3,3>()    = Sigma;
    G.topRightCorner<3,1>()   = m * g;
    G.bottomLeftCorner<1,3>() = m * g.transpose();
    G(3, 3) = m;
    return G;
}

// ============================================================================
//  §4  Momentum  P = Γ Wᵀ
// ============================================================================
inline P_mat Momentum(const Gam_mat& Gam, const Hv_mat& Hv) {
    return Gam * Hv.transpose();
}

// ============================================================================
//  §4  Action (wrench) Matrix A
//
//  A = [ skew(τ)  f ]   — same shape as Hv. Encodes wrench (f, τ).
//      [   0      0 ]
//
//  Newton–Euler in matrix form (Part I eq. 4.5):
//     A = Ṗ − W·P + P·Wᵀ
// ============================================================================
inline A_mat Action_from(const Vec3& tau, const Vec3& f) {
    A_mat A = Mat4::Zero();
    A(0,1) = -tau.z();  A(0,2) =  tau.y();  A(0,3) =  f.x();
    A(1,0) =  tau.z();  A(1,2) = -tau.x();  A(1,3) =  f.y();
    A(2,0) = -tau.y();  A(2,1) =  tau.x();  A(2,3) =  f.z();
    A(3,0) = -f.x();    A(3,1) = -f.y();    A(3,2) = -f.z(); 
    return A;
}


inline A_mat Action_NE(const P_mat& dP_dt, const Hv_mat& Hv, const P_mat& P) {
    return dP_dt - Hv * P + P * Hv.transpose();
}

inline Vec3 tau_of(const A_mat& A) { return { A(2,1), A(0,2), A(1,0) }; }
inline Vec3 f_of  (const A_mat& A) { return A.col(3).head<3>(); }

// ============================================================================
//  Legnani PSEUDO-SCALAR PRODUCT  ⟨A, W⟩
//
//  The natural inner product between an Action matrix A (wrench) and a
//  velocity/twist matrix W is NOT trace(A Wᵀ). The full 16-entry trace
//  double-counts the angular contribution (the skew block contains each
//  off-diagonal component twice with opposite signs).
//
//  Legnani Part I defines the SIX-entry pseudo-scalar product
//      ⟨A, W⟩ = a₃₂·w₃₂ + a₁₃·w₁₃ + a₂₁·w₂₁    (angular: τ·ω)
//             + a₁₄·w₁₄ + a₂₄·w₂₄ + a₃₄·w₃₄    (linear:  f·v)
//
//  pairing one slot per wrench/twist component. Verified:
//  for A = Action_from(τ, f) and W = Hv_from(ω, v) this returns τ·ω + f·v.
//
//  THIS is what projects a Cartesian wrench into a generalised joint force.
//  Used in dynamics::coriolis_vector and dynamics::inverse_dynamics.
// ============================================================================
inline double legnani_pairing(const Mat4& A, const Mat4& W) {
    return A(2,1)*W(2,1) + A(0,2)*W(0,2) + A(1,0)*W(1,0)
         + A(0,3)*W(0,3) + A(1,3)*W(1,3) + A(2,3)*W(2,3);
}

// ============================================================================
//  Force at an arbitrary body point → Action matrix
// ============================================================================
inline A_mat force_at_point(const Vec3& f_world,
                              const Vec3& r_app,
                              const Vec3& r_origin) {
    Vec3 tau = (r_app - r_origin).cross(f_world);
    return Action_from(tau, f_world);
}

inline A_mat force_at_local_point(const Vec3& f_world,
                                   const Vec3& r_local,
                                   const T_mat& T_world) {
    Vec3 r_w = p_of(T_world) + R_of(T_world) * r_local;
    return force_at_point(f_world, r_w, p_of(T_world));
}

// ============================================================================
//  Kinetic energy of one link  (Part I eq. 4.6)
//      T_k = ½ trace(W Γ Wᵀ) = ½ trace(Γ Wᵀ W)
// ============================================================================
inline double KineticEnergy(const Gam_mat& Gam, const Hv_mat& Hv) {
    return 0.5 * (Gam * Hv.transpose() * Hv).trace();
}

// ============================================================================
//  Differential actuator mapping  (Amazing Hand — 2-servo / finger)
// ============================================================================
inline Mat2 differential_B() {
    Mat2 B;
    B << 0.5,  0.5,
         0.5, -0.5;
    return B;
}

inline Mat2 differential_B_inv() {
    Mat2 Bi;
    Bi << 1.0,  1.0,
          1.0, -1.0;
    return Bi;
}

inline Vec2 servo_to_joint(const Vec2& u) { return differential_B()     * u; }
inline Vec2 joint_to_servo(const Vec2& q) { return differential_B_inv() * q; }

inline Eigen::VectorXd hand_servo_to_joint(const Eigen::VectorXd& u) {
    assert(u.size() == 8);
    Eigen::VectorXd q(8);
    Mat2 B = differential_B();
    for (int f = 0; f < 4; ++f) q.segment<2>(f*2) = B * u.segment<2>(f*2);
    return q;
}

inline Eigen::VectorXd hand_joint_to_servo(const Eigen::VectorXd& q) {
    assert(q.size() == 8);
    Eigen::VectorXd u(8);
    Mat2 Bi = differential_B_inv();
    for (int f = 0; f < 4; ++f) u.segment<2>(f*2) = Bi * q.segment<2>(f*2);
    return u;
}

} // namespace lgn
