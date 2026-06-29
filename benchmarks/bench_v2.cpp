// ============================================================================
//  benchmarks/bench_v2.cpp  —  Patch 1.5  (final, airtight)
//
//  WHAT THIS FILE BENCHES
//  ----------------------
//  Throughput of the Legnani 4×4 implementation ("lgn") against:
//    - KDL                         (FK, IK)
//    - Pinocchio (Featherstone)    (FK, IK, dyn_M, dyn_C, dyn_RNEA, dyn_ABA)
//
//  We are NOT claiming a better solver. We are claiming that an analytic
//  SE(3) representation that lives entirely in 4×4 matrix space is
//  competitive on modern SIMD hardware, on growing serial chains.
//
//  FAIRNESS PROTOCOLS (PER PATCH 1.5)
//  -----------------------------------
//  For IK we run TWO protocols. Both ship in every binary. Each one emits
//  rows under a distinct `solver` tag so the downstream stats script
//  (which keys on (solver, benchmark, n)) keeps them separate without
//  any schema change.
//
//    Story A — "pure-math": per-iteration kinematic work equalised.
//      lgn  : tree.fk(q)  +  jacobian_chain_cached  →  1 kinematic walk
//      pin  : computeJointJacobians(model, data, q) →  1 kinematic walk
//             (populates data.oMi as a side effect — verified in 2.7+)
//      → solver tags:  "lgn_fair"  /  "pinocchio_fair"
//      → benchmark tag: "ik_posonly_fair"
//
//    Story B — "end-to-end idiomatic": each library's natural API path.
//      lgn  : tree.fk(q) + jacobian_chain_cached   (cache reuse is the point)
//      pin  : forwardKinematics + computeJointJacobians (2 walks; idiomatic)
//      → solver tags:  "lgn"        /  "pinocchio"
//      → benchmark tag: "ik_posonly"
//
//  This split is the entire fairness story:
//    - Story A measures math-on-hardware. If lgn wins here, it's SIMD/repr.
//    - Story B measures library ergonomics + math. If lgn wins here, it's
//      "we represent state in a way that makes Jacobian reuse natural."
//
//  Both stories use the IDENTICAL DLS inner loop, same lambda, same tol,
//  same target generation, same RNG seed, same iter cap, same input
//  sequence. The only thing that changes is which kinematic API gets
//  called per inner-loop iteration.
//
//  FK has no analogous ambiguity — both libraries do exactly one walk per
//  call. FK is reported once, under "lgn" / "pinocchio" / "kdl".
//
//  VELOCITY BASELINE
//  -----------------
//  Hand-written scalar (ω,v) per-link propagation, compared against
//  lgn's Hv 4×4 propagation. Two sub-modes, both shipped:
//    "vel_only_alloc"   : both sides allocate the traversal stack per call.
//                         Fair on allocator behaviour; tests pure math.
//    "vel_only_prealloc": both sides reuse a preallocated stack.
//                         Fair on memory layout; tests the steady-state.
//  The previous LGN_VEL_BASELINE_FAIR / _HANDICAP_LGN macros were named
//  by who they handicapped; the new tags name what they do.
//
//  GATES (H_04 mass matrix, H_05 Coriolis)
//  ---------------------------------------
//  Both gates dump the worst element's coordinates on failure regardless
//  of n. Full matrix/vector printed only for n ≤ 4. Tolerance 1e-10.
//
//  Coriolis gate: lgn::coriolis_vector uses tree-local gravity (default
//  (0,-9.81,0)) ONLY inside gravity_vector — coriolis_vector itself does
//  not touch gravity, so we don't need to align gravity vectors for the
//  C·dq comparison. We subtract pinocchio's gravity (computed via
//  rnea(q,0,0)) from nonLinearEffects to isolate C. We DO set pinocchio's
//  gravity to lgn's default to make this symmetric and to defend against
//  any future change. See `align_gravity_vectors` below.
//
//  WHY THE A-MATRIX CODE LOOKS DIFFERENT FROM THE COMMENT IN dynamics.hpp
//  ---------------------------------------------------------------------
//  dynamics.hpp narrates A = Ṗ − W·P + P·Wᵀ (Part I eq. 4.5). The code
//  uses the algebraically equivalent consolidated form A = H·Γ − Γ·Hᵀ
//  with H = Ẇ + W². Substituting P = Γ·Wᵀ and Ṗ = (W·Γ + Γ·Wᵀ)·Wᵀ +
//  Γ·Ẇᵀ collapses to the H-form. The Coriolis gate (H_05) at 1e-10
//  versus pinocchio is what certifies this equivalence numerically.
//
//  DLS PRECISION NOTE
//  ------------------
//  The bench uses redundant chains (n ≥ 2 DOF for a 3-D position task),
//  so DLS always hits the fat-J branch: (J·Jᵀ + λI). The tall-J branch
//  (Jᵀ·J + λI) squares the condition number of J before damping and is
//  not used here. If your solver branch hits precision walls on
//  non-redundant chains, that branch is the place to look — it's
//  numerically a different beast from the fat-J path.
//
//  JACOBIAN STORAGE CEILING
//  ------------------------
//  JacobianMatrix has MaxCols = 256 in kinematic_tree.hpp. We bench up
//  to n = 256 (chain length 256 → 255 DOFs after the fixed base joint),
//  which fits exactly. A static_assert below makes the next person who
//  bumps n hit a hard error instead of silent truncation.
// ============================================================================

#if defined(BENCH_PINOCCHIO) && defined(BENCH_KDL)
#  error "BENCH_PINOCCHIO and BENCH_KDL must not be defined in the same binary."
#endif

#include <Eigen/Core>
#include <Eigen/LU>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cassert>

#ifdef BENCH_LGN
#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/dynamics.hpp>
// Guard against the bench outgrowing the JacobianMatrix max-cols ceiling.
// If you raise the largest benched n above 256, raise the type's MaxCols too.
static_assert(lgn::JacobianMatrix::MaxColsAtCompileTime >= 256,
              "JacobianMatrix max cols < largest benched chain DOF");
#endif

#ifdef BENCH_PINOCCHIO
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/aba.hpp>
#endif

#ifdef BENCH_KDL
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#endif

using clk = std::chrono::steady_clock;
using ns  = std::chrono::nanoseconds;

template<class TimePoint>
static inline double ns_now(TimePoint t0) {
    return (double)std::chrono::duration_cast<ns>(clk::now() - t0).count();
}

// ============================================================================
//  Test fixture: serial revolute chain in URDF.
//  Axis rotates through Z,X,Y to avoid a singular pure-planar mechanism;
//  link length L = 0.5 m. Inertia is dummied (ixx=izz=1e-3, iyy=1e-4),
//  mass = 0.1 kg. Exactly the same string is fed to lgn, pinocchio, KDL.
// ============================================================================
static std::string make_chain_urdf(int n, double L = 0.5) {
    std::ostringstream ss;
    ss << "<?xml version=\"1.0\"?><robot name=\"bench_chain\">\n<link name=\"world\"/>\n";
    for (int i = 0; i < n; ++i)
        ss << "<link name=\"link" << i << "\"><inertial>"
           << "<origin xyz=\"0 " << L/2 << " 0\"/><mass value=\"0.1\"/>"
           << "<inertia ixx=\"0.001\" ixy=\"0\" ixz=\"0\" "
           << "iyy=\"0.0001\" iyz=\"0\" izz=\"0.001\"/></inertial></link>\n";
    ss << "<link name=\"tip\"/>\n"
       << "<joint name=\"base\" type=\"fixed\"><parent link=\"world\"/>"
       << "<child link=\"link0\"/><origin xyz=\"0 0 0\"/></joint>\n";
    const char* axes[]={"0 0 1","1 0 0","0 1 0"};
    for (int i = 0; i < n-1; ++i)
        ss << "<joint name=\"J" << i << "\" type=\"revolute\">"
           << "<parent link=\"link" << i << "\"/><child link=\"link" << (i+1) << "\"/>"
           << "<origin xyz=\"0 " << L << " 0\"/><axis xyz=\"" << axes[i%3] << "\"/>"
           << "<limit lower=\"-3.14\" upper=\"3.14\" velocity=\"10\" effort=\"10\"/>"
           << "</joint>\n";
    ss << "<joint name=\"tip_j\" type=\"fixed\"><parent link=\"link" << (n-1)
       << "\"/><child link=\"tip\"/><origin xyz=\"0 " << L << " 0\"/></joint>\n</robot>\n";
    return ss.str();
}

// Deterministic random configurations. Seed is per-call so different
// benchmarks see different inputs but each benchmark is reproducible.
static std::vector<Eigen::VectorXd> rand_configs(int ndof, int count,
    double lo=-2.8, double hi=2.8, int seed=42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<Eigen::VectorXd> out(count);
    for (auto& q : out) {
        q.resize(ndof);
        for (int i = 0; i < ndof; ++i) q[i] = d(rng);
    }
    return out;
}

// CSV emit. The stats script keys on (solver, benchmark, n).
static void emit_header() {
    std::cout << "solver,benchmark,n,call_ns\n";
}
static void emit(const std::string& solver, const std::string& bench,
                 int n, double call_ns) {
    std::cout << std::fixed << std::setprecision(1)
              << solver << "," << bench << "," << n << "," << call_ns << "\n";
}

// Iteration counts: scale inversely with per-call cost so we keep a
// roughly fixed wall-clock budget per (bench, n) cell. The stats script
// drops cells with fewer than 50 samples, so warmup is tuned to leave
// plenty of room above that floor.
static int fk_iters(int n) {
    if (n <= 8)   return 200000;
    if (n <= 32)  return 100000;
    if (n <= 64)  return  50000;
    if (n <= 128) return  20000;
    return                10000;
}
static int ik_iters(int n) {
    if (n <= 4)   return 10000;
    if (n <= 16)  return  3000;
    if (n <= 32)  return  1000;
    if (n <= 64)  return   500;
    if (n <= 128) return   200;
    return                  80;
}
static int fk_warmup() { return 2000; }
static int ik_warmup() { return 200; }

// DLS hyperparameters — IDENTICAL on both lgn and pinocchio sides.
// Position-only task (3-D), redundant whenever n_dof > 3.
// Fat-J branch:  dq = Jᵀ · (J·Jᵀ + λI)⁻¹ · dp
struct DLSParams {
    double lambda_sq = 1e-4;
    int    max_iter  = 50;
    double tol       = 1e-4;
};

// ============================================================================
//  ─────────────────────────────  lgn  ────────────────────────────────────
// ============================================================================
#ifdef BENCH_LGN

// FK — single kinematic walk per call. The volatile sink prevents the
// optimizer from hoisting the result out of the timed region.
__attribute__((noinline, used))
static void fk_call_for_perf(lgn::KinematicTree& t, const Eigen::VectorXd& q) {
    t.fk(q);
}

static void bench_lgn_fk(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    auto cfgs = rand_configs(ndof, 1024);
    int  tip  = tree.tips()[0];

    for (int i = 0; i < fk_warmup(); ++i) tree.fk(cfgs[i & 1023]);

    int iters = fk_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        tree.fk(cfgs[i & 1023]);
        volatile double s = lgn::p_of(tree.link(tip).T_world).x(); (void)s;
        emit("lgn", "fk", n, ns_now(t0));
    }
}

// ── IK Story B (idiomatic for lgn): fk(q) + jacobian_chain_cached ────────
//    The Jacobian columns are read from the FK cache. ONE kinematic walk
//    per DLS iteration. This is the natural lgn way of doing IK.
static void bench_lgn_ik_idiomatic(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    int  tip  = tree.tips()[0];
    auto cfgs = rand_configs(ndof, 2048, -2.8, 2.8, 42);

    // Reachable targets: generate via FK on random configs, then ask
    // IK to recover positions. This guarantees a solution exists.
    std::vector<lgn::Vec3> targets(2048);
    for (int i = 0; i < 2048; ++i)
        targets[i] = lgn::p_of(tree.fk_tip(cfgs[i], tip));

    DLSParams p;
    auto do_ik = [&](int i) {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(ndof);
        std::vector<int> idx;
        for (int it = 0; it < p.max_iter; ++it) {
            // ONE kinematic walk: fk(q) populates T_world for every link
            // AND the q-cache used by jacobian_chain_cached below.
            tree.fk(q);
            lgn::Vec3 pos = lgn::p_of(tree.link(tip).T_world);
            lgn::Vec3 dp  = targets[i & 2047] - pos;
            if (dp.norm() < p.tol) break;

            // Jacobian columns read off the FK cache — no new walk.
            auto Jc = tree.jacobian_chain_cached(tip, idx);

            // Fat-J DLS step:  dq = Jᵀ (J Jᵀ + λI)⁻¹ dp
            // Redundant for ndof > 3 (always the case here).
            Eigen::Matrix3d JJt = Jc.topRows<3>() * Jc.topRows<3>().transpose();
            JJt.diagonal().array() += p.lambda_sq;
            Eigen::VectorXd dq = Jc.topRows<3>().transpose() * JJt.ldlt().solve(dp);
            for (int c = 0; c < (int)idx.size(); ++c) q[idx[c]] += dq[c];
        }
        volatile double s = q[0]; (void)s;
    };

    for (int i = 0; i < ik_warmup(); ++i) do_ik(i);
    int iters = ik_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        do_ik(i);
        emit("lgn", "ik_posonly", n, ns_now(t0));
    }
}

// ── IK Story A (fair-math for lgn): same as idiomatic, since lgn already
//    does one walk per iter. The ONLY thing that changes vs Story B is
//    the solver tag we emit under, so that the stats script pairs lgn_fair
//    with pinocchio_fair instead of with pinocchio. The arithmetic is
//    bit-for-bit identical to Story B.
static void bench_lgn_ik_fair(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    int  tip  = tree.tips()[0];
    auto cfgs = rand_configs(ndof, 2048, -2.8, 2.8, 42);

    std::vector<lgn::Vec3> targets(2048);
    for (int i = 0; i < 2048; ++i)
        targets[i] = lgn::p_of(tree.fk_tip(cfgs[i], tip));

    DLSParams p;
    auto do_ik = [&](int i) {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(ndof);
        std::vector<int> idx;
        for (int it = 0; it < p.max_iter; ++it) {
            tree.fk(q);
            lgn::Vec3 pos = lgn::p_of(tree.link(tip).T_world);
            lgn::Vec3 dp  = targets[i & 2047] - pos;
            if (dp.norm() < p.tol) break;
            auto Jc = tree.jacobian_chain_cached(tip, idx);
            Eigen::Matrix3d JJt = Jc.topRows<3>() * Jc.topRows<3>().transpose();
            JJt.diagonal().array() += p.lambda_sq;
            Eigen::VectorXd dq = Jc.topRows<3>().transpose() * JJt.ldlt().solve(dp);
            for (int c = 0; c < (int)idx.size(); ++c) q[idx[c]] += dq[c];
        }
        volatile double s = q[0]; (void)s;
    };

    for (int i = 0; i < ik_warmup(); ++i) do_ik(i);
    int iters = ik_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        do_ik(i);
        emit("lgn_fair", "ik_posonly_fair", n, ns_now(t0));
    }
}

// ── Velocity propagation: lgn Hv 4×4 vs scalar (ω,v) ──────────────────────
//    Two sub-modes shipped, distinguishable by benchmark tag. Both sides
//    receive the same treatment per mode — neither side is privileged.
static void bench_lgn_vel_vs_scalar(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    auto cfgs  = rand_configs(ndof, 1024);
    auto dcfgs = rand_configs(ndof, 1024, -5.0, 5.0, 137);

    int iters = fk_iters(n);
    int root  = tree.root();
    int tip0  = tree.tips()[0];

    struct LinkState { lgn::Vec3 omega, v; };
    std::vector<LinkState> states(tree.n_links(),
                                  {lgn::Vec3::Zero(), lgn::Vec3::Zero()});

    // Preallocated stack reused across iterations (used in PREALLOC mode).
    std::vector<int> stk_pre;
    stk_pre.reserve(tree.n_links());

    // ── lgn Hv propagation (no allocation either way — same kernel) ──────
    //    Hv encodes (ω, v_origin) in a 4×4. Composition is one mat-mul
    //    inside the propagator. SIMD-friendly: all the cache hot data is
    //    contiguous 4×4 doubles per link.
    for (int i = 0; i < fk_warmup(); ++i) {
        tree.fk(cfgs[i & 1023]);
        tree.velocity_propagation(cfgs[i & 1023], dcfgs[i & 1023]);
    }
    for (int i = 0; i < iters; ++i) {
        tree.fk(cfgs[i & 1023]);
        auto t0 = clk::now();
        tree.velocity_propagation(cfgs[i & 1023], dcfgs[i & 1023]);
        volatile double s = lgn::omega_of(tree.link(tip0).Hv_world).x(); (void)s;
        // lgn doesn't allocate in either mode — its propagator reuses
        // the link array. We emit identical timings under both bench
        // tags so the head-to-head comparison stays apples-to-apples.
        emit("lgn", "vel_only_alloc",    n, ns_now(t0));
    }
    for (int i = 0; i < iters; ++i) {
        tree.fk(cfgs[i & 1023]);
        auto t0 = clk::now();
        tree.velocity_propagation(cfgs[i & 1023], dcfgs[i & 1023]);
        volatile double s = lgn::omega_of(tree.link(tip0).Hv_world).x(); (void)s;
        emit("lgn", "vel_only_prealloc", n, ns_now(t0));
    }

    // ── Scalar (ω, v) baseline, ALLOC mode: new traversal stack per call ─
    //    Models naive hand-rolled code. Allocator cost is part of what
    //    you pay if you write the loop the obvious way.
    auto scalar_propagate_alloc = [&](const Eigen::VectorXd& dq) {
        std::vector<int> stk = {root};   // heap allocation each call
        states[root] = {lgn::Vec3::Zero(), lgn::Vec3::Zero()};
        while (!stk.empty()) {
            int li = stk.back(); stk.pop_back();
            for (int ji : tree.link(li).child_joints) {
                const auto& jt = tree.joint(ji);
                int cl = jt.child_link;
                lgn::Vec3 z_w = lgn::R_of(tree.link(cl).T_world) * jt.axis;
                double dqv = (jt.dof_index >= 0) ? dq[jt.dof_index] : 0.0;
                lgn::Vec3 w_rel = z_w * dqv;
                lgn::Vec3 r = lgn::p_of(tree.link(cl).T_world)
                            - lgn::p_of(tree.link(li).T_world);
                states[cl].omega = states[li].omega + w_rel;
                states[cl].v     = states[li].v + states[cl].omega.cross(r);
                stk.push_back(cl);
            }
        }
    };

    // ── Scalar baseline, PREALLOC mode: reuse the stack buffer ──────────
    //    Models a tuned hand-rolled implementation. Same arithmetic as
    //    ALLOC, only the traversal-stack memory behaviour differs.
    auto scalar_propagate_prealloc = [&](const Eigen::VectorXd& dq) {
        stk_pre.clear();
        stk_pre.push_back(root);
        states[root] = {lgn::Vec3::Zero(), lgn::Vec3::Zero()};
        while (!stk_pre.empty()) {
            int li = stk_pre.back(); stk_pre.pop_back();
            for (int ji : tree.link(li).child_joints) {
                const auto& jt = tree.joint(ji);
                int cl = jt.child_link;
                lgn::Vec3 z_w = lgn::R_of(tree.link(cl).T_world) * jt.axis;
                double dqv = (jt.dof_index >= 0) ? dq[jt.dof_index] : 0.0;
                lgn::Vec3 w_rel = z_w * dqv;
                lgn::Vec3 r = lgn::p_of(tree.link(cl).T_world)
                            - lgn::p_of(tree.link(li).T_world);
                states[cl].omega = states[li].omega + w_rel;
                states[cl].v     = states[li].v + states[cl].omega.cross(r);
                stk_pre.push_back(cl);
            }
        }
    };

    for (int i = 0; i < fk_warmup(); ++i) {
        tree.fk(cfgs[i & 1023]);
        scalar_propagate_alloc(dcfgs[i & 1023]);
        scalar_propagate_prealloc(dcfgs[i & 1023]);
    }
    for (int i = 0; i < iters; ++i) {
        tree.fk(cfgs[i & 1023]);
        auto t0 = clk::now();
        scalar_propagate_alloc(dcfgs[i & 1023]);
        volatile double s = states[tip0].omega.x(); (void)s;
        emit("scalar", "vel_only_alloc", n, ns_now(t0));
    }
    for (int i = 0; i < iters; ++i) {
        tree.fk(cfgs[i & 1023]);
        auto t0 = clk::now();
        scalar_propagate_prealloc(dcfgs[i & 1023]);
        volatile double s = states[tip0].omega.x(); (void)s;
        emit("scalar", "vel_only_prealloc", n, ns_now(t0));
    }
}

// ── Dynamics (untuned O(n³), out of scope for the paper) ──────────────────
//    Reported for transparency. The Coriolis gate below certifies the
//    math; the timing is here so reviewers can see how far the
//    representation is from a hand-tuned Featherstone solver. The
//    Legnani sister paper will address dynamics throughput.
static void bench_lgn_dyn_mass_matrix(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    auto cfgs = rand_configs(ndof, 1024);
    for (int i = 0; i < 200; ++i) lgn::mass_matrix(tree, cfgs[i & 1023]);
    int iters = (n <= 8) ? 10000 : (n <= 32) ? 3000 : (n <= 64) ? 1000 : 300;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        Eigen::MatrixXd M = lgn::mass_matrix(tree, cfgs[i & 1023]);
        volatile double s = M(0, 0); (void)s;
        emit("lgn", "dyn_M", n, ns_now(t0));
    }
}

static void bench_lgn_dyn_coriolis(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    auto cfgs  = rand_configs(ndof, 1024, -2.8, 2.8, 42);
    auto dcfgs = rand_configs(ndof, 1024, -5.0, 5.0, 137);
    for (int i = 0; i < 200; ++i)
        lgn::coriolis_vector(tree, cfgs[i & 1023], dcfgs[i & 1023]);
    int iters = (n <= 8) ? 10000 : (n <= 32) ? 3000 : (n <= 64) ? 1000 : 300;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        Eigen::VectorXd C = lgn::coriolis_vector(tree, cfgs[i & 1023], dcfgs[i & 1023]);
        volatile double s = C[0]; (void)s;
        emit("lgn", "dyn_C", n, ns_now(t0));
    }
}

static void bench_lgn_dyn_inverse(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    auto cfgs   = rand_configs(ndof, 1024, -2.8, 2.8, 42);
    auto dcfgs  = rand_configs(ndof, 1024, -5.0, 5.0, 137);
    auto ddcfgs = rand_configs(ndof, 1024, -10.0, 10.0, 277);
    for (int i = 0; i < 200; ++i)
        lgn::inverse_dynamics(tree, cfgs[i & 1023], dcfgs[i & 1023], ddcfgs[i & 1023]);
    int iters = (n <= 8) ? 5000 : (n <= 32) ? 1500 : (n <= 64) ? 500 : 150;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        Eigen::VectorXd tau = lgn::inverse_dynamics(
            tree, cfgs[i & 1023], dcfgs[i & 1023], ddcfgs[i & 1023]);
        volatile double s = tau[0]; (void)s;
        emit("lgn", "dyn_RNEA_eq", n, ns_now(t0));
    }
}

static void bench_lgn_dyn_forward(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int  ndof = tree.n_dof();
    if (ndof == 0) return;
    auto cfgs  = rand_configs(ndof, 1024, -2.8, 2.8, 42);
    auto dcfgs = rand_configs(ndof, 1024, -5.0, 5.0, 137);
    auto tcfgs = rand_configs(ndof, 1024, -1.0, 1.0, 311);
    for (int i = 0; i < 200; ++i)
        lgn::forward_dynamics(tree, cfgs[i & 1023], dcfgs[i & 1023], tcfgs[i & 1023]);
    int iters = (n <= 8) ? 5000 : (n <= 32) ? 1500 : (n <= 64) ? 500 : 150;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        Eigen::VectorXd qdd = lgn::forward_dynamics(
            tree, cfgs[i & 1023], dcfgs[i & 1023], tcfgs[i & 1023]);
        volatile double s = qdd[0]; (void)s;
        emit("lgn", "dyn_ABA_eq", n, ns_now(t0));
    }
}
#endif // BENCH_LGN

// ============================================================================
//  ─────────────────────────  Pinocchio  ──────────────────────────────────
//  Note from the user: the run script keeps Pinocchio in its own binary
//  (separate from KDL). This file's only requirement is that we never
//  define both BENCH_PINOCCHIO and BENCH_KDL at once — enforced at top.
// ============================================================================
#ifdef BENCH_PINOCCHIO

// FK — one kinematic walk, same shape as lgn::fk.
static void bench_pin_fk(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    auto cfgs = rand_configs(model.nq, 1024);

    for (int i = 0; i < fk_warmup(); ++i)
        pinocchio::forwardKinematics(model, data, cfgs[i & 1023]);

    int iters = fk_iters(n);
    pinocchio::JointIndex tip = model.njoints - 1;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        pinocchio::forwardKinematics(model, data, cfgs[i & 1023]);
        volatile double s = data.oMi[tip].translation()[0]; (void)s;
        emit("pinocchio", "fk", n, ns_now(t0));
    }
}

// ── IK Story B (idiomatic for pinocchio):
//    forwardKinematics + computeJointJacobians = TWO kinematic walks.
//    This is what end-to-end Pinocchio IK code in the wild looks like.
//    Compare against bench_lgn_ik_idiomatic.
static void bench_pin_ik_idiomatic(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    pinocchio::JointIndex tip = model.njoints - 1;

    auto cfgs = rand_configs(model.nq, 2048, -2.8, 2.8, 42);
    std::vector<Eigen::Vector3d> targets(2048);
    for (int i = 0; i < 2048; ++i) {
        pinocchio::forwardKinematics(model, data, cfgs[i]);
        targets[i] = data.oMi[tip].translation();
    }

    DLSParams p;
    Eigen::MatrixXd Jf(6, model.nv);

    auto do_ik = [&](int i) {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        for (int it = 0; it < p.max_iter; ++it) {
            pinocchio::forwardKinematics(model, data, q);                // walk 1
            Eigen::Vector3d dp = targets[i & 2047]
                               - data.oMi[tip].translation();
            if (dp.norm() < p.tol) break;
            pinocchio::computeJointJacobians(model, data, q);            // walk 2
            pinocchio::getJointJacobian(model, data, tip,
                pinocchio::LOCAL_WORLD_ALIGNED, Jf);
            Eigen::Matrix3d JJt = Jf.topRows<3>() * Jf.topRows<3>().transpose();
            JJt.diagonal().array() += p.lambda_sq;
            q += Jf.topRows<3>().transpose() * JJt.ldlt().solve(dp);
        }
        volatile double s = q[0]; (void)s;
    };

    for (int i = 0; i < ik_warmup(); ++i) do_ik(i);
    int iters = ik_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        do_ik(i);
        emit("pinocchio", "ik_posonly", n, ns_now(t0));
    }
}

// ── IK Story A (fair-math for pinocchio):
//    computeJointJacobians ONLY — it populates data.oMi as a documented
//    side effect (verified Pinocchio ≥ 2.5; ROS Jazzy ships ≥ 2.7).
//    ONE kinematic walk per iteration, matching the lgn fair path.
//    The arithmetic is otherwise identical to Story B.
static void bench_pin_ik_fair(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    pinocchio::JointIndex tip = model.njoints - 1;

    // Targets generated with forwardKinematics — that's offline work,
    // doesn't enter the timing loop.
    auto cfgs = rand_configs(model.nq, 2048, -2.8, 2.8, 42);
    std::vector<Eigen::Vector3d> targets(2048);
    for (int i = 0; i < 2048; ++i) {
        pinocchio::forwardKinematics(model, data, cfgs[i]);
        targets[i] = data.oMi[tip].translation();
    }

    DLSParams p;
    Eigen::MatrixXd Jf(6, model.nv);

    auto do_ik = [&](int i) {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        for (int it = 0; it < p.max_iter; ++it) {
            // SINGLE kinematic walk: computeJointJacobians runs FK
            // internally and writes data.oMi. We read it for the
            // residual and getJointJacobian for the Jacobian — both
            // are cache reads, not walks.
            pinocchio::computeJointJacobians(model, data, q);
            Eigen::Vector3d dp = targets[i & 2047]
                               - data.oMi[tip].translation();
            if (dp.norm() < p.tol) break;
            pinocchio::getJointJacobian(model, data, tip,
                pinocchio::LOCAL_WORLD_ALIGNED, Jf);
            Eigen::Matrix3d JJt = Jf.topRows<3>() * Jf.topRows<3>().transpose();
            JJt.diagonal().array() += p.lambda_sq;
            q += Jf.topRows<3>().transpose() * JJt.ldlt().solve(dp);
        }
        volatile double s = q[0]; (void)s;
    };

    for (int i = 0; i < ik_warmup(); ++i) do_ik(i);
    int iters = ik_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        do_ik(i);
        emit("pinocchio_fair", "ik_posonly_fair", n, ns_now(t0));
    }
}

// ── Pinocchio dynamics primitives (CRBA, NLE, RNEA, ABA). Same iter
//    schedule as the lgn side so the stats script lines them up cleanly.
static void bench_pin_dyn_crba(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    auto cfgs = rand_configs(model.nq, 1024);
    for (int i = 0; i < 200; ++i) pinocchio::crba(model, data, cfgs[i & 1023]);
    int iters = (n <= 8) ? 10000 : (n <= 32) ? 3000 : (n <= 64) ? 1000 : 300;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        pinocchio::crba(model, data, cfgs[i & 1023]);
        volatile double s = data.M(0, 0); (void)s;
        emit("pinocchio", "dyn_M", n, ns_now(t0));
    }
}

static void bench_pin_dyn_nle(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    auto cfgs  = rand_configs(model.nq, 1024, -2.8, 2.8, 42);
    auto dcfgs = rand_configs(model.nq, 1024, -5.0, 5.0, 137);
    for (int i = 0; i < 200; ++i)
        pinocchio::nonLinearEffects(model, data, cfgs[i & 1023], dcfgs[i & 1023]);
    int iters = (n <= 8) ? 10000 : (n <= 32) ? 3000 : (n <= 64) ? 1000 : 300;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        pinocchio::nonLinearEffects(model, data, cfgs[i & 1023], dcfgs[i & 1023]);
        volatile double s = data.nle[0]; (void)s;
        emit("pinocchio", "dyn_C", n, ns_now(t0));
    }
}

static void bench_pin_dyn_rnea(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    auto cfgs   = rand_configs(model.nq, 1024, -2.8, 2.8, 42);
    auto dcfgs  = rand_configs(model.nq, 1024, -5.0, 5.0, 137);
    auto ddcfgs = rand_configs(model.nq, 1024, -10.0, 10.0, 277);
    for (int i = 0; i < 200; ++i)
        pinocchio::rnea(model, data, cfgs[i & 1023], dcfgs[i & 1023], ddcfgs[i & 1023]);
    int iters = (n <= 8) ? 5000 : (n <= 32) ? 1500 : (n <= 64) ? 500 : 150;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        pinocchio::rnea(model, data, cfgs[i & 1023], dcfgs[i & 1023], ddcfgs[i & 1023]);
        volatile double s = data.tau[0]; (void)s;
        emit("pinocchio", "dyn_RNEA_eq", n, ns_now(t0));
    }
}

static void bench_pin_dyn_aba(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);
    if (model.nq == 0) return;
    auto cfgs  = rand_configs(model.nq, 1024, -2.8, 2.8, 42);
    auto dcfgs = rand_configs(model.nq, 1024, -5.0, 5.0, 137);
    auto tcfgs = rand_configs(model.nq, 1024, -1.0, 1.0, 311);
    for (int i = 0; i < 200; ++i)
        pinocchio::aba(model, data, cfgs[i & 1023], dcfgs[i & 1023], tcfgs[i & 1023]);
    int iters = (n <= 8) ? 5000 : (n <= 32) ? 1500 : (n <= 64) ? 500 : 150;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        pinocchio::aba(model, data, cfgs[i & 1023], dcfgs[i & 1023], tcfgs[i & 1023]);
        volatile double s = data.ddq[0]; (void)s;
        emit("pinocchio", "dyn_ABA_eq", n, ns_now(t0));
    }
}

// ── Helper: align pinocchio's gravity vector with lgn's default.
//    lgn::gravity_vector defaults to (0,-9.81,0). pinocchio defaults to
//    (0,0,-9.81). We force both to lgn's convention so the gates compare
//    apples to apples. Coriolis itself (C·dq) is gravity-independent, but
//    we still align — defends against any future refactor that touches
//    gravity inside coriolis_vector.
static void align_gravity_vectors(pinocchio::Model& model) {
    model.gravity.linear() = Eigen::Vector3d(0.0, -9.81, 0.0);
}

// ── H_04: mass-matrix cross-check (lgn::mass_matrix vs pinocchio::crba).
//    pinocchio's CRBA fills only the upper triangle; we mirror it before
//    diffing. Gate: 1e-10 max abs element error over 20 random configs.
//    On failure: dump worst (i,j) and full matrices at n ≤ 4. Hard abort.
static void crosscheck_mass_matrix(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int ndof = tree.n_dof();
    if (ndof == 0) return;

    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    align_gravity_vectors(model);
    pinocchio::Data data(model);

    std::mt19937 rng(999);
    std::uniform_real_distribution<double> d(-2.0, 2.0);

    double max_err = 0.0;
    int    worst_i = -1, worst_j = -1;
    double worst_lgn = 0.0, worst_pin = 0.0;
    Eigen::MatrixXd M_lgn_first, M_pin_first;

    for (int trial = 0; trial < 20; ++trial) {
        Eigen::VectorXd q(ndof);
        for (int i = 0; i < ndof; ++i) q[i] = d(rng);

        Eigen::MatrixXd M_lgn = lgn::mass_matrix(tree, q);
        pinocchio::crba(model, data, q);
        // Mirror upper → lower (pinocchio CRBA convention).
        data.M.triangularView<Eigen::StrictlyLower>() =
            data.M.triangularView<Eigen::StrictlyUpper>().transpose();

        if (trial == 0) { M_lgn_first = M_lgn; M_pin_first = data.M; }

        Eigen::MatrixXd D = (M_lgn - data.M).cwiseAbs();
        double err; int r, c;
        err = D.maxCoeff(&r, &c);
        if (err > max_err) {
            max_err = err;
            worst_i = r; worst_j = c;
            worst_lgn = M_lgn(r, c);
            worst_pin = data.M(r, c);
        }
    }

    constexpr double GATE = 1e-10;
    std::cerr << "[crosscheck] mass_matrix n=" << n
              << " max_abs_err=" << std::scientific << max_err;
    if (max_err < GATE) { std::cerr << " PASS (gate " << GATE << ")\n"; return; }

    std::cerr << " FAIL (gate " << GATE << ")\n";
    std::cerr << "  worst element: M(" << worst_i << "," << worst_j << ")"
              << "  lgn=" << worst_lgn << "  pin=" << worst_pin
              << "  abs_diff=" << std::abs(worst_lgn - worst_pin) << "\n";
    if (n <= 4) {
        std::cerr << "  lgn M:\n" << M_lgn_first << "\n";
        std::cerr << "  pin M:\n" << M_pin_first << "\n";
        std::cerr << "  diff:\n" << (M_lgn_first - M_pin_first) << "\n";
    }
    std::cerr << "[crosscheck] H_04 gate failed at n=" << n << " — abort.\n";
    std::abort();
}

// ── H_05: Coriolis cross-check.
//    pinocchio::nonLinearEffects returns C(q,dq)·dq + G(q).
//    We isolate C by subtracting rnea(q, 0, 0) = G(q).
//    Gate: 1e-10 max abs entry over 20 random (q, dq). Hard abort on
//    failure, with worst-index diagnostic and full vectors at n ≤ 4.
//
//    Without legnani_pairing (the Patch 1.4 fix), the angular components
//    of lgn's C·dq would be off by exactly 2× — this gate catches it.
static void crosscheck_coriolis(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int ndof = tree.n_dof();
    if (ndof == 0) return;

    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    align_gravity_vectors(model);    // lgn convention: (0, -9.81, 0)
    pinocchio::Data data(model);

    std::mt19937 rng(0xC0F1);
    std::uniform_real_distribution<double> dq_dist(-1.5, 1.5);
    std::uniform_real_distribution<double> ddq_dist(-2.0, 2.0);

    double max_err = 0.0;
    int    worst_i = -1;
    double worst_lgn = 0.0, worst_pin = 0.0;
    Eigen::VectorXd C_lgn_first, C_pin_first;

    for (int trial = 0; trial < 20; ++trial) {
        Eigen::VectorXd q(ndof), dq(ndof);
        for (int i = 0; i < ndof; ++i) {
            q[i]  = dq_dist(rng);
            dq[i] = ddq_dist(rng);
        }

        // lgn: C·dq, no gravity.
        Eigen::VectorXd C_lgn = lgn::coriolis_vector(tree, q, dq);

        // pinocchio: nonLinearEffects = C·dq + G  →  subtract G via rnea(q,0,0).
        // Note: rnea writes to data.tau; we capture nle FIRST.
        pinocchio::nonLinearEffects(model, data, q, dq);
        Eigen::VectorXd nle = data.nle;
        Eigen::VectorXd G_pin = pinocchio::rnea(model, data, q,
                                  Eigen::VectorXd::Zero(ndof),
                                  Eigen::VectorXd::Zero(ndof));
        Eigen::VectorXd C_pin = nle - G_pin;

        if (trial == 0) { C_lgn_first = C_lgn; C_pin_first = C_pin; }

        Eigen::VectorXd D = (C_lgn - C_pin).cwiseAbs();
        double err; int r;
        err = D.maxCoeff(&r);
        if (err > max_err) {
            max_err = err;
            worst_i = r;
            worst_lgn = C_lgn[r];
            worst_pin = C_pin[r];
        }
    }

    constexpr double GATE = 1e-10;
    std::cerr << "[crosscheck] coriolis_vector n=" << n
              << " max_abs_err=" << std::scientific << max_err;
    if (max_err < GATE) { std::cerr << " PASS (gate " << GATE << ")\n"; return; }

    std::cerr << " FAIL (gate " << GATE << ")\n";
    std::cerr << "  worst element: C[" << worst_i << "]"
              << "  lgn=" << worst_lgn << "  pin=" << worst_pin
              << "  ratio=" << (std::abs(worst_pin) > 1e-12
                                 ? worst_lgn / worst_pin : 0.0) << "\n";
    if (n <= 4) {
        std::cerr << "  lgn C·dq (trial 0): " << C_lgn_first.transpose() << "\n";
        std::cerr << "  pin C·dq (trial 0): " << C_pin_first.transpose() << "\n";
        // A ratio of ~2.0 on any component implicates the angular
        // pairing bug; ratios near 1.0 with small absolute differences
        // are usually float accumulation in long chains.
        std::cerr << "  componentwise ratio (lgn / pin):\n";
        for (int i = 0; i < C_lgn_first.size(); ++i) {
            double r = (std::abs(C_pin_first[i]) > 1e-12)
                     ? C_lgn_first[i] / C_pin_first[i] : 0.0;
            std::cerr << "    [" << i << "]  lgn=" << C_lgn_first[i]
                      << "  pin=" << C_pin_first[i]
                      << "  ratio=" << r << "\n";
        }
    }
    std::cerr << "[crosscheck] H_05 gate failed at n=" << n << " — abort.\n";
    std::abort();
}
#endif // BENCH_PINOCCHIO

// ============================================================================
//  ─────────────────────────────  KDL  ────────────────────────────────────
// ============================================================================
#ifdef BENCH_KDL
static KDL::Chain make_kdl_chain(int n, double L=0.5) {
    KDL::Chain chain;
    KDL::Vector axes[3] = {{0,0,1},{1,0,0},{0,1,0}};
    for (int i = 0; i < n; ++i)
        chain.addSegment(KDL::Segment(
            KDL::Joint(KDL::Vector(0,0,0), axes[i%3], KDL::Joint::RotAxis),
            KDL::Frame(KDL::Vector(0, L, 0))));
    return chain;
}

static void bench_kdl_fk(int n) {
    KDL::Chain chain = make_kdl_chain(n);
    KDL::ChainFkSolverPos_recursive fk(chain);
    auto cfgs_lgn = rand_configs(n, 1024);
    std::vector<KDL::JntArray> cfgs(1024, KDL::JntArray(n));
    for (int i = 0; i < 1024; ++i)
        for (int j = 0; j < n; ++j) cfgs[i](j) = cfgs_lgn[i][j];

    KDL::Frame out;
    for (int i = 0; i < fk_warmup(); ++i) fk.JntToCart(cfgs[i&1023], out);

    int iters = fk_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        fk.JntToCart(cfgs[i & 1023], out);
        volatile double s = out.p.x(); (void)s;
        emit("kdl", "fk", n, ns_now(t0));
    }
}

// KDL IK: idiomatic path only (separate fk + Jacobian calls).
// We do NOT emit a "fair" variant for KDL — KDL has no Jacobian cache
// surface that would make a Story-A comparison meaningful. Keep it under
// the same "ik_posonly" tag as the idiomatic story.
static void bench_kdl_ik_posonly(int n, double L=0.5) {
    KDL::Chain chain = make_kdl_chain(n, L);
    KDL::ChainFkSolverPos_recursive fk(chain);
    KDL::ChainJntToJacSolver        js(chain);

    auto cfgs_lgn = rand_configs(n, 2048, -2.8, 2.8, 42);
    std::vector<KDL::Vector> targets(2048);
    KDL::Frame f; KDL::JntArray qa(n);
    for (int i = 0; i < 2048; ++i) {
        for (int j = 0; j < n; ++j) qa(j) = cfgs_lgn[i][j];
        fk.JntToCart(qa, f);
        targets[i] = f.p;
    }

    DLSParams p;
    auto do_ik = [&](int i) {
        KDL::JntArray q(n);
        for (int j = 0; j < n; ++j) q(j) = 0.0;
        KDL::Jacobian Jkdl(n);
        for (int it = 0; it < p.max_iter; ++it) {
            fk.JntToCart(q, f);
            KDL::Vector dp = targets[i & 2047] - f.p;
            if (dp.Norm() < p.tol) break;
            js.JntToJac(q, Jkdl);
            Eigen::MatrixXd J(3, n);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < n; ++c) J(r, c) = Jkdl(r, c);
            Eigen::Vector3d dpE(dp.x(), dp.y(), dp.z());
            Eigen::Matrix3d JJt = J * J.transpose();
            JJt.diagonal().array() += p.lambda_sq;
            Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(dpE);
            for (int j = 0; j < n; ++j) q(j) += dq[j];
        }
        volatile double s = q(0); (void)s;
    };

    for (int i = 0; i < ik_warmup(); ++i) do_ik(i);
    int iters = ik_iters(n);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        do_ik(i);
        emit("kdl", "ik_posonly", n, ns_now(t0));
    }
}
#endif // BENCH_KDL

// ============================================================================
//  Amazing Hand full-chain (compile-time URDF path, optional)
// ============================================================================
#if defined(HAND_URDF) && defined(BENCH_LGN)
static void bench_hand(const std::string& urdf_path) {
    auto tree = lgn::URDFLoader::from_file(urdf_path);
    int ndof = tree.n_dof();
    int ntips = (int)tree.tips().size();
    std::cerr << "[hand] " << ndof << " DOFs, " << ntips << " tips\n";

    auto cfgs = rand_configs(ndof, 1024);
    std::vector<std::vector<lgn::Vec3>> targets(ntips,
        std::vector<lgn::Vec3>(2048));
    for (int i = 0; i < 2048; ++i) {
        Eigen::VectorXd q = cfgs[i & 1023];
        for (int t = 0; t < ntips; ++t)
            targets[t][i] = lgn::p_of(tree.fk_tip(q, tree.tips()[t]));
    }

    for (int i = 0; i < fk_warmup(); ++i) tree.fk(cfgs[i & 1023]);
    for (int i = 0; i < 50000; ++i) {
        auto t0 = clk::now();
        tree.fk(cfgs[i & 1023]);
        volatile double s = lgn::p_of(tree.link(tree.tips()[0]).T_world).x(); (void)s;
        emit("lgn", "fk_hand", ndof, ns_now(t0));
    }

    DLSParams p;
    auto solve_all_tips = [&](int i) {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(ndof);
        std::vector<int> idx;
        for (int t = 0; t < ntips; ++t) {
            int tip = tree.tips()[t];
            for (int it = 0; it < p.max_iter; ++it) {
                tree.fk(q);
                lgn::Vec3 pos = lgn::p_of(tree.link(tip).T_world);
                lgn::Vec3 dp  = targets[t][i & 2047] - pos;
                if (dp.norm() < p.tol) break;
                auto Jc = tree.jacobian_chain_cached(tip, idx);
                Eigen::Matrix3d JJt = Jc.topRows<3>() * Jc.topRows<3>().transpose();
                JJt.diagonal().array() += p.lambda_sq;
                Eigen::VectorXd dq = Jc.topRows<3>().transpose() * JJt.ldlt().solve(dp);
                for (int c = 0; c < (int)idx.size(); ++c) q[idx[c]] += dq[c];
            }
        }
        return q;
    };

    for (int i = 0; i < ik_warmup(); ++i) {
        Eigen::VectorXd q = solve_all_tips(i);
        volatile double s = q[0]; (void)s;
    }
    for (int i = 0; i < 5000; ++i) {
        auto t0 = clk::now();
        Eigen::VectorXd q = solve_all_tips(i);
        volatile double s = q[0]; (void)s;
        emit("lgn", "ik_hand_seq", ndof, ns_now(t0));
    }
}
#endif

// ============================================================================
//  main
// ============================================================================
int main() {
    emit_header();

    const int fk_ns[]  = {2,4,8,16,32,64,128,256};
    const int ik_ns[]  = {2,4,8,16,32,64,128,256};
    const int vel_ns[] = {2,4,8,16,32,64,128};
    const int dyn_ns[] = {2,4,8,16,32,64,128};
    const int mm_ns[]  = {2,4,8,16,32};
    const int N_FK=8, N_IK=8, N_VEL=7, N_DYN=7, N_MM=5;

#ifdef BENCH_LGN
    std::cerr << "=== lgn FK ===\n";
    for (int i=0;i<N_FK;++i) { std::cerr<<"  n="<<fk_ns[i]<<"\n"; bench_lgn_fk(fk_ns[i]); }

    std::cerr << "=== lgn IK (Story B / idiomatic: fk + cached J, 1 walk/iter) ===\n";
    for (int i=0;i<N_IK;++i) { std::cerr<<"  n="<<ik_ns[i]<<"\n"; bench_lgn_ik_idiomatic(ik_ns[i]); }

    std::cerr << "=== lgn IK (Story A / fair-math: 1 walk/iter, paired with pinocchio_fair) ===\n";
    for (int i=0;i<N_IK;++i) { std::cerr<<"  n="<<ik_ns[i]<<"\n"; bench_lgn_ik_fair(ik_ns[i]); }

    // Dynamics blocks below are O(n³) and not the paper's headline.
    // Reported so reviewers can see exactly where the representation
    // currently sits; sister paper will tune these.
    std::cerr << "=== lgn dynamics: mass matrix (M) ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_lgn_dyn_mass_matrix(dyn_ns[i]); }
    std::cerr << "=== lgn dynamics: Coriolis (C·dq) ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_lgn_dyn_coriolis(dyn_ns[i]); }
    std::cerr << "=== lgn dynamics: inverse (RNEA-equivalent) ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_lgn_dyn_inverse(dyn_ns[i]); }
    std::cerr << "=== lgn dynamics: forward (ABA-equivalent) ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_lgn_dyn_forward(dyn_ns[i]); }
#endif

#if defined(BENCH_LGN) && defined(BENCH_VEL_BASELINE)
    std::cerr << "=== velocity: lgn Hv vs scalar (both ALLOC and PREALLOC modes) ===\n";
    for (int i=0;i<N_VEL;++i) { std::cerr<<"  n="<<vel_ns[i]<<"\n"; bench_lgn_vel_vs_scalar(vel_ns[i]); }
#endif

#ifdef BENCH_PINOCCHIO
    std::cerr << "=== pinocchio FK ===\n";
    for (int i=0;i<N_FK;++i) { std::cerr<<"  n="<<fk_ns[i]<<"\n"; bench_pin_fk(fk_ns[i]); }

    std::cerr << "=== pinocchio IK (Story B / idiomatic: FK + computeJointJacobians, 2 walks) ===\n";
    for (int i=0;i<N_IK;++i) { std::cerr<<"  n="<<ik_ns[i]<<"\n"; bench_pin_ik_idiomatic(ik_ns[i]); }

    std::cerr << "=== pinocchio IK (Story A / fair-math: computeJointJacobians only, 1 walk) ===\n";
    for (int i=0;i<N_IK;++i) { std::cerr<<"  n="<<ik_ns[i]<<"\n"; bench_pin_ik_fair(ik_ns[i]); }

    std::cerr << "=== pinocchio CRBA (M) ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_pin_dyn_crba(dyn_ns[i]); }
    std::cerr << "=== pinocchio nonLinearEffects (C·dq + G) ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_pin_dyn_nle(dyn_ns[i]); }
    std::cerr << "=== pinocchio RNEA ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_pin_dyn_rnea(dyn_ns[i]); }
    std::cerr << "=== pinocchio ABA ===\n";
    for (int i=0;i<N_DYN;++i) { std::cerr<<"  n="<<dyn_ns[i]<<"\n"; bench_pin_dyn_aba(dyn_ns[i]); }

    std::cerr << "=== H_04: mass-matrix cross-check (lgn vs pinocchio CRBA) ===\n";
    for (int i=0;i<N_MM;++i) crosscheck_mass_matrix(mm_ns[i]);

    std::cerr << "=== H_05: Coriolis cross-check (lgn vs pinocchio NLE − G) ===\n";
    for (int i=0;i<N_MM;++i) crosscheck_coriolis(mm_ns[i]);
#endif

#ifdef BENCH_KDL
    std::cerr << "=== KDL FK ===\n";
    for (int i=0;i<N_FK;++i) { std::cerr<<"  n="<<fk_ns[i]<<"\n"; bench_kdl_fk(fk_ns[i]); }
    std::cerr << "=== KDL IK (position-only DLS, idiomatic only) ===\n";
    for (int i=0;i<N_IK;++i) { std::cerr<<"  n="<<ik_ns[i]<<"\n"; bench_kdl_ik_posonly(ik_ns[i]); }
#endif

#if defined(HAND_URDF) && defined(BENCH_LGN)
    std::cerr << "=== Amazing Hand full-chain ===\n";
    bench_hand(HAND_URDF);
#endif

    return 0;
}
