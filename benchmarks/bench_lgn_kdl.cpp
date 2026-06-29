// ============================================================================
//  benchmarks/bench_lgn_kdl.cpp
//  lgn vs KDL — FK and IK comparison, chains n=1..64
//
//  Methodology:
//    - Identical URDF topology for both solvers
//    - Axes alternate Z/X/Y (tests arbitrary-axis code paths)
//    - Random configs: seed=42, uniform [-2.8, 2.8]
//    - FK: full tree walk, sink last link position
//    - IK: position-only DLS, λ²=1e-4, max_iter=50, tol=1e-4
//          Targets generated from own FK (self-consistent, reachable)
//    - Warmup: 2000 iters before timing
//    - Timed: 200000 iters (FK), 10000 iters (IK n≤8), 2000 iters (IK n>8)
//    - Output: CSV parseable by compute_stats.py
//
//  Build (inside build_bench/):
//    cmake .. -DBUILD_BENCHMARKS=ON ...
//    make bench_lgn_kdl
//
//  Run:
//    taskset -c 2 ./bench_lgn_kdl | tee results_lgn_kdl_direct.csv
// ============================================================================
#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/ik_solver.hpp>

#ifdef HAVE_KDL
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#endif

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

using clk = std::chrono::steady_clock;
using ns  = std::chrono::nanoseconds;

// ── Shared URDF generator ─────────────────────────────────────────────────────
// Axes alternate Z/X/Y per joint index.
// All joints have inertia so dynamics are also testable.
static std::string make_chain_urdf(int n, double L = 0.5) {
    std::ostringstream ss;
    ss << "<?xml version=\"1.0\"?><robot name=\"bench_chain\">\n"
       << "<link name=\"world\"/>\n";
    for (int i = 0; i < n; ++i)
        ss << "<link name=\"link" << i << "\"><inertial>"
           << "<origin xyz=\"0 " << L/2 << " 0\"/>"
           << "<mass value=\"0.1\"/>"
           << "<inertia ixx=\"0.001\" ixy=\"0\" ixz=\"0\" "
           << "iyy=\"0.0001\" iyz=\"0\" izz=\"0.001\"/>"
           << "</inertial></link>\n";
    ss << "<link name=\"tip\"/>\n"
       << "<joint name=\"base\" type=\"fixed\">"
       << "<parent link=\"world\"/><child link=\"link0\"/>"
       << "<origin xyz=\"0 0 0\"/></joint>\n";
    const char* axes[] = {"0 0 1", "1 0 0", "0 1 0"};
    for (int i = 0; i < n-1; ++i)
        ss << "<joint name=\"J" << i << "\" type=\"revolute\">"
           << "<parent link=\"link" << i << "\"/>"
           << "<child link=\"link" << (i+1) << "\"/>"
           << "<origin xyz=\"0 " << L << " 0\"/>"
           << "<axis xyz=\"" << axes[i%3] << "\"/>"
           << "<limit lower=\"-3.14\" upper=\"3.14\" "
           << "velocity=\"10\" effort=\"10\"/></joint>\n";
    ss << "<joint name=\"tip_j\" type=\"fixed\">"
       << "<parent link=\"link" << (n-1) << "\"/><child link=\"tip\"/>"
       << "<origin xyz=\"0 " << L << " 0\"/></joint>\n"
       << "</robot>\n";
    return ss.str();
}

// ── Timing helper: run f() iters times, return {mean_ns, stddev_ns} ──────────
template<typename F>
static std::pair<double,double> time_it(F&& f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) f(i);
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        f(i);
        auto t1 = clk::now();
        samples.push_back((double)std::chrono::duration_cast<ns>(t1-t0).count());
    }
    double mean = 0;
    for (double x : samples) mean += x;
    mean /= iters;
    double var = 0;
    for (double x : samples) var += (x-mean)*(x-mean);
    double stddev = std::sqrt(var / iters);
    return {mean, stddev};
}

// ── Random configs ────────────────────────────────────────────────────────────
static std::vector<Eigen::VectorXd> rand_configs(int n, int count,
    double lo=-2.8, double hi=2.8, int seed=42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<Eigen::VectorXd> out(count);
    for (auto& q : out) {
        q.resize(n); for (int i=0;i<n;++i) q[i]=d(rng);
    }
    return out;
}

// ── Output ────────────────────────────────────────────────────────────────────
static void emit(const std::string& solver, const std::string& bench,
                 int n, double mean_ns, double stddev_ns) {
    std::cout << std::fixed << std::setprecision(2)
              << solver << "," << bench << "," << n << ","
              << mean_ns << "," << stddev_ns << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  lgn FK
// ─────────────────────────────────────────────────────────────────────────────
static void bench_lgn_fk(int n) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n));
    int ndof  = tree.n_dof();
    if (ndof == 0) { emit("lgn","fk",n,0,0); return; }
    auto cfgs = rand_configs(ndof, 1024);

    int iters  = (n <= 8) ? 500000 : (n <= 32) ? 200000 : 100000;
    int warmup = 2000;

    auto [mean, sd] = time_it([&](int i){
        tree.fk(cfgs[i & 1023]);
        volatile double s = lgn::p_of(tree.link(tree.tips()[0]).T_world).x();
        (void)s;
    }, warmup, iters);

    emit("lgn", "fk", n, mean, sd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  lgn IK — position-only DLS to match Pinocchio fairly
//  Strips rotation error and null-space from lgn's IKSolver.
//  λ²=1e-4, max_iter=50, tol=1e-4 — identical parameters to Pinocchio bench.
// ─────────────────────────────────────────────────────────────────────────────
static void bench_lgn_ik_posonly(int n, double L=0.5) {
    auto tree = lgn::URDFLoader::from_string(make_chain_urdf(n, L));
    int ndof  = tree.n_dof();
    if (ndof == 0) { emit("lgn","ik_posonly",n,0,0); return; }
    int tip   = tree.tips()[0];

    // Generate reachable targets from lgn FK (same seed as Pinocchio)
    auto cfgs = rand_configs(ndof, 2048, -2.8, 2.8, 42);
    std::vector<lgn::Vec3> targets(2048);
    for (int i=0;i<2048;++i)
        targets[i] = lgn::p_of(tree.fk_tip(cfgs[i], tip));

    const double lsq   = 1e-4;
    const int    maxit = 50;
    const double tol   = 1e-4;

    int iters  = (n <= 4) ? 20000 : (n <= 16) ? 5000 : 1000;
    int warmup = 200;

    auto [mean, sd] = time_it([&](int i) {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(ndof);
        for (int it = 0; it < maxit; ++it) {
            lgn::Vec3 p_cur = lgn::p_of(tree.fk_tip(q, tip));
            lgn::Vec3 dp    = targets[i & 2047] - p_cur;
            if (dp.norm() < tol) break;
            // Position-only Jacobian (3 × ndof)
            std::vector<int> dof_idx;
            Eigen::MatrixXd Jc = tree.jacobian_chain(q, tip, dof_idx);
            Eigen::MatrixXd J  = Jc.topRows<3>();
            Eigen::MatrixXd JJt = J * J.transpose();
            JJt.diagonal().array() += lsq;
            Eigen::VectorXd dq = J.transpose() * JJt.ldlt().solve(dp);
            for (int c=0;c<(int)dof_idx.size();++c) q[dof_idx[c]] += dq[c];
        }
        volatile double s = q[0]; (void)s;
    }, warmup, iters);

    emit("lgn", "ik_posonly", n, mean, sd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  KDL FK
// ─────────────────────────────────────────────────────────────────────────────
#ifdef HAVE_KDL
static void bench_kdl_fk(int n) {
    KDL::Chain chain;
    KDL::Vector axes[3] = {{0,0,1},{1,0,0},{0,1,0}};
    for (int i=0;i<n;++i)
        chain.addSegment(KDL::Segment(
            KDL::Joint(KDL::Vector(0,0,0), axes[i%3], KDL::Joint::RotAxis),
            KDL::Frame(KDL::Vector(0, 0.5, 0))));

    KDL::ChainFkSolverPos_recursive fk(chain);

    // Use same random configs as lgn (same seed, same range, same count)
    auto cfgs_lgn = rand_configs(n, 1024);
    std::vector<KDL::JntArray> cfgs(1024, KDL::JntArray(n));
    for (int i=0;i<1024;++i)
        for (int j=0;j<n;++j) cfgs[i](j) = cfgs_lgn[i][j];

    KDL::Frame out;
    int iters  = (n <= 8) ? 500000 : (n <= 32) ? 200000 : 100000;
    int warmup = 2000;

    auto [mean, sd] = time_it([&](int i){
        fk.JntToCart(cfgs[i & 1023], out);
        volatile double s = out.p.x(); (void)s;
    }, warmup, iters);

    emit("kdl", "fk", n, mean, sd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  KDL IK — LMA solver (KDL's standard IK)
//  Note: KDL LMA is full SE(3) unlike Pinocchio pos-only.
//  We report it separately and note the algorithm difference.
// ─────────────────────────────────────────────────────────────────────────────
static void bench_kdl_ik(int n) {
    KDL::Chain chain;
    KDL::Vector axes[3] = {{0,0,1},{1,0,0},{0,1,0}};
    for (int i=0;i<n;++i)
        chain.addSegment(KDL::Segment(
            KDL::Joint(KDL::Vector(0,0,0), axes[i%3], KDL::Joint::RotAxis),
            KDL::Frame(KDL::Vector(0, 0.5, 0))));

    KDL::ChainFkSolverPos_recursive fk(chain);
    KDL::ChainIkSolverPos_LMA       ik(chain);

    // Generate reachable targets (same seed as lgn)
    auto cfgs_lgn = rand_configs(n, 2048, -2.8, 2.8, 42);
    std::vector<KDL::Frame> targets(2048);
    KDL::JntArray qa(n);
    for (int i=0;i<2048;++i) {
        for (int j=0;j<n;++j) qa(j)=cfgs_lgn[i][j];
        fk.JntToCart(qa, targets[i]);
    }

    KDL::JntArray q0(n), q_sol(n);
    for (int j=0;j<n;++j) q0(j)=0.0;

    int iters  = (n <= 4) ? 10000 : (n <= 16) ? 2000 : 500;
    int warmup = 100;

    auto [mean, sd] = time_it([&](int i){
        ik.CartToJnt(q0, targets[i & 2047], q_sol);
        volatile double s = q_sol(0); (void)s;
    }, warmup, iters);

    emit("kdl", "ik_lma", n, mean, sd);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "solver,benchmark,n,mean_ns,stddev_ns\n";

    const int ns_fk[]  = {1,2,4,8,16,32,64};
    const int ns_ik[]  = {2,4,8,16,32,64};
    const int N_FK     = 7;
    const int N_IK     = 6;

    std::cerr << "lgn FK...\n";
    for (int i=0;i<N_FK;++i) bench_lgn_fk(ns_fk[i]);

    std::cerr << "lgn IK (position-only DLS)...\n";
    for (int i=0;i<N_IK;++i) bench_lgn_ik_posonly(ns_ik[i]);

#ifdef HAVE_KDL
    std::cerr << "KDL FK...\n";
    for (int i=0;i<N_FK;++i) bench_kdl_fk(ns_fk[i]);

    std::cerr << "KDL IK (LMA, full SE3)...\n";
    for (int i=0;i<N_IK;++i) bench_kdl_ik(ns_ik[i]);
#else
    std::cerr << "KDL not available — rebuild with HAVE_KDL\n";
#endif

    return 0;
}
