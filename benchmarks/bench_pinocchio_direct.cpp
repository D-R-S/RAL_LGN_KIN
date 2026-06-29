// ============================================================================
//  benchmarks/bench_pinocchio_direct.cpp
//  Pinocchio 3.9.0 — FK and IK, chains n=1..64
//  No Google Benchmark framework (crashes under repeated fixture init).
//  Uses std::chrono::steady_clock — same methodology as bench_lgn_kdl.cpp.
//
//  Methodology (identical to bench_lgn_kdl.cpp):
//    - Same URDF topology (same axes, link lengths, inertias)
//    - Same random seed (42), same config range [-2.8, 2.8]
//    - FK: full forwardKinematics(), sink last joint frame position
//    - IK: position-only DLS, λ²=1e-4, max_iter=50, tol=1e-4
//          Targets from own FK (self-consistent, reachable)
//    - Warmup: 2000 iters before timing
//    - Same iteration counts as lgn bench
//    - Output: same CSV format as bench_lgn_kdl.cpp
//
//  Build:
//    g++ -O3 -march=native -ffast-math -std=c++17 \
//      benchmarks/bench_pinocchio_direct.cpp \
//      -I/opt/ros/jazzy/include \
//      $(pkg-config --cflags eigen3) \
//      -Wl,-rpath,/opt/ros/jazzy/lib/x86_64-linux-gnu \
//      /opt/ros/jazzy/lib/x86_64-linux-gnu/libpinocchio_parsers.so.3.9.0 \
//      /opt/ros/jazzy/lib/x86_64-linux-gnu/libpinocchio_default.so.3.9.0 \
//      /opt/ros/jazzy/lib/x86_64-linux-gnu/liburdfdom_model.so.4.0 \
//      /opt/ros/jazzy/lib/x86_64-linux-gnu/liburdfdom_world.so.4.0 \
//      /opt/ros/jazzy/lib/x86_64-linux-gnu/liburdfdom_sensor.so.4.0 \
//      -o bench_pinocchio_direct
//
//  Run:
//    taskset -c 2 ./bench_pinocchio_direct | tee results_pinocchio_direct.csv
// ============================================================================
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
using ns  = std::chrono::nanoseconds;

// ── Identical URDF generator to bench_lgn_kdl.cpp ────────────────────────────
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

// ── Timing helper: identical to bench_lgn_kdl.cpp ────────────────────────────
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
    return {mean, std::sqrt(var / iters)};
}

// ── Random configs — seed=42, range=[-2.8,2.8], identical to lgn bench ───────
static std::vector<Eigen::VectorXd> rand_configs(int nq, int count, int seed=42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-2.8, 2.8);
    std::vector<Eigen::VectorXd> out(count);
    for (auto& q : out) {
        q.resize(nq);
        for (int i=0;i<nq;++i) q[i]=d(rng);
    }
    return out;
}

static void emit(const std::string& bench, int n, double mean_ns, double stddev_ns) {
    std::cout << std::fixed << std::setprecision(2)
              << "pinocchio," << bench << "," << n << ","
              << mean_ns << "," << stddev_ns << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pinocchio FK
// ─────────────────────────────────────────────────────────────────────────────
static void bench_pin_fk(int n) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n), model);
    pinocchio::Data data(model);

    if (model.nq == 0) { emit("fk", n, 0, 0); return; }

    // Same seed, same count as lgn bench
    auto cfgs = rand_configs(model.nq, 1024);

    int iters  = (n <= 4) ? 200000 : (n <= 8) ? 100000 : (n <= 16) ? 50000 : (n <= 32) ? 20000 : 10000;
    int warmup = 1000;

    auto [mean, sd] = time_it([&](int i){
        pinocchio::forwardKinematics(model, data, cfgs[i & 1023]);
        volatile double s = data.oMi.back().translation()[0]; (void)s;
    }, warmup, iters);

    emit("fk", n, mean, sd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pinocchio IK — position-only DLS
//  Identical algorithm to bench_lgn_kdl.cpp lgn IK:
//    λ²=1e-4, max_iter=50, tol=1e-4
//    Targets generated from own FK (seed=42 configs)
// ─────────────────────────────────────────────────────────────────────────────
static void bench_pin_ik_posonly(int n, double L=0.5) {
    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(make_chain_urdf(n, L), model);
    pinocchio::Data data(model);

    if (model.nq == 0) { emit("ik_posonly", n, 0, 0); return; }

    pinocchio::JointIndex tip = model.njoints - 1;

    // Generate targets: same seed=42, same count as lgn bench
    auto cfgs = rand_configs(model.nq, 2048);
    std::vector<Eigen::Vector3d> targets(2048);
    for (int i=0;i<2048;++i) {
        pinocchio::forwardKinematics(model, data, cfgs[i]);
        targets[i] = data.oMi[tip].translation();
    }

    const double lsq   = 1e-4;
    const int    maxit = 50;
    const double tol   = 1e-4;

    int iters  = (n <= 4) ? 10000 : (n <= 8) ? 3000 : (n <= 16) ? 1000 : (n <= 32) ? 300 : 100;
    int warmup = 100;

    Eigen::MatrixXd Jf(6, model.nv);

    auto [mean, sd] = time_it([&](int i){
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
        for (int it=0;it<maxit;++it) {
            pinocchio::forwardKinematics(model, data, q);
            Eigen::Vector3d dp = targets[i & 2047] - data.oMi[tip].translation();
            if (dp.norm() < tol) break;
            pinocchio::computeJointJacobians(model, data, q);
            pinocchio::getJointJacobian(model, data, tip,
                pinocchio::LOCAL_WORLD_ALIGNED, Jf);
            Eigen::MatrixXd J = Jf.topRows<3>();
            Eigen::MatrixXd JJt = J * J.transpose();
            JJt.diagonal().array() += lsq;
            q += J.transpose() * JJt.ldlt().solve(dp);
        }
        volatile double s = q[0]; (void)s;
    }, warmup, iters);

    emit("ik_posonly", n, mean, sd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "solver,benchmark,n,mean_ns,stddev_ns\n";

    const int ns_fk[] = {1,2,4,8,16,32,64};
    const int ns_ik[] = {2,4,8,16,32,64};
    const int N_FK    = 7;
    const int N_IK    = 6;

    std::cerr << "Pinocchio FK...\n";
    for (int i=0;i<N_FK;++i) bench_pin_fk(ns_fk[i]);

    std::cerr << "Pinocchio IK (position-only DLS)...\n";
    for (int i=0;i<N_IK;++i) bench_pin_ik_posonly(ns_ik[i]);

    return 0;
}
