// ============================================================================
//  tools/twin_sim.cpp
//
//  Side-by-side visual demo: lgn vs pinocchio computing the SAME physics.
//
//  WHAT YOU SEE
//  ------------
//  Two pendulums falling under gravity, drawn next to each other in a
//  gnuplot window. Same URDF, same initial condition, same integrator,
//  same dt. One uses lgn's 4x4 Legnani forward dynamics; the other uses
//  pinocchio's ABA. In the corner: ||q_lgn - q_pin||_inf.
//
//  IF THE MATH IS RIGHT the two stick figures move identically and the
//  residual stays at fp64 noise (~1e-10 region). If anything is wrong
//  they drift apart visibly and the residual grows.
//
//  This is the demo for the paper video and for co-authors who can't
//  read the bench harness but can read a moving picture.
//
//  BUILD
//  -----
//      g++ -O2 -std=c++17 tools/twin_sim.cpp \
//          -I /path/to/lgn/include \
//          $(pkg-config --cflags --libs pinocchio) \
//          -o twin_sim
//
//  RUN
//  ---
//      ./twin_sim                  # n=3, 5s, dt=1ms
//      ./twin_sim --n 7
//      ./twin_sim --duration 10
//      ./twin_sim --no-plot        # CSV to stdout, no gnuplot
//
//  FAIRNESS
//  --------
//  - SAME URDF string handed to both engines (one make_chain_urdf call).
//  - SAME initial condition (q0 = pi/4 on every joint, dq0 = 0).
//  - SAME gravity vector (0, -9.81, 0). lgn's default is already this;
//    pinocchio's default is (0, 0, -9.81), so we override to match.
//  - SAME integrator: semi-implicit Euler. Same dt on both sides.
//  - Each engine integrates its OWN state independently from t=0.
//    They are never resynced. The residual is therefore an honest
//    measurement of cumulative numerical agreement.
// ============================================================================

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/dynamics.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/aba.hpp>

// ── Identical robot for both engines. Revolute, axes cycling Z/X/Y so
//    the chain isn't trivially planar; equal link lengths; equal mass.
static std::string make_chain_urdf(int n, double L = 0.5) {
    std::ostringstream ss;
    ss << "<?xml version=\"1.0\"?><robot name=\"twin\">\n<link name=\"world\"/>\n";
    for (int i = 0; i < n; ++i)
        ss << "<link name=\"l" << i << "\"><inertial>"
           << "<origin xyz=\"0 " << L/2 << " 0\"/><mass value=\"0.1\"/>"
           << "<inertia ixx=\"0.001\" ixy=\"0\" ixz=\"0\" "
           << "iyy=\"0.0001\" iyz=\"0\" izz=\"0.001\"/></inertial></link>\n";
    ss << "<link name=\"tip\"/>\n"
       << "<joint name=\"base\" type=\"fixed\"><parent link=\"world\"/>"
       << "<child link=\"l0\"/><origin xyz=\"0 0 0\"/></joint>\n";
    const char* axes[] = {"0 0 1", "1 0 0", "0 1 0"};
    for (int i = 0; i < n - 1; ++i)
        ss << "<joint name=\"J" << i << "\" type=\"revolute\">"
           << "<parent link=\"l" << i << "\"/><child link=\"l" << (i+1) << "\"/>"
           << "<origin xyz=\"0 " << L << " 0\"/><axis xyz=\"" << axes[i%3] << "\"/>"
           << "<limit lower=\"-3.14\" upper=\"3.14\" velocity=\"10\" effort=\"10\"/>"
           << "</joint>\n";
    ss << "<joint name=\"tip_j\" type=\"fixed\"><parent link=\"l" << (n-1)
       << "\"/><child link=\"tip\"/><origin xyz=\"0 " << L << " 0\"/></joint>\n"
       << "</robot>\n";
    return ss.str();
}

// ── 2D projection of every joint origin + final tip position, for gnuplot.
//    Both engines render the SAME node set:
//      [base origin, joint 1 origin, joint 2 origin, ..., last joint origin, tip]
//    The base is the world origin. Each joint origin is the placement of
//    that joint in the world. The tip is the end of the last link, which
//    is the visually-interesting moving point.
//
//    Without this, the two engines disagree on which auxiliary frames to
//    draw: lgn's KinematicTree enumerates every URDF <link> (including the
//    "tip" fixed-frame link and the "world" root), while pinocchio's
//    data.oMi enumerates joints, collapsing fixed joints into the parent
//    body. Same physics, different draw lists → an apparent "extra dot"
//    on the lgn side. We avoid the mismatch by reading exactly the same
//    geometric quantities on both sides.

struct StickFigure {
    std::vector<double> xs, ys;
};

static StickFigure poses_lgn(lgn::KinematicTree& tree,
                             const Eigen::VectorXd& q) {
    tree.fk(q);
    StickFigure f;
    f.xs.push_back(0.0); f.ys.push_back(0.0);     // base (world origin)
    // Walk the joints in order; each joint's child-link T_world gives
    // the joint origin in the world frame.
    for (int ji = 0; ji < tree.n_joints(); ++ji) {
        const auto& jt = tree.joint(ji);
        if (jt.type == lgn::JointType::Fixed) continue;
        const auto& T = tree.link(jt.child_link).T_world;
        lgn::Vec3 p = lgn::p_of(T);
        f.xs.push_back(p.x());
        f.ys.push_back(p.y());
    }
    // Tip: world position of the "tip" link, which is the URDF's terminal
    // fixed-frame attached to the last moving link.
    int tip = tree.link_index("tip");
    lgn::Vec3 ptip = lgn::p_of(tree.link(tip).T_world);
    f.xs.push_back(ptip.x());
    f.ys.push_back(ptip.y());
    return f;
}

static StickFigure poses_pin(pinocchio::Model& model, pinocchio::Data& data,
                             const Eigen::VectorXd& q) {
    pinocchio::forwardKinematics(model, data, q);
    StickFigure f;
    f.xs.push_back(0.0); f.ys.push_back(0.0);     // base (world origin)
    // pinocchio joint 0 is the universe joint (world origin); skip it.
    // Joints 1..njoints-1 are the actual moving joints.
    for (pinocchio::JointIndex j = 1; j < (pinocchio::JointIndex)model.njoints; ++j) {
        const auto& t = data.oMi[j].translation();
        f.xs.push_back(t[0]);
        f.ys.push_back(t[1]);
    }
    // Tip: same convention as lgn. The terminal link is the last moving
    // joint's child body, displaced by L along its local Y. We read it
    // off the last joint's placement times the link length.
    // For our chain, the joint at index njoints-1 has its child body
    // origin at the last moving joint frame; the tip sits one link L
    // along that body's Y axis.
    const double L = 0.5;
    pinocchio::JointIndex last = model.njoints - 1;
    Eigen::Vector3d tip_local(0.0, L, 0.0);
    Eigen::Vector3d tip_world = data.oMi[last].act(tip_local);
    f.xs.push_back(tip_world.x());
    f.ys.push_back(tip_world.y());
    return f;
}

int main(int argc, char** argv) {
    int    n        = 3;
    double duration = 5.0;
    double dt       = 1e-3;
    bool   plot     = true;
    double q0_val   = M_PI / 4.0;   // starting joint angle, all joints

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--n"        && i+1 < argc) n        = std::atoi(argv[++i]);
        else if (a == "--duration" && i+1 < argc) duration = std::atof(argv[++i]);
        else if (a == "--dt"       && i+1 < argc) dt       = std::atof(argv[++i]);
        else if (a == "--q0"       && i+1 < argc) q0_val   = std::atof(argv[++i]);
        else if (a == "--no-plot")               plot     = false;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--n N] [--duration S] [--dt S] [--q0 RAD] [--no-plot]\n";
            return 1;
        }
    }

    // ── Build both engines from the same URDF string ─────────────────────
    const std::string urdf = make_chain_urdf(n);

    auto tree = lgn::URDFLoader::from_string(urdf);
    int  ndof = tree.n_dof();
    if (ndof == 0) { std::cerr << "no DOFs\n"; return 1; }

    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(urdf, model);
    pinocchio::Data data(model);
    // Align gravity: lgn defaults to (0, -9.81, 0); pinocchio defaults
    // to (0, 0, -9.81). Force pinocchio onto lgn's convention so the
    // two engines see the SAME world.
    model.gravity.linear() = Eigen::Vector3d(0.0, -9.81, 0.0);

    if (model.nv != ndof) {
        std::cerr << "DOF mismatch: lgn=" << ndof
                  << " pin=" << model.nv << " — URDF loaded inconsistently\n";
        return 1;
    }

    // ── Identical initial condition for both ─────────────────────────────
    Eigen::VectorXd q_lgn  = Eigen::VectorXd::Constant(ndof, q0_val);
    Eigen::VectorXd dq_lgn = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd q_pin  = q_lgn;
    Eigen::VectorXd dq_pin = dq_lgn;
    Eigen::VectorXd tau    = Eigen::VectorXd::Zero(ndof);   // free fall

    // ── gnuplot pipe ─────────────────────────────────────────────────────
    FILE* gp = nullptr;
    if (plot) {
        gp = popen("gnuplot", "w");
        if (!gp) {
            std::cerr << "could not open gnuplot; rerun with --no-plot\n";
            return 1;
        }
        // One plot, two stick figures side by side. Pinocchio drawn at
        // x_offset so the figures don't overlap. Residual shown in the
        // window title bar (gnuplot updates it each frame).
        const double L = 0.5;
        const double reach = (n + 1) * L;
        const double x_offset = 2.5 * reach;
        std::fprintf(gp,
            "set term qt size 900,500 title 'twin_sim'\n"
            "set xrange [%f:%f]\n"
            "set yrange [%f:%f]\n"
            "set size ratio -1\n"
            "set grid\n"
            "unset key\n"
            "set label 1 'lgn'        at %f, %f center font ',14'\n"
            "set label 2 'pinocchio'  at %f, %f center font ',14'\n",
            -reach, x_offset + reach,
            -reach * 1.1, reach * 0.3,
            0.0,        reach * 0.25,
            x_offset,   reach * 0.25);
        std::fflush(gp);
        // Stash offset for the per-frame loop.
        std::fprintf(gp, "x_off = %f\n", x_offset);
        std::fflush(gp);
    } else {
        // Header for CSV-to-stdout mode.
        std::cout << "t,engine";
        for (int i = 0; i < ndof; ++i) std::cout << ",q" << i;
        std::cout << "\n";
    }

    // ── Sim loop ─────────────────────────────────────────────────────────
    // Semi-implicit Euler. Identical on both sides:
    //     ddq = forward_dynamics(q, dq, tau)
    //     dq  += ddq * dt
    //     q   += dq  * dt
    //
    using clk = std::chrono::steady_clock;
    auto wall_t0 = clk::now();
    int  steps = (int)std::ceil(duration / dt);
    int  draw_every = std::max(1, (int)std::round(1.0 / (60.0 * dt))); // ~60 Hz

    for (int s = 0; s < steps; ++s) {
        // lgn step
        Eigen::VectorXd ddq_lgn = lgn::forward_dynamics(tree, q_lgn, dq_lgn, tau);
        dq_lgn += ddq_lgn * dt;
        q_lgn  += dq_lgn  * dt;

        // pinocchio step
        pinocchio::aba(model, data, q_pin, dq_pin, tau);
        dq_pin += data.ddq * dt;
        q_pin  += dq_pin  * dt;

        if (s % draw_every != 0) continue;

        double t   = s * dt;
        double res = (q_lgn - q_pin).cwiseAbs().maxCoeff();

        if (plot) {
            // Build the two stick-figure polylines and pipe them to gnuplot
            // using inline data blocks. Two `plot` commands per frame:
            // one for lgn (left), one for pin (right).
            auto fl = poses_lgn(tree, q_lgn);
            auto fp = poses_pin(model, data, q_pin);

            std::fprintf(gp,
                "set title sprintf('t = %%5.2f s    ||q_{lgn} - q_{pin}||_inf = %%.2e', "
                "%f, %g)\n", t, res);
            std::fprintf(gp,
                "plot '-' with linespoints lt 1 lw 2 pt 7 ps 1.2 lc rgb '#E69F4D', "
                     "'-' with linespoints lt 1 lw 2 pt 7 ps 1.2 lc rgb '#5D8A66'\n");
            // lgn polyline
            for (size_t i = 0; i < fl.xs.size(); ++i)
                std::fprintf(gp, "%f %f\n", fl.xs[i], fl.ys[i]);
            std::fprintf(gp, "e\n");
            // pinocchio polyline (shifted right)
            for (size_t i = 0; i < fp.xs.size(); ++i)
                std::fprintf(gp, "%f %f\n", fp.xs[i] + 2.5 * (n + 1) * 0.5, fp.ys[i]);
            std::fprintf(gp, "e\n");
            std::fflush(gp);
        } else {
            std::cout << t << ",lgn";
            for (int i = 0; i < ndof; ++i) std::cout << "," << q_lgn[i];
            std::cout << "\n";
            std::cout << t << ",pin";
            for (int i = 0; i < ndof; ++i) std::cout << "," << q_pin[i];
            std::cout << "\n";
        }

        // Wall-clock pacing so the animation runs at real time.
        if (plot) {
            auto target = wall_t0 + std::chrono::duration_cast<clk::duration>(
                std::chrono::duration<double>(t));
            std::this_thread::sleep_until(target);
        }
    }

    // ── Final residual to stderr (separate from data stream) ─────────────
    double res_final = (q_lgn - q_pin).cwiseAbs().maxCoeff();
    std::cerr << "\nfinal residual ||q_lgn - q_pin||_inf = "
              << res_final << "\n";
    std::cerr << "(at " << duration << " s, " << steps << " steps, dt = "
              << dt << " s)\n";

    if (gp) {
        std::fprintf(gp, "pause -1 'press Enter to close...'\n");
        std::fflush(gp);
        pclose(gp);
    }
    return 0;
}
