// ============================================================================
//  tools/twin_sim_a.cpp
//
//  Window 1 (qt 0): lgn | pinocchio stick figures side by side
//  Window 2 (qt 1): per-link energy  E_i(t) as animated line plot
//                   — watch the wave travel base→tip→base
//
//  BUILD
//      g++ -O2 -std=c++17 tools/twin_sim_a.cpp \
//          -I /path/to/lgn/include \
//          $(pkg-config --cflags --libs pinocchio) \
//          -o twin_sim_a
//
//  RUN
//      ./twin_sim_a
//      ./twin_sim_a --n 7 --duration 15
//      ./twin_sim_a --no-plot        # CSV: t,link,KE,PE,E
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
#include <vector>

#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/dynamics.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/aba.hpp>

// ── URDF ─────────────────────────────────────────────────────────────────────
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
    for (int i = 0; i < n-1; ++i)
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

// ── Stick figures ─────────────────────────────────────────────────────────────
struct StickFigure { std::vector<double> xs, ys; };

static StickFigure poses_lgn(lgn::KinematicTree& tree, const Eigen::VectorXd& q) {
    tree.fk(q);
    StickFigure f;
    f.xs.push_back(0.0); f.ys.push_back(0.0);
    for (int ji = 0; ji < tree.n_joints(); ++ji) {
        const auto& jt = tree.joint(ji);
        if (jt.type == lgn::JointType::Fixed) continue;
        lgn::Vec3 p = lgn::p_of(tree.link(jt.child_link).T_world);
        f.xs.push_back(p.x()); f.ys.push_back(p.y());
    }
    lgn::Vec3 ptip = lgn::p_of(tree.link(tree.link_index("tip")).T_world);
    f.xs.push_back(ptip.x()); f.ys.push_back(ptip.y());
    return f;
}

static StickFigure poses_pin_from_data(pinocchio::Model& model,
                                        pinocchio::Data& data) {
    StickFigure f;
    f.xs.push_back(0.0); f.ys.push_back(0.0);
    for (pinocchio::JointIndex j = 1;
         j < (pinocchio::JointIndex)model.njoints; ++j) {
        const auto& t = data.oMi[j].translation();
        f.xs.push_back(t[0]); f.ys.push_back(t[1]);
    }
    pinocchio::FrameIndex tid = model.getFrameId("tip");
    const auto& tt = data.oMf[tid].translation();
    f.xs.push_back(tt[0]); f.ys.push_back(tt[1]);
    return f;
}

// ── Per-link energy ───────────────────────────────────────────────────────────
struct LinkEnergy { double KE, PE, E; };

static std::vector<LinkEnergy> link_energies(
        pinocchio::Data& data,
        const std::vector<Eigen::Vector3d>& com_prev,
        std::vector<Eigen::Vector3d>& com_curr,
        double m, double L, double dt, int n)
{
    const double g = 9.81;
    const Eigen::Vector3d lc(0.0, L/2.0, 0.0);
    std::vector<LinkEnergy> out(n);
    com_curr.resize(n);
    for (int i = 0; i < n; ++i) {
        pinocchio::JointIndex j = (pinocchio::JointIndex)(i+1);
        com_curr[i] = data.oMi[j].act(lc);
        Eigen::Vector3d v = (com_curr[i] - com_prev[i]) / dt;
        double KE = 0.5 * m * v.squaredNorm();
        double PE = m * g * com_curr[i][1];
        out[i] = { KE, PE, KE+PE };
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int    n        = 7;
    double duration = 10.0;
    double dt       = 1e-3;
    bool   plot     = true;
    double q0_val   = M_PI / 4.0;
    double link_m   = 0.1;
    double link_L   = 0.5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--n"        && i+1 < argc) n        = std::atoi(argv[++i]);
        else if (a == "--duration" && i+1 < argc) duration = std::atof(argv[++i]);
        else if (a == "--dt"       && i+1 < argc) dt       = std::atof(argv[++i]);
        else if (a == "--q0"       && i+1 < argc) q0_val   = std::atof(argv[++i]);
        else if (a == "--no-plot")                plot     = false;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--n N] [--duration S] [--dt S] [--q0 RAD] [--no-plot]\n";
            return 1;
        }
    }

    // ── Build engines ─────────────────────────────────────────────────────
    const std::string urdf = make_chain_urdf(n, link_L);

    auto tree = lgn::URDFLoader::from_string(urdf);
    int  ndof = tree.n_dof();
    if (ndof == 0) { std::cerr << "no DOFs\n"; return 1; }

    pinocchio::Model model;
    pinocchio::urdf::buildModelFromXML(urdf, model);
    pinocchio::Data data(model);
    model.gravity.linear() = Eigen::Vector3d(0.0, -9.81, 0.0);

    if (model.nv != ndof) {
        std::cerr << "DOF mismatch lgn=" << ndof << " pin=" << model.nv << "\n";
        return 1;
    }

    // ── Initial conditions ────────────────────────────────────────────────
    Eigen::VectorXd q_lgn  = Eigen::VectorXd::Constant(ndof, q0_val);
    Eigen::VectorXd dq_lgn = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd q_pin  = q_lgn;
    Eigen::VectorXd dq_pin = dq_lgn;
    Eigen::VectorXd tau    = Eigen::VectorXd::Zero(ndof);

    // ── Energy bootstrap ──────────────────────────────────────────────────
    // Run one FK at t=0 to get initial CoM positions.
    // IMPORTANT: we do NOT call aba here; ddq is not needed yet.
    pinocchio::forwardKinematics(model, data, q_pin);
    pinocchio::updateFramePlacements(model, data);
    const Eigen::Vector3d lc0(0.0, link_L/2.0, 0.0);
    std::vector<Eigen::Vector3d> com_prev(n), com_curr(n);
    for (int i = 0; i < n; ++i)
        com_prev[i] = data.oMi[(pinocchio::JointIndex)(i+1)].act(lc0);
    std::vector<double> E_prev(n);
    { const double g=9.81;
      for (int i=0;i<n;++i) E_prev[i] = link_m*g*com_prev[i][1]; }

    // ── Open two gnuplot windows ──────────────────────────────────────────
    //  gp0: stick figures   (qt window 0)
    //  gp1: energy wave     (qt window 1)
    FILE* gp0 = nullptr;
    FILE* gp1 = nullptr;
    if (plot) {
        gp0 = popen("gnuplot", "w");
        gp1 = popen("gnuplot", "w");
        if (!gp0 || !gp1) {
            std::cerr << "could not open gnuplot; use --no-plot\n"; return 1;
        }
        const double reach = (n+1)*link_L;

        // Window 0: stick figures
        std::fprintf(gp0,
            "set term qt 0 size 900,500 title 'twin_sim — dynamics'\n"
            "set grid\nunset key\n");
        std::fflush(gp0);

        // Window 1: energy wave
        // X = link index (0=base … n-1=tip)
        // Y = total mechanical energy E_i = KE_i + PE_i
        // One curve per frame, redrawn each time.
        // We also draw KE and PE separately so you see the exchange.
        std::fprintf(gp1,
            "set term qt 1 size 800,500 title 'twin_sim — energy wave'\n"
            "set grid\n"
            "set xrange [-0.5:%f]\n"
            "set xlabel 'link index  (0=base, %d=tip)'\n"
            "set ylabel 'energy  [J]'\n"
            "set key top right\n",
            (double)n-0.5, n-1);
        std::fflush(gp1);
    } else {
        std::cout << "t,link,KE,PE,E\n";
    }

    // ── Sim loop ──────────────────────────────────────────────────────────
    using clk = std::chrono::steady_clock;
    auto wall_t0    = clk::now();
    int  steps      = (int)std::ceil(duration / dt);
    int  draw_every = std::max(1, (int)std::round(1.0 / (60.0 * dt)));

    // E_scale for y-axis: auto from first 100 draw frames, then frozen
    double E_max       = 0.0;
    int    E_scale_cnt = 0;
    bool   E_scale_fixed = false;

    for (int s = 0; s < steps; ++s) {
        // ── lgn step ──────────────────────────────────────────────────────
        Eigen::VectorXd ddq_lgn = lgn::forward_dynamics(tree, q_lgn, dq_lgn, tau);
        dq_lgn += ddq_lgn * dt;
        q_lgn  += dq_lgn  * dt;

        // ── pinocchio step ────────────────────────────────────────────────
        // Order matters: aba reads (q_pin, dq_pin) at current state,
        // writes ddq into data.ddq. We integrate AFTER.
        pinocchio::aba(model, data, q_pin, dq_pin, tau);
        // Save ddq before updateFramePlacements touches data internals
        Eigen::VectorXd ddq_pin = data.ddq;
        dq_pin += ddq_pin * dt;
        q_pin  += dq_pin  * dt;

        // Update frames once — reused by energy calc and stick figure
        pinocchio::forwardKinematics(model, data, q_pin);
        pinocchio::updateFramePlacements(model, data);

        // ── Energy ────────────────────────────────────────────────────────
        auto en = link_energies(data, com_prev, com_curr,
                                link_m, link_L, dt, n);
        com_prev = com_curr;

        if (s % draw_every != 0) continue;

        double t   = s * dt;
        double res = (q_lgn - q_pin).cwiseAbs().maxCoeff();

        // Auto-scale energy y-axis
        if (!E_scale_fixed) {
            for (int i = 0; i < n; ++i)
                E_max = std::max(E_max, std::abs(en[i].E));
            if (++E_scale_cnt >= 100 && E_max > 0.0)
                E_scale_fixed = true;
        }
        double ymax = E_max > 0.0 ? E_max * 1.3 : 0.1;

        if (plot) {
            const double reach = (n+1)*link_L;

            // ── Window 0: two stick figures ───────────────────────────────
            // Left panel: lgn. Right panel: pinocchio, shifted +2*reach.
            std::fprintf(gp0,
                "set xrange [%f:%f]\n"
                "set yrange [%f:%f]\n"
                "set size ratio -1\n"
                "set title sprintf('t = %%.2f s   ||Δq||_inf = %%.2e', %f, %g)\n"
                "plot '-' w linespoints lt 1 lw 2 pt 7 ps 1.4 lc rgb '#E69F4D' t 'lgn', "
                     "'-' w linespoints lt 1 lw 2 pt 7 ps 1.4 lc rgb '#5D8A66' t 'pinocchio'\n",
                -reach, 4.0*reach,
                -reach*1.1, reach*0.3,
                t, res);
            // lgn
            auto fl = poses_lgn(tree, q_lgn);
            for (size_t i = 0; i < fl.xs.size(); ++i)
                std::fprintf(gp0, "%f %f\n", fl.xs[i], fl.ys[i]);
            std::fprintf(gp0, "e\n");
            // pinocchio — shifted right so figures don't overlap
            auto fp = poses_pin_from_data(model, data);
            for (size_t i = 0; i < fp.xs.size(); ++i)
                std::fprintf(gp0, "%f %f\n", fp.xs[i] + 2.5*reach, fp.ys[i]);
            std::fprintf(gp0, "e\n");
            std::fflush(gp0);

            // ── Window 1: energy wave ─────────────────────────────────────
            // Three lines: KE, PE, E=KE+PE across link index
            std::fprintf(gp1,
                "set yrange [%f:%f]\n"
                "set title sprintf('Energy per link   t = %%.2f s', %f)\n"
                "plot '-' w linespoints lt 1 lw 2 pt 7 ps 1.2 lc rgb '#D6604D' t 'KE', "
                     "'-' w linespoints lt 1 lw 2 pt 7 ps 1.2 lc rgb '#2166AC' t 'PE', "
                     "'-' w linespoints lt 1 lw 2 pt 7 ps 1.2 lc rgb '#444444' t 'E=KE+PE'\n",
                -0.05*ymax, ymax, t);
            // KE
            for (int i = 0; i < n; ++i)
                std::fprintf(gp1, "%d %f\n", i, en[i].KE);
            std::fprintf(gp1, "e\n");
            // PE  (shifted down by minimum so base of chain = 0)
            double pe_min = en[0].PE;
            for (int i = 1; i < n; ++i) pe_min = std::min(pe_min, en[i].PE);
            for (int i = 0; i < n; ++i)
                std::fprintf(gp1, "%d %f\n", i, en[i].PE - pe_min);
            std::fprintf(gp1, "e\n");
            // E total (same shift)
            double e_min = en[0].E;
            for (int i = 1; i < n; ++i) e_min = std::min(e_min, en[i].E);
            for (int i = 0; i < n; ++i)
                std::fprintf(gp1, "%d %f\n", i, en[i].E - e_min);
            std::fprintf(gp1, "e\n");
            std::fflush(gp1);

        } else {
            for (int i = 0; i < n; ++i)
                std::cout << t << "," << i << ","
                          << en[i].KE << "," << en[i].PE << "," << en[i].E << "\n";
        }

        // Real-time pacing
        if (plot) {
            auto target = wall_t0 + std::chrono::duration_cast<clk::duration>(
                std::chrono::duration<double>(t));
            std::this_thread::sleep_until(target);
        }
    }

    double res_final = (q_lgn - q_pin).cwiseAbs().maxCoeff();
    std::cerr << "\nfinal residual ||q_lgn - q_pin||_inf = " << res_final << "\n";
    std::cerr << "(at " << duration << "s, " << steps
              << " steps, dt=" << dt << "s)\n";

    if (gp0) {
        std::fprintf(gp0, "pause -1 'press Enter to close...'\n"); std::fflush(gp0);
        pclose(gp0);
    }
    if (gp1) {
        std::fprintf(gp1, "pause -1 'press Enter to close...'\n"); std::fflush(gp1);
        pclose(gp1);
    }
    return 0;
}
