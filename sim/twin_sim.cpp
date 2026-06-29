// ============================================================================
//  tools/twin_sim.cpp
//
//  Single gnuplot window with multiplot layout:
//    ┌──────────────────┬──────────────────┐
//    │                  │  Energy (KE/PE)  │
//    │                  ├──────────────────┤
//    │  Dynamics        │  |ΔE| per link   │   (log y)
//    │  (lgn|pinocchio) ├──────────────────┤
//    │                  │  Lyapunov ||δq|| │   (log y, chaos only)
//    └──────────────────┴──────────────────┘
// ============================================================================
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
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
    for (int i = 0; i < n-1; ++i) {
        const char* org = (i == 0) ? "0 0 0" : nullptr;
        std::string origin_xyz = org ? std::string(org)
                                     : std::string("0 ") + std::to_string(L) + " 0";
        ss << "<joint name=\"J" << i << "\" type=\"revolute\">"
           << "<parent link=\"l" << i << "\"/><child link=\"l" << (i+1) << "\"/>"
           << "<origin xyz=\"" << origin_xyz << "\"/><axis xyz=\"0 0 1\"/>"
           << "<limit lower=\"-3.14\" upper=\"3.14\" velocity=\"10\" effort=\"10\"/>"
           << "</joint>\n";
    }
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
        pinocchio::Model& model,
        pinocchio::Data& data,
        const std::vector<Eigen::Vector3d>& com_prev,
        std::vector<Eigen::Vector3d>& com_curr,
        double m, double L, double dt, int n)
{
    const double g = 9.81;
    const Eigen::Vector3d lc(0.0, L/2.0, 0.0);
    std::vector<LinkEnergy> out(n);
    com_curr.resize(n);
    com_curr[0] = lc;
    out[0] = { 0.0, m * g * com_curr[0][1], m * g * com_curr[0][1] };
    for (int i = 1; i < n; ++i) {
        pinocchio::JointIndex j = (pinocchio::JointIndex)i;
        com_curr[i] = data.oMi[j].act(lc);
        Eigen::Vector3d v = (com_curr[i] - com_prev[i]) / dt;
        double KE = 0.5 * m * v.squaredNorm();
        double PE = m * g * com_curr[i][1];
        out[i] = { KE, PE, KE+PE };
    }
    return out;
}

static std::vector<LinkEnergy> link_energies_lgn(
        lgn::KinematicTree& tree,
        const Eigen::VectorXd& q,
        const std::vector<Eigen::Vector3d>& com_prev,
        std::vector<Eigen::Vector3d>& com_curr,
        double m, double L, double dt, int n)
{
    const double g = 9.81;
    tree.fk(q);
    std::vector<LinkEnergy> out(n);
    com_curr.resize(n);
    const lgn::Vec3 lc_local(0.0, L/2.0, 0.0);
    com_curr[0] = Eigen::Vector3d(lc_local.x(), lc_local.y(), lc_local.z());
    out[0] = { 0.0, m * g * com_curr[0][1], m * g * com_curr[0][1] };
    for (int i = 1; i < n; ++i) {
        std::string name = "l" + std::to_string(i);
        const auto& T = tree.link(tree.link_index(name)).T_world;
        lgn::Vec3 p = lgn::p_of(T) + lgn::R_of(T) * lc_local;
        com_curr[i] = Eigen::Vector3d(p.x(), p.y(), p.z());
        Eigen::Vector3d v = (com_curr[i] - com_prev[i]) / dt;
        double KE = 0.5 * m * v.squaredNorm();
        double PE = m * g * com_curr[i][1];
        out[i] = { KE, PE, KE+PE };
    }
    return out;
}

// ── Scenario file format & parser ────────────────────────────────────────────
struct Scenario {
    std::string name;
    std::string description;
    int    n        = 7;
    double duration = 10.0;
    double dt       = 1e-3;
    double eps      = 1e-8;
    double damping  = 0.0;
    bool   chaos    = false;
    std::vector<std::pair<int,double>> q_init;
    std::vector<std::pair<int,double>> dq_init;
};

static double eval_value(std::string s) {
    auto trim = [](std::string& x) {
        size_t a = x.find_first_not_of(" \t\r\n");
        size_t b = x.find_last_not_of(" \t\r\n");
        x = (a == std::string::npos) ? "" : x.substr(a, b-a+1);
    };
    trim(s);
    if (s.empty()) return std::nan("");
    bool neg = false;
    if (s[0] == '-') { neg = true; s = s.substr(1); trim(s); }
    else if (s[0] == '+') { s = s.substr(1); trim(s); }
    auto resolve_atom = [](const std::string& a) -> double {
        std::string t = a;
        std::string tl = t;
        for (auto& c : tl) c = (char)std::tolower((unsigned char)c);
        if (tl == "pi") return M_PI;
        try { return std::stod(t); } catch (...) { return std::nan(""); }
    };
    size_t op_pos = std::string::npos;
    char   op_ch  = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '*' || s[i] == '/') { op_pos = i; op_ch = s[i]; break; }
    }
    double val;
    if (op_pos == std::string::npos) {
        val = resolve_atom(s);
    } else {
        std::string lhs = s.substr(0, op_pos);
        std::string rhs = s.substr(op_pos + 1);
        trim(lhs); trim(rhs);
        double a = resolve_atom(lhs);
        double b = resolve_atom(rhs);
        if (std::isnan(a) || std::isnan(b)) return std::nan("");
        val = (op_ch == '*') ? (a * b) : (a / b);
    }
    return neg ? -val : val;
}

static std::vector<Scenario> load_scenarios(const std::string& path_hint) {
    std::vector<std::string> candidates;
    if (!path_hint.empty()) candidates.push_back(path_hint);
    candidates.push_back("scenarios.md");
    candidates.push_back("../scenarios.md");
    candidates.push_back("../../scenarios.md");
    candidates.push_back("../sim/scenarios.md");
    candidates.push_back("../../sim/scenarios.md");
    if (const char* home = std::getenv("HOME")) {
        candidates.push_back(std::string(home) + "/lgn_hand_ik/sim/scenarios.md");
    }
    std::ifstream in;
    std::string used;
    for (const auto& c : candidates) {
        in.open(c);
        if (in.is_open()) { used = c; break; }
        in.clear();
    }
    if (!in.is_open()) {
        std::cerr << "warning: scenarios.md not found anywhere; "
                     "tried:";
        for (const auto& c : candidates) std::cerr << "\n  " << c;
        std::cerr << "\nfalling back to built-in default scenario\n";
        Scenario def;
        def.name = "Built-in default (horizontal release)";
        def.q_init.push_back({0, M_PI / 2.0});
        return { def };
    }
    in.seekg(0, std::ios::end);
    auto sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::cerr << "[scenarios] reading from " << used
              << " (" << sz << " bytes)\n";
    std::vector<Scenario> out;
    Scenario cur;
    bool have_cur = false;
    bool in_code  = false;
    int  heading_count = 0;
    int  code_block_count = 0;
    std::string line;
    auto commit = [&]() {
        bool has_content = !cur.q_init.empty()
                        || !cur.dq_init.empty()
                        || cur.chaos
                        || cur.damping != 0.0
                        || cur.n != 7
                        || cur.duration != 10.0
                        || cur.dt != 1e-3;
        if (have_cur && has_content) { out.push_back(cur); }
        cur = Scenario();
        have_cur = false;
    };
    auto is_fence = [](const std::string& l) {
        size_t i = 0;
        while (i < l.size() && (l[i] == ' ' || l[i] == '\t')) ++i;
        return i + 3 <= l.size() && l[i] == '`' && l[i+1] == '`' && l[i+2] == '`';
    };
    auto strip_bom = [](std::string& l) {
        if (l.size() >= 3
            && (unsigned char)l[0] == 0xEF
            && (unsigned char)l[1] == 0xBB
            && (unsigned char)l[2] == 0xBF) {
            l.erase(0, 3);
        }
    };
    bool first_line = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (first_line) { strip_bom(line); first_line = false; }
        if (is_fence(line)) {
            in_code = !in_code;
            if (in_code) ++code_block_count;
            continue;
        }
        if (!in_code) {
            if (!line.empty() && line[0] == '#') {
                commit();
                ++heading_count;
                std::string h = line;
                size_t i = 0;
                while (i < h.size() && h[i] == '#') ++i;
                while (i < h.size() && (h[i] == ' ' || h[i] == '\t')) ++i;
                size_t j = i;
                while (j < h.size() && std::isdigit((unsigned char)h[j])) ++j;
                if (j > i && j < h.size() && (h[j] == '.' || h[j] == ')')) {
                    ++j;
                    while (j < h.size() && (h[j] == ' ' || h[j] == '\t')) ++j;
                    i = j;
                }
                cur.name = h.substr(i);
                have_cur = true;
            } else if (have_cur && !line.empty()) {
                std::string t = line;
                size_t a = t.find_first_not_of(" \t");
                if (a != std::string::npos) t = t.substr(a);
                if (t == "---" || t == "***" || t == "___") continue;
                if (!cur.description.empty()) cur.description += ' ';
                cur.description += t;
            }
            continue;
        }
        if (!have_cur) continue;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        auto strip = [](std::string& x) {
            size_t a = x.find_first_not_of(" \t\r\n");
            size_t b = x.find_last_not_of(" \t\r\n");
            x = (a == std::string::npos) ? "" : x.substr(a, b-a+1);
        };
        strip(key); strip(val);
        if (key.empty() || val.empty()) continue;
        auto parse_indexed = [&](const std::string& prefix,
                                 std::vector<std::pair<int,double>>& dst) -> bool {
            if (key.size() <= prefix.size() + 2) return false;
            if (key.substr(0, prefix.size()) != prefix) return false;
            if (key[prefix.size()] != '[' || key.back() != ']') return false;
            std::string idx = key.substr(prefix.size()+1,
                                         key.size() - prefix.size() - 2);
            try {
                int i = std::stoi(idx);
                double v = eval_value(val);
                if (std::isnan(v)) {
                    std::cerr << "[scenarios] could not parse value: "
                              << key << " = " << val << "\n";
                    return true;
                }
                dst.push_back({i, v});
                return true;
            } catch (...) { return false; }
        };
        if (parse_indexed("q",  cur.q_init))  continue;
        if (parse_indexed("dq", cur.dq_init)) continue;
        if (key == "n")        { cur.n = (int)eval_value(val); continue; }
        if (key == "duration") { cur.duration = eval_value(val); continue; }
        if (key == "dt")       { cur.dt = eval_value(val); continue; }
        if (key == "eps")      { cur.eps = eval_value(val); continue; }
        if (key == "damping")  { cur.damping = eval_value(val); continue; }
        if (key == "chaos")    {
            std::string vl = val;
            for (auto& c : vl) c = (char)std::tolower((unsigned char)c);
            cur.chaos = (vl == "true" || vl == "1" || vl == "yes");
            continue;
        }
        std::cerr << "[scenarios] unknown key in '" << cur.name
                  << "': " << key << "\n";
    }
    commit();
    std::cerr << "[scenarios] parsed " << heading_count << " heading(s), "
              << code_block_count << " code block(s), "
              << out.size() << " usable scenario(s)\n";
    if (out.empty()) {
        std::cerr << "warning: scenarios.md had no usable scenarios; using default\n";
        Scenario def;
        def.name = "Built-in default (horizontal release)";
        def.q_init.push_back({0, M_PI / 2.0});
        out.push_back(def);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int    scenario_idx = 1;
    bool   list_only    = false;
    std::string scen_path;
    int    cli_n        = -1;
    double cli_duration = -1.0;
    double cli_dt       = -1.0;
    double cli_eps      = -1.0;
    double cli_damping  = -1.0;
    int    cli_chaos    = -1;
    bool   plot         = true;
    double link_m       = 0.1;
    double link_L       = 0.5;
    bool   diff_mode    = false;
    double diff_tau_lgn = 0.0;
    double diff_tau_pin = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--scenario" && i+1 < argc) scenario_idx = std::atoi(argv[++i]);
        else if (a == "--scenarios" && i+1 < argc) scen_path    = argv[++i];
        else if (a == "--list")                    list_only    = true;
        else if (a == "--n"        && i+1 < argc) cli_n        = std::atoi(argv[++i]);
        else if (a == "--duration" && i+1 < argc) cli_duration = std::atof(argv[++i]);
        else if (a == "--dt"       && i+1 < argc) cli_dt       = std::atof(argv[++i]);
        else if (a == "--eps"      && i+1 < argc) cli_eps      = std::atof(argv[++i]);
        else if (a == "--damping"  && i+1 < argc) cli_damping  = std::atof(argv[++i]);
        else if (a == "--chaos")                  cli_chaos    = 1;
        else if (a == "--no-chaos")               cli_chaos    = 0;
        else if (a == "--no-plot")                plot         = false;
        else if (a == "--diff" && i+2 < argc) {
            diff_mode    = true;
            diff_tau_lgn = std::atof(argv[++i]);
            diff_tau_pin = std::atof(argv[++i]);
        }
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--scenario K] [--scenarios PATH] [--list]"
                         " [--n N] [--duration S] [--dt S] [--eps E]"
                         " [--damping B] [--chaos|--no-chaos] [--no-plot]"
                         " [--diff TAU_LGN TAU_PIN]\n";
            return 1;
        }
    }

    auto scenarios = load_scenarios(scen_path);
    if (list_only) {
        std::cout << "Available scenarios (use --scenario K):\n";
        for (size_t k = 0; k < scenarios.size(); ++k) {
            std::cout << "  " << (k+1) << ".  " << scenarios[k].name << "\n";
            if (!scenarios[k].description.empty())
                std::cout << "        " << scenarios[k].description << "\n";
        }
        return 0;
    }
    if (scenario_idx < 1 || scenario_idx > (int)scenarios.size()) {
        std::cerr << "scenario " << scenario_idx << " out of range (1.."
                  << scenarios.size() << "). Use --list to see all.\n";
        return 1;
    }

    Scenario sc = scenarios[scenario_idx - 1];
    if (cli_n        > 0  ) sc.n        = cli_n;
    if (cli_duration > 0.0) sc.duration = cli_duration;
    if (cli_dt       > 0.0) sc.dt       = cli_dt;
    if (cli_eps      > 0.0) sc.eps      = cli_eps;
    if (cli_damping  >= 0.0) sc.damping = cli_damping;
    if (cli_chaos    >= 0 ) sc.chaos    = (cli_chaos == 1);

    int    n        = sc.n;
    double duration = sc.duration;
    double dt       = sc.dt;
    double eps      = sc.eps;
    double damping  = sc.damping;
    bool   chaos    = sc.chaos;

    std::cerr << "[scenario] #" << scenario_idx << " — " << sc.name << "\n";
    if (!sc.description.empty())
        std::cerr << "           " << sc.description << "\n";
    std::cerr << "           n=" << n << "  duration=" << duration
              << "s  dt=" << dt << "s  chaos=" << (chaos?"on":"off")
              << "  damping=" << damping << "\n";
    if (diff_mode) {
        std::cerr << "[diff] joint-0 torque  lgn=" << diff_tau_lgn
                  << " N·m   pin=" << diff_tau_pin << " N·m"
                  << "  (constant, applied every step)\n";
    }

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

    Eigen::VectorXd q_lgn  = Eigen::VectorXd::Zero(ndof);
    Eigen::VectorXd dq_lgn = Eigen::VectorXd::Zero(ndof);
    for (auto& [idx, val] : sc.q_init) {
        if (idx < 0 || idx >= ndof) {
            std::cerr << "[scenario] q[" << idx << "] ignored (ndof=" << ndof << ")\n";
            continue;
        }
        q_lgn[idx] = val;
    }
    for (auto& [idx, val] : sc.dq_init) {
        if (idx < 0 || idx >= ndof) {
            std::cerr << "[scenario] dq[" << idx << "] ignored (ndof=" << ndof << ")\n";
            continue;
        }
        dq_lgn[idx] = val;
    }

    Eigen::VectorXd q_pin  = q_lgn;
    Eigen::VectorXd dq_pin = dq_lgn;

    pinocchio::Data data_cha(model);
    Eigen::VectorXd q_cha  = q_pin;
    Eigen::VectorXd dq_cha = dq_pin;
    if (chaos) {
        q_cha[0] += eps;
        std::cerr << "chaos mode: perturbing joint 0 by eps=" << eps << "\n";
    }

    pinocchio::forwardKinematics(model, data, q_pin);
    pinocchio::updateFramePlacements(model, data);

    const Eigen::Vector3d lc0(0.0, link_L/2.0, 0.0);
    std::vector<Eigen::Vector3d> com_prev(n), com_curr(n);
    com_prev[0] = lc0;
    for (int i = 1; i < n; ++i)
        com_prev[i] = data.oMi[(pinocchio::JointIndex)i].act(lc0);

    std::vector<Eigen::Vector3d> com_prev_lgn(n), com_curr_lgn(n);
    tree.fk(q_lgn);
    com_prev_lgn[0] = lc0;
    for (int i = 1; i < n; ++i) {
        std::string name = "l" + std::to_string(i);
        const auto& T = tree.link(tree.link_index(name)).T_world;
        lgn::Vec3 lc_local(0.0, link_L/2.0, 0.0);
        lgn::Vec3 p = lgn::p_of(T) + lgn::R_of(T) * lc_local;
        com_prev_lgn[i] = Eigen::Vector3d(p.x(), p.y(), p.z());
    }

    // ── Dynamics-pane framing (in *data* coords, not panel pixels) ────────
    const double reach = n * link_L;
    const double pane_w = 2.5 * reach;
    const double pane_h = 3.0 * reach;
    const double anchor_x_lgn = pane_w * 0.80;
    const double anchor_x_pin = pane_w * 0.80 + pane_w;
    const double y_top        = pane_h * (1.0/3.0);
    const double y_bot        = -pane_h * (2.0/3.0);
    const double view_xmin = 0.0;
    const double view_xmax = 2.0 * pane_w;
    const double view_ymin = y_bot;
    const double view_ymax = y_top + 0.15 * pane_h;

    // ── Open ONE gnuplot window with multiplot ────────────────────────────
    FILE* gp = nullptr;
    if (plot) {
        gp = popen("gnuplot", "w");
        if (!gp) { std::cerr << "could not open gnuplot; use --no-plot\n"; return 1; }
        std::fprintf(gp,
            "set term qt 0 size 1500,900 position 40,40 "
                "title 'twin_sim — fused view'\n"
            "set multiplot title 'Waiting to start...' font ',13'\n"
            "unset multiplot\n");
        std::fflush(gp);
        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        std::cerr << "\nWindow ready. Press Enter to start simulation... ";
        std::cin.get();
    } else {
        std::cout << "t,link,KE_pin,PE_pin,E_pin,KE_lgn,PE_lgn,E_lgn";
        if (chaos) std::cout << ",delta_q_norm";
        std::cout << "\n";
    }

    using clk = std::chrono::steady_clock;
    auto wall_t0    = clk::now();
    int  steps      = (int)std::ceil(duration / dt);
    int  draw_every = std::max(1, (int)std::round(1.0 / (60.0 * dt)));

    double E_max = 0.0; int E_cnt = 0; bool E_fixed = false;
    std::vector<double> lyap_t, lyap_d;
    lyap_t.reserve(steps / draw_every + 1);
    lyap_d.reserve(steps / draw_every + 1);

    for (int s = 0; s < steps; ++s) {
        Eigen::VectorXd tau_lgn = -damping * dq_lgn;
        Eigen::VectorXd tau_pin = -damping * dq_pin;
        if (diff_mode && ndof > 0) {
            tau_lgn[0] += diff_tau_lgn;
            tau_pin[0] += diff_tau_pin;
        }

        Eigen::VectorXd ddq_lgn = lgn::forward_dynamics(tree, q_lgn, dq_lgn, tau_lgn);
        dq_lgn += ddq_lgn * dt;
        q_lgn  += dq_lgn  * dt;

        pinocchio::aba(model, data, q_pin, dq_pin, tau_pin);
        Eigen::VectorXd ddq_pin = data.ddq;
        dq_pin += ddq_pin * dt;
        q_pin  += dq_pin  * dt;

        pinocchio::forwardKinematics(model, data, q_pin);
        pinocchio::updateFramePlacements(model, data);

        double delta_norm = 0.0;
        if (chaos) {
            Eigen::VectorXd tau_cha = -damping * dq_cha;
            pinocchio::aba(model, data_cha, q_cha, dq_cha, tau_cha);
            Eigen::VectorXd ddq_cha = data_cha.ddq;
            dq_cha += ddq_cha * dt;
            q_cha  += dq_cha  * dt;
            delta_norm = (q_pin - q_cha).norm();
        }

        auto en = link_energies(model, data, com_prev, com_curr,
                                link_m, link_L, dt, n);
        com_prev = com_curr;

        auto en_lgn = link_energies_lgn(tree, q_lgn, com_prev_lgn, com_curr_lgn,
                                         link_m, link_L, dt, n);
        com_prev_lgn = com_curr_lgn;

        if (s % draw_every != 0) continue;

        double t   = s * dt;
        double res = (q_lgn - q_pin).cwiseAbs().maxCoeff();

        if (!E_fixed) {
            const double h_ref_local = n * link_L;
            const double pe_shift = link_m * 9.81 * h_ref_local;
            for (int i = 0; i < n; ++i) {
                E_max = std::max(E_max, std::abs(en[i].KE));
                E_max = std::max(E_max, std::abs(en_lgn[i].KE));
                E_max = std::max(E_max, std::abs(en[i].PE     + pe_shift));
                E_max = std::max(E_max, std::abs(en_lgn[i].PE + pe_shift));
            }
            if (++E_cnt >= 100 && E_max > 0.0) E_fixed = true;
        }
        double ymax = E_max > 0.0 ? E_max * 1.15 : 0.1;

        if (chaos) {
            lyap_t.push_back(t);
            lyap_d.push_back(delta_norm > 0.0 ? delta_norm : 1e-20);
        }

        if (plot) {

            // 1. PRE-LOAD DATA BLOCKS INTO GNUPLOT MEMORY ─────────────────
            std::fprintf(gp, "$poses_lgn << EOD\n");
            auto fl = poses_lgn(tree, q_lgn);
            for (size_t i = 0; i < fl.xs.size(); ++i)
                std::fprintf(gp, "%f %f\n", fl.xs[i] + anchor_x_lgn, fl.ys[i] + y_top);
            std::fprintf(gp, "EOD\n");

            std::fprintf(gp, "$poses_pin << EOD\n");
            auto fp = poses_pin_from_data(model, data);
            for (size_t i = 0; i < fp.xs.size(); ++i)
                std::fprintf(gp, "%f %f\n", fp.xs[i] + anchor_x_pin, fp.ys[i] + y_top);
            std::fprintf(gp, "EOD\n");

            const double h_ref = n * link_L;
            const double pe_shift_const = link_m * 9.81 * h_ref;

            std::fprintf(gp, "$eke_pin << EOD\n");
            for (int i=0;i<n;++i) std::fprintf(gp, "%d %f\n", i, en[i].KE);
            std::fprintf(gp, "EOD\n");

            std::fprintf(gp, "$eke_lgn << EOD\n");
            for (int i=0;i<n;++i) std::fprintf(gp, "%d %f\n", i, en_lgn[i].KE);
            std::fprintf(gp, "EOD\n");

            std::fprintf(gp, "$epe_pin << EOD\n");
            for (int i=0;i<n;++i) std::fprintf(gp, "%d %f\n", i, en[i].PE + pe_shift_const);
            std::fprintf(gp, "EOD\n");

            std::fprintf(gp, "$epe_lgn << EOD\n");
            for (int i=0;i<n;++i) std::fprintf(gp, "%d %f\n", i, en_lgn[i].PE + pe_shift_const);
            std::fprintf(gp, "EOD\n");

            // NEW: |Δenergy| per link, floored at 1e-16 for log scale
            std::fprintf(gp, "$ediff << EOD\n");
            for (int i=0;i<n;++i) {
                double dKE = std::abs(en[i].KE - en_lgn[i].KE);
                double dPE = std::abs(en[i].PE - en_lgn[i].PE);
                if (dKE < 1e-16) dKE = 1e-16;
                if (dPE < 1e-16) dPE = 1e-16;
                std::fprintf(gp, "%d %e %e\n", i, dKE, dPE);
            }
            std::fprintf(gp, "EOD\n");

            if (chaos && lyap_t.size() >= 2) {
                std::fprintf(gp, "$lyap << EOD\n");
                for (size_t k = 0; k < lyap_t.size(); ++k)
                    std::fprintf(gp, "%f %e\n", lyap_t[k], lyap_d[k]);
                std::fprintf(gp, "EOD\n");
            }

            // 2. EXECUTE DRAWING COMMANDS ─────────────────────────────────
            std::fprintf(gp,
                "set multiplot title sprintf('twin_sim   t = %%.2f s   "
                    "||Δq||_inf = %%.2e   b = %%.4f', %f, %g, %g) "
                    "font ',12'\n",
                t, res, damping);

            // Left column = dynamics (full height).
            // Right column = 3 stacked panels: energy, |ΔE|, Lyapunov.
            // If chaos is off, the Lyapunov panel is skipped and the
            // remaining two right panels expand to fill the right column.

            const double LEFT_W   = 0.50;
            const double RIGHT_W  = 1.0 - LEFT_W;
            const double RIGHT_X  = LEFT_W;
            const int n_right     = chaos ? 3 : 2;
            const double TOP_PAD  = 0.04;   // leave room for multiplot title
            const double BOT_PAD  = 0.02;
            const double RIGHT_H_AVAIL = 1.0 - TOP_PAD - BOT_PAD;
            const double RIGHT_PANEL_H = RIGHT_H_AVAIL / n_right;

            // ── LEFT: dynamics ────────────────────────────────────────
            std::fprintf(gp,
                "set origin 0.0, %f\n"
                "set size   %f, %f\n"
                "set lmargin 6\nset rmargin 2\nset tmargin 1\nset bmargin 3\n"
                "unset logscale\n"
                "set grid\n"
                "unset key\n"
                "set xlabel ''\n"
                "set ylabel ''\n"
                "set xrange [%f:%f]\n"
                "set yrange [%f:%f]\n"
                "set size ratio -1\n"
                "set label 1 'lgn'       at %f, %f center font ',12' front\n"
                "set label 2 'pinocchio' at %f, %f center font ',12' front\n"
                "set title 'dynamics'\n"
                "plot $poses_lgn w linespoints lt 1 lw 2 pt 7 ps 1.3 lc rgb '#E67E22' t 'lgn', "
                     "$poses_pin w linespoints lt 1 lw 2 pt 7 ps 1.3 lc rgb '#1F5FA8' t 'pinocchio'\n",
                BOT_PAD,
                LEFT_W, 1.0 - TOP_PAD - BOT_PAD,
                view_xmin, view_xmax,
                view_ymin, view_ymax,
                anchor_x_lgn, y_top + pane_h*0.08,
                anchor_x_pin, y_top + pane_h*0.08);

            std::fprintf(gp,
                "unset label 1\n"
                "unset label 2\n"
                "set size noratio\n");

            // ── RIGHT TOP: KE/PE per link ─────────────────────────────
            double y_top_panel = BOT_PAD + (n_right - 1) * RIGHT_PANEL_H;
            std::fprintf(gp,
                "set origin %f, %f\n"
                "set size   %f, %f\n"
                "set lmargin 10\nset rmargin 3\nset tmargin 2\nset bmargin 3\n"
                "unset logscale\n"
                "set grid\n"
                "set key top right\n"
                "set xrange [-0.5:%f]\n"
                "set yrange [%f:%f]\n"
                "set xlabel ''\n"
                "set ylabel 'energy  [J]'\n"
                "set title 'KE / PE per link'\n"
                "plot $eke_pin w linespoints lt 1 lw 2.5 pt 7 ps 1.2 lc rgb '#1F5FA8' t 'KE pin', "
                     "$eke_lgn w linespoints lt 1 lw 2.5 pt 6 ps 1.3 lc rgb '#E67E22' t 'KE lgn', "
                     "$epe_pin w linespoints lt 1 lw 1.5 pt 7 ps 0.8 lc rgb '#9DB8D8' t 'PE pin', "
                     "$epe_lgn w linespoints lt 1 lw 1.5 pt 6 ps 1.0 lc rgb '#F5C99B' t 'PE lgn'\n",
                RIGHT_X, y_top_panel,
                RIGHT_W, RIGHT_PANEL_H,
                (double)n - 0.5,
                -0.05*ymax, ymax);

            // ── RIGHT MID: |ΔE| per link (log y) ──────────────────────
            double y_mid_panel = BOT_PAD + (n_right - 2) * RIGHT_PANEL_H;
            const char* mid_xlabel = chaos ? "" : "link index";
            std::fprintf(gp,
                "set origin %f, %f\n"
                "set size   %f, %f\n"
                "set lmargin 10\nset rmargin 3\nset tmargin 2\nset bmargin 3\n"
                "set grid\n"
                "set logscale y\n"
                "set format y '10^{%%L}'\n"
                "set key top right\n"
                "set xrange [-0.5:%f]\n"
                "set yrange [1e-16:*]\n"
                "set xlabel '%s'\n"
                "set ylabel '|ΔE|  [J]  (log)'\n"
                "set title '|KE_{pin}-KE_{lgn}|  and  |PE_{pin}-PE_{lgn}|  per link'\n"
                "plot $ediff using 1:2 w linespoints lt 1 lw 2 pt 7 ps 1.0 lc rgb '#B2182B' t '|ΔKE|', "
                     "$ediff using 1:3 w linespoints lt 1 lw 1.5 pt 6 ps 1.0 lc rgb '#762A83' t '|ΔPE|'\n",
                RIGHT_X, y_mid_panel,
                RIGHT_W, RIGHT_PANEL_H,
                (double)n - 0.5,
                mid_xlabel);

            // ── RIGHT BOTTOM: Lyapunov (only when chaos is on) ────────
            if (chaos && lyap_t.size() >= 2) {
                double y_bot_panel = BOT_PAD;
                std::fprintf(gp,
                    "set origin %f, %f\n"
                    "set size   %f, %f\n"
                    "set lmargin 10\nset rmargin 3\nset tmargin 2\nset bmargin 4\n"
                    "set grid\n"
                    "set logscale y\n"
                    "set format y '10^{%%L}'\n"
                    "set key top left\n"
                    "set xrange [0:%f]\n"
                    "set yrange [*:*]\n"
                    "set xlabel 't  [s]'\n"
                    "set ylabel '||δq(t)||₂  (log)'\n"
                    "set title sprintf('Lyapunov  ε=%g  b=%g')\n"
                    "plot $lyap w lines lw 2 lc rgb '#B2182B' t '||δq||₂'\n",
                    RIGHT_X, y_bot_panel,
                    RIGHT_W, RIGHT_PANEL_H,
                    duration, eps, damping);
            }

            // Reset linear y for next frame's dynamics plot
            std::fprintf(gp,
                "unset logscale\n"
                "unset format\n");

            std::fprintf(gp, "unset multiplot\n");
            std::fflush(gp);

        } else {
            for (int i = 0; i < n; ++i) {
                std::cout << t << "," << i << ","
                          << en[i].KE     << "," << en[i].PE     << "," << en[i].E << ","
                          << en_lgn[i].KE << "," << en_lgn[i].PE << "," << en_lgn[i].E;
                if (chaos) std::cout << "," << delta_norm;
                std::cout << "\n";
            }
        }

        if (plot) {
            auto target = wall_t0 + std::chrono::duration_cast<clk::duration>(
                std::chrono::duration<double>(t));
            std::this_thread::sleep_until(target);
        }
    }

    double res_final = (q_lgn - q_pin).cwiseAbs().maxCoeff();
    std::cerr << "\nfinal residual ||q_lgn - q_pin||_inf = " << res_final << "\n";
    if (chaos && !lyap_d.empty()) {
        double d0   = lyap_d.front();
        double dend = lyap_d.back();
        std::cerr << "chaos: ||δq|| grew from " << d0
                  << " to " << dend
                  << "  (ratio " << dend/d0 << ")\n";
    }
    std::cerr << "(at " << duration << "s, " << steps
              << " steps, dt=" << dt << "s, damping=" << damping << ")\n";

    if (gp) {
        std::fprintf(gp, "pause -1 'press Enter to close...'\n");
        std::fflush(gp);
        pclose(gp);
    }
    return 0;
}
