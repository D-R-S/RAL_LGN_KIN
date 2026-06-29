// ============================================================================
//  tools/twin_sim_a.cpp
//
//  Physics-only twin simulator. No plotting.
//
//  Streams one CSV record per frame to stdout. Pipe into the Python
//  renderer for a flicker-free animated view:
//
//      ./twin_sim_a --scenario 3                 | python3 twin_sim_render.py
//      ./twin_sim_a --chaos --duration 20        | python3 twin_sim_render.py
//      ./twin_sim_a --scenario 1 > run.csv       # capture for later replay
//      python3 twin_sim_render.py < run.csv      # replay from file
//
//  ── STREAM FORMAT ──────────────────────────────────────────────────────────
//  First two lines are a header block, prefix '# '. Everything after is data.
//
//    # twin_sim v1
//    # n=<int> dt=<f> duration=<f> damping=<f> chaos=<0|1> link_L=<f>
//    # scenario="<text>"
//    # COLUMNS: kind, ... (defined per kind below)
//
//  Each subsequent line is one record. First field is the record kind:
//
//    FRAME,<t>,<residual_inf>,<delta_norm_or_nan>
//        Marks the start of a new frame. residual = ||q_lgn - q_pin||_inf.
//        delta_norm is ||q_pin - q_cha||_2 when chaos is on, else 'nan'.
//
//    POSE_LGN,<x0>,<y0>,<x1>,<y1>,...        n+1 (x,y) pairs
//    POSE_PIN,<x0>,<y0>,<x1>,<y1>,...        n+1 (x,y) pairs
//        Stick-figure points for the most recent FRAME. Untranslated: the
//        renderer applies any anchor offsets it wants.
//
//    ENERGY,<idx>,<KE_pin>,<PE_pin>,<KE_lgn>,<PE_lgn>
//        One row per link (idx = 0..n-1) for the most recent FRAME.
//        PE is reported raw (no offset shift); renderer can shift if it
//        likes.
//
//  Records belonging to a frame come strictly between two FRAME markers.
//  The renderer accumulates everything since the last FRAME and redraws
//  when it sees the next FRAME marker (or stdin closes).
//
//  Lines starting with '#' are comments and may appear anywhere; the
//  renderer prints them to its own stderr.
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
struct LinkEnergy { double KE, PE; };
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
    out[0] = { 0.0, m * g * com_curr[0][1] };
    for (int i = 1; i < n; ++i) {
        pinocchio::JointIndex j = (pinocchio::JointIndex)i;
        com_curr[i] = data.oMi[j].act(lc);
        Eigen::Vector3d v = (com_curr[i] - com_prev[i]) / dt;
        out[i] = { 0.5 * m * v.squaredNorm(), m * g * com_curr[i][1] };
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
    out[0] = { 0.0, m * g * com_curr[0][1] };
    for (int i = 1; i < n; ++i) {
        std::string name = "l" + std::to_string(i);
        const auto& T = tree.link(tree.link_index(name)).T_world;
        lgn::Vec3 p = lgn::p_of(T) + lgn::R_of(T) * lc_local;
        com_curr[i] = Eigen::Vector3d(p.x(), p.y(), p.z());
        Eigen::Vector3d v = (com_curr[i] - com_prev[i]) / dt;
        out[i] = { 0.5 * m * v.squaredNorm(), m * g * com_curr[i][1] };
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
        std::string tl = a;
        for (auto& c : tl) c = (char)std::tolower((unsigned char)c);
        if (tl == "pi") return M_PI;
        try { return std::stod(a); } catch (...) { return std::nan(""); }
    };
    size_t op_pos = std::string::npos;
    char   op_ch  = 0;
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i] == '*' || s[i] == '/') { op_pos = i; op_ch = s[i]; break; }
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
    if (const char* home = std::getenv("HOME"))
        candidates.push_back(std::string(home) + "/lgn_hand_ik/sim/scenarios.md");
    std::ifstream in;
    std::string used;
    for (const auto& c : candidates) {
        in.open(c);
        if (in.is_open()) { used = c; break; }
        in.clear();
    }
    if (!in.is_open()) {
        std::cerr << "warning: scenarios.md not found anywhere; tried:";
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
    bool have_cur = false, in_code = false, first_line = true;
    int  heading_count = 0, code_block_count = 0;
    std::string line;
    auto commit = [&]() {
        bool has_content = !cur.q_init.empty() || !cur.dq_init.empty()
                        || cur.chaos || cur.damping != 0.0 || cur.n != 7
                        || cur.duration != 10.0 || cur.dt != 1e-3;
        if (have_cur && has_content) out.push_back(cur);
        cur = Scenario();
        have_cur = false;
    };
    auto is_fence = [](const std::string& l) {
        size_t i = 0;
        while (i < l.size() && (l[i] == ' ' || l[i] == '\t')) ++i;
        return i + 3 <= l.size() && l[i] == '`' && l[i+1] == '`' && l[i+2] == '`';
    };
    auto strip_bom = [](std::string& l) {
        if (l.size() >= 3 && (unsigned char)l[0] == 0xEF
            && (unsigned char)l[1] == 0xBB && (unsigned char)l[2] == 0xBF)
            l.erase(0, 3);
    };
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
            std::string idx = key.substr(prefix.size()+1, key.size()-prefix.size()-2);
            try {
                int i = std::stoi(idx);
                double v = eval_value(val);
                if (std::isnan(v)) {
                    std::cerr << "[scenarios] could not parse: "
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
        std::cerr << "warning: no usable scenarios; using default\n";
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
    int    cli_n = -1;
    double cli_duration = -1.0, cli_dt = -1.0, cli_eps = -1.0, cli_damping = -1.0;
    int    cli_chaos = -1;
    bool   realtime  = true;   // pace stdout to wall-clock for live viewing
    double link_m = 0.1, link_L = 0.5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--scenario"  && i+1 < argc) scenario_idx = std::atoi(argv[++i]);
        else if (a == "--scenarios" && i+1 < argc) scen_path    = argv[++i];
        else if (a == "--list")                     list_only   = true;
        else if (a == "--n"         && i+1 < argc) cli_n        = std::atoi(argv[++i]);
        else if (a == "--duration"  && i+1 < argc) cli_duration = std::atof(argv[++i]);
        else if (a == "--dt"        && i+1 < argc) cli_dt       = std::atof(argv[++i]);
        else if (a == "--eps"       && i+1 < argc) cli_eps      = std::atof(argv[++i]);
        else if (a == "--damping"   && i+1 < argc) cli_damping  = std::atof(argv[++i]);
        else if (a == "--chaos")                    cli_chaos   = 1;
        else if (a == "--no-chaos")                 cli_chaos   = 0;
        else if (a == "--fast")                     realtime    = false;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--scenario K] [--scenarios PATH] [--list]"
                         " [--n N] [--duration S] [--dt S] [--eps E]"
                         " [--damping B] [--chaos|--no-chaos] [--fast]\n";
            std::cerr << "  (stream CSV to stdout; pipe to twin_sim_render.py)\n";
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
            std::cerr << "[scenario] q[" << idx << "] ignored\n";
            continue;
        }
        q_lgn[idx] = val;
    }
    for (auto& [idx, val] : sc.dq_init) {
        if (idx < 0 || idx >= ndof) {
            std::cerr << "[scenario] dq[" << idx << "] ignored\n";
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
    // ── Stream header ─────────────────────────────────────────────────────
    std::printf("# twin_sim v1\n");
    std::printf("# n=%d dt=%.9g duration=%.9g damping=%.9g chaos=%d link_L=%.9g link_m=%.9g\n",
                n, dt, duration, damping, chaos?1:0, link_L, link_m);
    std::printf("# scenario=\"%s\"\n", sc.name.c_str());
    std::printf("# COLUMNS: FRAME,t,residual_inf,delta_norm\n");
    std::printf("# COLUMNS: POSE_LGN,x0,y0,x1,y1,... (n+1 pairs)\n");
    std::printf("# COLUMNS: POSE_PIN,x0,y0,x1,y1,... (n+1 pairs)\n");
    std::printf("# COLUMNS: ENERGY,idx,KE_pin,PE_pin,KE_lgn,PE_lgn\n");
    std::fflush(stdout);
    using clk = std::chrono::steady_clock;
    auto wall_t0    = clk::now();
    int  steps      = (int)std::ceil(duration / dt);
    int  draw_every = std::max(1, (int)std::round(1.0 / (60.0 * dt)));
    for (int s = 0; s < steps; ++s) {
        Eigen::VectorXd tau_lgn = -damping * dq_lgn;
        Eigen::VectorXd tau_pin = -damping * dq_pin;
        Eigen::VectorXd ddq_lgn = lgn::forward_dynamics(tree, q_lgn, dq_lgn, tau_lgn);
        dq_lgn += ddq_lgn * dt;
        q_lgn  += dq_lgn  * dt;
        pinocchio::aba(model, data, q_pin, dq_pin, tau_pin);
        Eigen::VectorXd ddq_pin = data.ddq;
        dq_pin += ddq_pin * dt;
        q_pin  += dq_pin  * dt;
        pinocchio::forwardKinematics(model, data, q_pin);
        pinocchio::updateFramePlacements(model, data);
        double delta_norm = std::nan("");
        if (chaos) {
            Eigen::VectorXd tau_cha = -damping * dq_cha;
            pinocchio::aba(model, data_cha, q_cha, dq_cha, tau_cha);
            Eigen::VectorXd ddq_cha = data_cha.ddq;
            dq_cha += ddq_cha * dt;
            q_cha  += dq_cha  * dt;
            delta_norm = (q_pin - q_cha).norm();
        }
        auto en     = link_energies    (model, data, com_prev,     com_curr,
                                         link_m, link_L, dt, n);
        com_prev = com_curr;
        auto en_lgn = link_energies_lgn(tree,        q_lgn,
                                         com_prev_lgn, com_curr_lgn,
                                         link_m, link_L, dt, n);
        com_prev_lgn = com_curr_lgn;
        if (s % draw_every != 0) continue;
        double t   = s * dt;
        double res = (q_lgn - q_pin).cwiseAbs().maxCoeff();
        // ── Emit frame ───────────────────────────────────────────────────
        std::printf("FRAME,%.6f,%.6e,", t, res);
        if (std::isnan(delta_norm)) std::printf("nan\n");
        else                         std::printf("%.6e\n", delta_norm);
        auto fl = poses_lgn(tree, q_lgn);
        auto fp = poses_pin_from_data(model, data);
        std::printf("POSE_LGN");
        for (size_t i = 0; i < fl.xs.size(); ++i)
            std::printf(",%.6f,%.6f", fl.xs[i], fl.ys[i]);
        std::printf("\n");
        std::printf("POSE_PIN");
        for (size_t i = 0; i < fp.xs.size(); ++i)
            std::printf(",%.6f,%.6f", fp.xs[i], fp.ys[i]);
        std::printf("\n");
        for (int i = 0; i < n; ++i) {
            std::printf("ENERGY,%d,%.6e,%.6e,%.6e,%.6e\n",
                        i, en[i].KE, en[i].PE, en_lgn[i].KE, en_lgn[i].PE);
        }
        std::fflush(stdout);
        if (realtime) {
            auto target = wall_t0 + std::chrono::duration_cast<clk::duration>(
                std::chrono::duration<double>(t));
            std::this_thread::sleep_until(target);
        }
    }
    double res_final = (q_lgn - q_pin).cwiseAbs().maxCoeff();
    std::cerr << "\nfinal residual ||q_lgn - q_pin||_inf = " << res_final << "\n";
    std::cerr << "(at " << duration << "s, " << steps
              << " steps, dt=" << dt << "s, damping=" << damping << ")\n";
    return 0;
}
