// ============================================================================
//  tools/twin_sim_a.cpp
//
//  Window 0 (left):  lgn | pinocchio stick figures
//  Window 1 (right): energy wave  KE / PE per link + |ΔE_tot|
//  Window 2 (right, chaos mode): Lyapunov divergence  ||δq(t)||  log-scale
//
//  RUN
//      ./twin_sim_a                          # scenario 1 from scenarios.md
//      ./twin_sim_a --scenario 3             # scenario 3 from scenarios.md
//      ./twin_sim_a --list                   # list available scenarios
//      ./twin_sim_a --chaos                  # enable Lyapunov twin
//      ./twin_sim_a --n 10 --duration 20     # override scenario values
//      ./twin_sim_a --no-plot                # CSV to stdout
//
//  Scenarios live in scenarios.md (in CWD or alongside the binary). Add a
//  new scenario by appending a block to that file — no recompile needed.
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
//  All joints are Z-axis revolute → a clean planar n-link pendulum in XY.
//
//  Topology note: J_0 attaches link 1 at link 0's frame ORIGIN (the anchor),
//  not at link 0's far end. This way link 1 pivots from the same point as
//  the anchor and the swinging chain hangs cleanly from the top — no
//  awkward "link 0 sticking out" segment to the side of the pivot.
//  Link 0 is still there for inertia bookkeeping but visually it collapses
//  to the anchor point.
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
        // J_0 origin sits at link 0's anchor (0,0,0); subsequent joints
        // sit at the end of their parent link as usual (0, L, 0).
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
//
//  Indexing convention (consistent across both engines):
//    i = 0       → world-anchored base "link 0" (fixed, never moves, KE=0)
//    i = 1..n-1  → moving links driven by joints J_{i-1}
//
//  Note: link 0 is fixed because the world→l0 joint is fixed in the URDF,
//  so its KE is exactly 0 and its PE is constant. We still report it so
//  the per-link plot has all n links visible.
//
//  In pinocchio, joint index k corresponds to link l_k (1-indexed; joint 0
//  is the universe). So link l_i has joint index i+0? No: pinocchio's
//  buildModelFromXML maps URDF revolute joint J_i (l_i → l_{i+1}) onto
//  pinocchio joint index (i+1), because pinocchio joint 0 is the universe.
//  The placement data.oMi[k] gives the frame at the start of link l_k,
//  i.e. l_{k} for k=1..n-1 corresponds to URDF link l_k.
//
//  Wait — that's confusing. Let me state it directly with what works:
//  for URDF link "l_i" (0-indexed), its world placement in pinocchio is
//  data.oMi[k] where k is the joint index whose CHILD is l_i. Since
//  joint J_i has child l_{i+1}, the joint index for link l_i is i (when
//  i≥1, via J_{i-1}). For i=0 (l_0) the parent joint is the fixed
//  world→l0 joint which doesn't get a movable index — but l_0 is fixed
//  at origin, so its COM is just (0, L/2, 0).
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
    // Link 0 is the world-anchored fixed base link.
    // Its COM is at the fixed offset lc in world frame (anchor is at origin).
    com_curr[0] = lc;
    out[0] = { 0.0, m * g * com_curr[0][1], m * g * com_curr[0][1] };
    // Links 1..n-1: driven by joints J_0..J_{n-2}, which in pinocchio
    // correspond to joint indices 1..n-1.
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
// Same per-link energy but driven by lgn's FK.
// Same indexing: link 0 is the fixed anchor, links 1..n-1 are the chain.
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
    // Link 0: fixed anchor, COM at lc_local in world (since base joint is fixed at origin)
    com_curr[0] = Eigen::Vector3d(lc_local.x(), lc_local.y(), lc_local.z());
    out[0] = { 0.0, m * g * com_curr[0][1], m * g * com_curr[0][1] };
    // Links 1..n-1: read placement from lgn's FK
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
//
//  scenarios.md format — each scenario is a heading + a fenced code block:
//
//      ## 1. Horizontal release
//      Whole chain horizontal, released from rest. Classic dramatic swing.
//      ```
//      n = 7
//      duration = 10
//      q[0] = pi/2
//      ```
//
//      ## 2. Inverted-ish kick
//      Top half up, bottom half down, small kick on the tip joint.
//      ```
//      n = 7
//      q[0] = -pi/2
//      q[3] = pi
//      dq[5] = 0.5
//      ```
//
//  Rules:
//   - Headings (lines starting with '#') delimit scenarios. The number in
//     the heading is informational — actual scenario index is by position
//     (1st block = #1, 2nd = #2, ...).
//   - Inside a fenced ``` block, simple `key = value` lines. Keys:
//       n, duration, dt, eps, chaos, q[i], dq[i]
//   - Values may be numbers or simple expressions: pi, pi/2, -pi/4, 2*pi,
//     0.5, -1.3, etc.
//   - Joint indices i in q[i]/dq[i] are 0-based and refer to the n-1
//     revolute joints in the chain.
//   - Lines outside fenced blocks (description text) are ignored.
struct Scenario {
    std::string name;          // heading text after the number
    std::string description;   // free text between heading and code block
    int    n        = 7;
    double duration = 10.0;
    double dt       = 1e-3;
    double eps      = 1e-8;
    bool   chaos    = false;
    std::vector<std::pair<int,double>> q_init;   // (joint_idx, value)
    std::vector<std::pair<int,double>> dq_init;  // (joint_idx, value)
};
// Tiny expression evaluator: supports number, pi, pi/N, N*pi, leading
// minus, simple binary * and / on two factors. Enough for "pi/2", "-pi/4",
// "2*pi", "1.57". Returns NaN on parse failure.
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
    // Split on '*' or '/'
    auto resolve_atom = [](const std::string& a) -> double {
        std::string t = a;
        // (lower-case lazy match: "pi" / "PI" / "Pi")
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
// Read a scenarios.md file and return all scenarios in order.
// Looks for the file at: <path> as given, then ./scenarios.md, then
// alongside argv[0] (best-effort).
static std::vector<Scenario> load_scenarios(const std::string& path_hint) {
    std::vector<std::string> candidates;
    if (!path_hint.empty()) candidates.push_back(path_hint);
    candidates.push_back("scenarios.md");
    candidates.push_back("../scenarios.md");
    candidates.push_back("../../scenarios.md");
    candidates.push_back("../sim/scenarios.md");
    candidates.push_back("../../sim/scenarios.md");
    // Also try absolute path the user mentioned in case the build was moved
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
    std::cerr << "[scenarios] reading from " << used << "\n";
    std::vector<Scenario> out;
    Scenario cur;
    bool have_cur = false;
    bool in_code  = false;
    int  heading_count = 0;
    int  code_block_count = 0;
    std::string line;
    auto commit = [&]() {
        // Only commit scenarios that actually contain at least one
        // settable initial condition. This filters out top-of-file
        // headings like "# twin_sim scenarios" that are documentation
        // rather than scenarios.
        bool has_content = !cur.q_init.empty()
                        || !cur.dq_init.empty()
                        || cur.chaos
                        || cur.n != 7
                        || cur.duration != 10.0
                        || cur.dt != 1e-3;
        if (have_cur && has_content) { out.push_back(cur); }
        cur = Scenario();
        have_cur = false;
    };
    // Helper: does this line start a fenced code block?
    // Accepts ``` optionally followed by an info string (e.g. ```yaml).
    auto is_fence = [](const std::string& l) {
        size_t i = 0;
        while (i < l.size() && (l[i] == ' ' || l[i] == '\t')) ++i;
        return i + 3 <= l.size() && l[i] == '`' && l[i+1] == '`' && l[i+2] == '`';
    };
    // Helper: strip BOM
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
        // Fenced block toggle (handles indented fences and ```lang)
        if (is_fence(line)) {
            in_code = !in_code;
            if (in_code) ++code_block_count;
            continue;
        }
        if (!in_code) {
            // Heading? Start of new scenario.
            if (!line.empty() && line[0] == '#') {
                commit();
                ++heading_count;
                // Strip leading #'s, spaces, and an optional "N." prefix
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
                // Skip markdown horizontal rules
                if (t == "---" || t == "***" || t == "___") continue;
                if (!cur.description.empty()) cur.description += ' ';
                cur.description += t;
            }
            continue;
        }
        // Inside fenced code block: parse key=value lines
        if (!have_cur) continue;
        // Strip comments after '#' (but only if not in middle of a value)
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
    // ── CLI ───────────────────────────────────────────────────────────────
    int    scenario_idx = 1;       // 1-based, matches md numbering
    bool   list_only    = false;
    std::string scen_path;         // optional explicit scenarios.md path
    // CLI overrides (applied AFTER loading the scenario). Sentinel values
    // (negative for unsigned-ish quantities) mean "not overridden".
    int    cli_n        = -1;
    double cli_duration = -1.0;
    double cli_dt       = -1.0;
    double cli_eps      = -1.0;
    int    cli_chaos    = -1;      // -1=unset, 0=off, 1=on
    bool   plot         = true;
    double link_m       = 0.1;
    double link_L       = 0.5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--scenario" && i+1 < argc) scenario_idx = std::atoi(argv[++i]);
        else if (a == "--scenarios" && i+1 < argc) scen_path    = argv[++i];
        else if (a == "--list")                    list_only    = true;
        else if (a == "--n"        && i+1 < argc) cli_n        = std::atoi(argv[++i]);
        else if (a == "--duration" && i+1 < argc) cli_duration = std::atof(argv[++i]);
        else if (a == "--dt"       && i+1 < argc) cli_dt       = std::atof(argv[++i]);
        else if (a == "--eps"      && i+1 < argc) cli_eps      = std::atof(argv[++i]);
        else if (a == "--chaos")                  cli_chaos    = 1;
        else if (a == "--no-chaos")               cli_chaos    = 0;
        else if (a == "--no-plot")                plot         = false;
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--scenario K] [--scenarios PATH] [--list]"
                         " [--n N] [--duration S] [--dt S] [--eps E]"
                         " [--chaos|--no-chaos] [--no-plot]\n";
            return 1;
        }
    }
    // ── Load scenarios ────────────────────────────────────────────────────
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
    // Apply CLI overrides
    if (cli_n        > 0  ) sc.n        = cli_n;
    if (cli_duration > 0.0) sc.duration = cli_duration;
    if (cli_dt       > 0.0) sc.dt       = cli_dt;
    if (cli_eps      > 0.0) sc.eps      = cli_eps;
    if (cli_chaos    >= 0 ) sc.chaos    = (cli_chaos == 1);
    int    n        = sc.n;
    double duration = sc.duration;
    double dt       = sc.dt;
    double eps      = sc.eps;
    bool   chaos    = sc.chaos;
    std::cerr << "[scenario] #" << scenario_idx << " — " << sc.name << "\n";
    if (!sc.description.empty())
        std::cerr << "           " << sc.description << "\n";
    std::cerr << "           n=" << n << "  duration=" << duration
              << "s  dt=" << dt << "s  chaos=" << (chaos?"on":"off") << "\n";
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
    // ── Initial conditions (from scenario) ────────────────────────────────
    //
    //  All joints default to 0. Scenario overrides q[i] and dq[i] selectively.
    //  Indices that exceed ndof-1 are silently clipped with a warning.
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
    Eigen::VectorXd tau    = Eigen::VectorXd::Zero(ndof);
    // ── Chaos: perturbed pinocchio twin ───────────────────────────────────
    pinocchio::Data data_cha(model);
    Eigen::VectorXd q_cha  = q_pin;
    Eigen::VectorXd dq_cha = dq_pin;
    if (chaos) {
        q_cha[0] += eps;
        std::cerr << "chaos mode: perturbing joint 0 by eps=" << eps << "\n";
    }
    // ── Energy bootstrap ──────────────────────────────────────────────────
    //
    //  Indexing convention (matches link_energies / link_energies_lgn):
    //    com[0]   = fixed anchor link COM (never moves)
    //    com[i]   = COM of link l_i for i=1..n-1, from engine FK
    pinocchio::forwardKinematics(model, data, q_pin);
    pinocchio::updateFramePlacements(model, data);
    const Eigen::Vector3d lc0(0.0, link_L/2.0, 0.0);
    std::vector<Eigen::Vector3d> com_prev(n), com_curr(n);
    com_prev[0] = lc0;
    for (int i = 1; i < n; ++i)
        com_prev[i] = data.oMi[(pinocchio::JointIndex)i].act(lc0);
    // lgn COM bootstrap (parallel buffers so finite-diff doesn't cross engines)
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
    // ── Plot framing ──────────────────────────────────────────────────────
    //
    //  Chain starts horizontal-left from the anchor, so the anchor lives
    //  on the RIGHT side of its pane and the chain swings into the left
    //  and downward arcs.
    //
    //  Pane: width  ~ 2.5 * reach   (room for full swing left + a bit right)
    //        height ~ 3.0 * reach
    //  Anchor:
    //        horizontal: 4/5 across pane (near right edge, since chain is left)
    //        vertical:   1/3 down from top
    const double reach = n * link_L;
    const double pane_w = 2.5 * reach;
    const double pane_h = 3.0 * reach;
    const double anchor_x_lgn = pane_w * 0.80;
    const double anchor_x_pin = pane_w * 0.80 + pane_w;
    const double y_top        = pane_h * (1.0/3.0);
    const double y_bot        = -pane_h * (2.0/3.0);
    // X view spans both panes
    const double view_xmin = 0.0;
    const double view_xmax = 2.0 * pane_w;
    const double view_ymin = y_bot;
    const double view_ymax = y_top + 0.15 * pane_h;      // a little headroom for labels
    // ── Open gnuplot windows ──────────────────────────────────────────────
    //
    //  Open them sequentially, fully draw each empty frame, THEN prompt.
    //  This matches the requested behavior: window, window[, window], enter, go.
    FILE* gp0 = nullptr;   // stick figures
    FILE* gp1 = nullptr;   // energy
    FILE* gp2 = nullptr;   // Lyapunov (chaos only)
    if (plot) {
        auto open_and_init = [](const std::string& cmds) -> FILE* {
            FILE* gp = popen("gnuplot", "w");
            if (!gp) return nullptr;
            std::fprintf(gp, "%s", cmds.c_str());
            std::fprintf(gp, "plot NaN notitle\n");
            std::fflush(gp);
            // Give Qt time to actually show the window before we open the next.
            std::this_thread::sleep_for(std::chrono::milliseconds(350));
            return gp;
        };
        int en_h = chaos ? 380 : 500;
        std::ostringstream cmd0;
        cmd0 << "set term qt 0 size 860,520 position 0,40"
                " title 'twin_sim — dynamics'\n"
                "set grid\nunset key\n"
             << "set xrange [" << view_xmin << ":" << view_xmax << "]\n"
             << "set yrange [" << view_ymin << ":" << view_ymax << "]\n"
                "set size ratio -1\n"
             << "set label 1 'lgn'       at " << anchor_x_lgn
                << ", " << (y_top + pane_h*0.08)
                << " center font ',13'\n"
             << "set label 2 'pinocchio' at " << anchor_x_pin
                << ", " << (y_top + pane_h*0.08)
                << " center font ',13'\n"
                "set title 'Waiting to start...'\n";
        std::ostringstream cmd1;
        cmd1 << "set term qt 1 size 860," << en_h
             << " position 880,40 title 'twin_sim — energy wave'\n"
                "set grid\n"
                "set xrange [-0.5:" << ((double)n-0.5) << "]\n"
                "set xlabel 'link index  (0=base, " << (n-1) << "=tip)'\n"
                "set ylabel 'energy  [J]'\n"
                "set key top right\n"
                "set title 'Waiting to start...'\n";
        // Sequential open: window 0 appears, then window 1, then (if chaos) window 2.
        gp0 = open_and_init(cmd0.str());
        gp1 = open_and_init(cmd1.str());
        if (chaos) {
            std::ostringstream cmd2;
            cmd2 << "set term qt 2 size 860,380 position 880,"
                 << (en_h + 80)
                 << " title 'twin_sim — Lyapunov divergence'\n"
                    "set grid\nset logscale y\n"
                    "set xlabel 't  [s]'\nset ylabel '||δq(t)||₂'\n"
                    "set key top left\n"
                    "set title 'Sensitivity to perturbation  ε = "
                 << eps << "  (log scale)'\n";
            gp2 = open_and_init(cmd2.str());
        }
        if (!gp0 || !gp1 || (chaos && !gp2)) {
            std::cerr << "could not open gnuplot; use --no-plot\n"; return 1;
        }
        std::cerr << "\nAll windows ready. Press Enter to start simulation... ";
        std::cin.get();
    } else {
        std::cout << "t,link,KE_pin,PE_pin,E_pin,KE_lgn,PE_lgn,E_lgn";
        if (chaos) std::cout << ",delta_q_norm";
        std::cout << "\n";
    }
    // ── Sim loop ──────────────────────────────────────────────────────────
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
      Eigen::VectorXd tau_cha = -damping * dq_cha;
        // ── lgn ───────────────────────────────────────────────────────────
        Eigen::VectorXd ddq_lgn = lgn::forward_dynamics(tree, q_lgn, dq_lgn, tau);
        dq_lgn += ddq_lgn * dt;
        q_lgn  += dq_lgn  * dt;
        // ── pinocchio nominal ─────────────────────────────────────────────
        pinocchio::aba(model, data, q_pin, dq_pin, tau);
        Eigen::VectorXd ddq_pin = data.ddq;
        dq_pin += ddq_pin * dt;
        q_pin  += dq_pin  * dt;
        pinocchio::forwardKinematics(model, data, q_pin);
        pinocchio::updateFramePlacements(model, data);
        // ── pinocchio perturbed (chaos mode) ─────────────────────────────
        double delta_norm = 0.0;
        if (chaos) {
            pinocchio::aba(model, data_cha, q_cha, dq_cha, tau);
            Eigen::VectorXd ddq_cha = data_cha.ddq;
            dq_cha += ddq_cha * dt;
            q_cha  += dq_cha  * dt;
            delta_norm = (q_pin - q_cha).norm();
        }
        // ── Energy ────────────────────────────────────────────────────────
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
            // Track the largest of KE and shifted PE over the first ~100
            // draw frames so the y-axis fits both kinds of curves.
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
            // ── Window 0: stick figures ───────────────────────────────────
            std::fprintf(gp0,
                "set xrange [%f:%f]\n"
                "set yrange [%f:%f]\n"
                "set size ratio -1\n"
                "set title sprintf('t = %%.2f s   ||Δq||_inf = %%.2e', %f, %g)\n"
                "plot '-' w linespoints lt 1 lw 2 pt 7 ps 1.4 lc rgb '#E67E22' t 'lgn', "
                     "'-' w linespoints lt 1 lw 2 pt 7 ps 1.4 lc rgb '#1F5FA8' t 'pinocchio'\n",
                view_xmin, view_xmax,
                view_ymin, view_ymax,
                t, res);
            auto fl = poses_lgn(tree, q_lgn);
            for (size_t i = 0; i < fl.xs.size(); ++i)
                std::fprintf(gp0, "%f %f\n", fl.xs[i] + anchor_x_lgn,
                                             fl.ys[i] + y_top);
            std::fprintf(gp0, "e\n");
            auto fp = poses_pin_from_data(model, data);
            for (size_t i = 0; i < fp.xs.size(); ++i)
                std::fprintf(gp0, "%f %f\n", fp.xs[i] + anchor_x_pin,
                                             fp.ys[i] + y_top);
            std::fprintf(gp0, "e\n");
            std::fflush(gp0);
            // ── Window 1: energy wave ─────────────────────────────────────
            //
            //  Four series, KE strong + PE faded, lgn=orange / pin=blue:
            //    KE pin   strong blue       (drawn first)
            //    KE lgn   strong orange     (drawn on top of pin)
            //    PE pin   faded  blue       (drawn first)
            //    PE lgn   faded  orange     (drawn on top of pin)
            //
            //  Plus a horizontal RED dashed line at y = (|ΔE_tot| / ymax) * ymax
            //  mapped through a log scale so even microscopic disagreement
            //  becomes visible:
            //      y_line = ymax * 0.05 * (log10(|ΔE_tot|) - L_LO) / (L_HI - L_LO)
            //  clamped to [0, ymax*0.9]. Reads as: "how many decades has
            //  the disagreement grown from 1e-14 (noise floor)?"
            //  Exact numerical value shown in the title.
            //
            //  PE reference is time-constant (h_ref = n·L) so curves don't
            //  drift on the y-axis between frames.
            const double h_ref = n * link_L;
            const double pe_shift_const = link_m * 9.81 * h_ref;
            double E_tot_pin = 0.0, E_tot_lgn = 0.0;
            for (int i = 0; i < n; ++i) {
                E_tot_pin += en[i].KE     + en[i].PE     + pe_shift_const;
                E_tot_lgn += en_lgn[i].KE + en_lgn[i].PE + pe_shift_const;
            }
            double dE_tot = std::abs(E_tot_pin - E_tot_lgn);
            // Log-scale the dE line so small values are visible.
            // 1e-14 → 0% of ymax,  1e-2 → 90% of ymax.
            const double L_LO = -14.0, L_HI = -2.0;
            double dE_log = (dE_tot > 0.0) ? std::log10(dE_tot) : L_LO;
            double dE_norm = (dE_log - L_LO) / (L_HI - L_LO);
            if (dE_norm < 0.0) dE_norm = 0.0;
            if (dE_norm > 1.0) dE_norm = 1.0;
            double dE_y = ymax * 0.9 * dE_norm;
            std::fprintf(gp1,
                "set yrange [%f:%f]\n"
                "set title sprintf('KE / PE per link   t = %%.2f s   |ΔE_{tot}| = %%.2e J  (red line, log-scaled)', %f, %g)\n"
                "plot '-' w linespoints lt 1 lw 2.5 pt 7 ps 1.3 lc rgb '#1F5FA8' t 'KE  pin', "
                     "'-' w linespoints lt 1 lw 2.5 pt 6 ps 1.4 lc rgb '#E67E22' t 'KE  lgn', "
                     "'-' w linespoints lt 1 lw 1.5 pt 7 ps 0.9 lc rgb '#9DB8D8' t 'PE  pin', "
                     "'-' w linespoints lt 1 lw 1.5 pt 6 ps 1.1 lc rgb '#F5C99B' t 'PE  lgn', "
                     "'-' w lines lt 2 lw 2 lc rgb '#B2182B' t '|ΔE_{tot}| (log-mapped)'\n",
                -0.05*ymax, ymax, t, dE_tot);
            for (int i=0;i<n;++i) std::fprintf(gp1, "%d %f\n", i, en[i].KE);
            std::fprintf(gp1, "e\n");
            for (int i=0;i<n;++i) std::fprintf(gp1, "%d %f\n", i, en_lgn[i].KE);
            std::fprintf(gp1, "e\n");
            for (int i=0;i<n;++i)
                std::fprintf(gp1, "%d %f\n", i, en[i].PE + pe_shift_const);
            std::fprintf(gp1, "e\n");
            for (int i=0;i<n;++i)
                std::fprintf(gp1, "%d %f\n", i, en_lgn[i].PE + pe_shift_const);
            std::fprintf(gp1, "e\n");
            std::fprintf(gp1, "%d %f\n", 0,   dE_y);
            std::fprintf(gp1, "%d %f\n", n-1, dE_y);
            std::fprintf(gp1, "e\n");
            std::fflush(gp1);
            // ── Window 2: Lyapunov divergence ─────────────────────────────
            if (gp2 && lyap_t.size() >= 2) {
                std::fprintf(gp2,
                    "set xrange [0:%f]\n"
                    "set title sprintf('||δq(t)||₂   ε=%g   t=%%.1fs', %f)\n"
                    "plot '-' w lines lw 2 lc rgb '#B2182B' t '||δq||₂'\n",
                    duration, eps, t);
                for (size_t k = 0; k < lyap_t.size(); ++k)
                    std::fprintf(gp2, "%f %e\n", lyap_t[k], lyap_d[k]);
                std::fprintf(gp2, "e\n");
                std::fflush(gp2);
            }
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
    // ── Final report ──────────────────────────────────────────────────────
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
              << " steps, dt=" << dt << "s)\n";
    auto close = [](FILE* gp) {
        std::fprintf(gp, "pause -1 'press Enter to close...'\n");
        std::fflush(gp);
        pclose(gp);
    };
    if (gp0) close(gp0);
    if (gp1) close(gp1);
    if (gp2) close(gp2);
    return 0;
}
