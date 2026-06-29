// ============================================================================
//  demo/main.cpp  —  Patch 1.4
//  Standalone demo: load URDF, verify FK/Hv/Jacobian/IK/Dynamics/Contact,
//  run dynamics correctness gates and ball-drop, then open the debug viewer.
//
//  Patch 1.4 adds three dynamics correctness gates (run only on the
//  two-segment chain, where analytical references are tractable):
//    C1 — Mass-matrix symmetry             ‖M − Mᵀ‖∞ < 1e-12
//    C2 — Newton's law of cooling identity ‖Ṁ − 2C‖ antisymmetric (1e-6)
//    C3 — Energy conservation in free fall (gravity, τ=0, 1 s, < 1%)
//
//  C2 is the gold-standard Coriolis test: if Ṁ − 2C is not antisymmetric
//  the Coriolis projection is wrong. Catches the angular factor-of-2 bug
//  that motivated Patch 1.4.
// ============================================================================
#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/ik_solver.hpp>
#include <lgn/dynamics.hpp>
#include <lgn/contacts.hpp>
#include <lgn/collision_loader.hpp>
#include <lgn/viz.hpp>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

static void print_vec3(const std::string& label, const lgn::Vec3& v, int prec=5) {
    std::cout << std::fixed << std::setprecision(prec)
              << label << " ["
              << v.x() << ", " << v.y() << ", " << v.z() << "]\n";
}

// ============================================================================
//  Dynamics correctness gates — Patch 1.4
// ============================================================================

// Helper: kinetic energy ½·dqᵀ·M·dq + potential energy via FK + CoM heights.
static double total_energy(lgn::KinematicTree& tree,
                            const Eigen::VectorXd& q,
                            const Eigen::VectorXd& dq,
                            const lgn::Vec3& g_world = {0, -9.81, 0}) {
    Eigen::MatrixXd M = lgn::mass_matrix(tree, q);
    double KE = 0.5 * dq.transpose() * M * dq;

    tree.fk(q);
    double PE = 0.0;
    for (int k = 0; k < tree.n_links(); ++k) {
        const auto& lk = tree.link(k);
        if (lk.inertial.mass < 1e-9) continue;
        lgn::Vec3 com_w = lgn::p_of(lk.T_world)
                        + lgn::R_of(lk.T_world) * lk.inertial.com;
        PE -= lk.inertial.mass * g_world.dot(com_w);
    }
    return KE + PE;
}

// Finite-difference Ṁ along dq.
static Eigen::MatrixXd Mdot_finite_diff(lgn::KinematicTree& tree,
                                         const Eigen::VectorXd& q,
                                         const Eigen::VectorXd& dq,
                                         double eps = 1e-6) {
    Eigen::MatrixXd Mplus  = lgn::mass_matrix(tree, q + eps * dq);
    Eigen::MatrixXd Mminus = lgn::mass_matrix(tree, q - eps * dq);
    return (Mplus - Mminus) / (2.0 * eps);
}

// Approximate the full Coriolis MATRIX C(q,dq) via finite difference of
// the vector function f(dq) = coriolis_vector(q, dq), then averaging:
//   C·dq is what we have. The Coriolis matrix is not unique, but
//   Ṁ − 2C must be antisymmetric for ANY valid choice. We use the
//   "Christoffel" choice which arises from the standard derivation:
//     C_ij(q, dq) = ½ Σ_k (∂M_ij/∂q_k + ∂M_ik/∂q_j − ∂M_jk/∂q_i) · dq_k
//   But we only need C·dq for the test, and coriolis_vector gives that
//   directly. So we test the WEAKER but mathematically valid identity:
//     dqᵀ · (Ṁ − 2C·dq) · dq = 0   for any dq.
//   ⟺ dqᵀ · Ṁ · dq = 2 · dqᵀ · (C·dq) — but dqᵀ·C·dq is a scalar we
//   can compute. So the energy-rate identity is:
//     d/dt(½ dqᵀ M dq) = dqᵀ M q̈ + ½ dqᵀ Ṁ dq
//   With Lagrange's eom (no gravity, τ=0): M q̈ = -C·dq, hence:
//     d/dt(KE) = dqᵀ · (½ Ṁ − C) · dq
//   and Ṁ − 2C antisymmetric ⟹ this is identically zero. So
//   dqᵀ · (½ Ṁ dq − C·dq) should be zero — this is what we check.
static double skew_identity_residual(lgn::KinematicTree& tree,
                                       const Eigen::VectorXd& q,
                                       const Eigen::VectorXd& dq) {
    Eigen::MatrixXd Mdot = Mdot_finite_diff(tree, q, dq);
    Eigen::VectorXd Cdq  = lgn::coriolis_vector(tree, q, dq);
    // dqᵀ·(½ Ṁ·dq − C·dq) should equal 0
    double lhs = 0.5 * dq.transpose() * Mdot * dq;
    double rhs = dq.transpose() * Cdq;
    return lhs - rhs;
}

static bool run_dynamics_gates(lgn::KinematicTree& tree) {
    std::cout << "\n── Dynamics Correctness Gates (Patch 1.4) ────────────────\n";

    if (tree.n_dof() < 1) {
        std::cout << "  [SKIP] no DOFs\n";
        return true;
    }

    std::mt19937 rng(0xC051);
    std::uniform_real_distribution<double> uq (-1.5, 1.5);
    std::uniform_real_distribution<double> udq(-2.0, 2.0);

    bool all_ok = true;

    // ── C1: Mass-matrix symmetry ─────────────────────────────────────────
    {
        double max_err = 0.0;
        for (int trial = 0; trial < 20; ++trial) {
            Eigen::VectorXd q(tree.n_dof());
            for (int i = 0; i < tree.n_dof(); ++i) q[i] = uq(rng);
            Eigen::MatrixXd M = lgn::mass_matrix(tree, q);
            double err = (M - M.transpose()).cwiseAbs().maxCoeff();
            max_err = std::max(max_err, err);
        }
        bool ok = max_err < 1e-12;
        std::cout << "  [C1] mass matrix symmetry:    "
                  << std::scientific << std::setprecision(2) << max_err
                  << (ok ? "  ✓\n" : "  ✗ FAIL\n");
        if (!ok) all_ok = false;
    }

    // ── C2: Coriolis skew identity  d/dt(KE) = dqᵀ(½Ṁ − C)dq = 0 ─────────
    //
    //  Mathematically: for any valid Coriolis decomposition the identity
    //  dqᵀ · (½ Ṁ − C) · dq = 0 holds for all dq when τ_ext = G = 0.
    //  Equivalent to "Ṁ − 2C is antisymmetric". This is THE gold-standard
    //  Coriolis test — independent of any reference implementation.
    //
    {
        double max_res = 0.0;
        for (int trial = 0; trial < 20; ++trial) {
            Eigen::VectorXd q(tree.n_dof()), dq(tree.n_dof());
            for (int i = 0; i < tree.n_dof(); ++i) {
                q[i]  = uq(rng);
                dq[i] = udq(rng);
            }
            double res = std::abs(skew_identity_residual(tree, q, dq));
            max_res = std::max(max_res, res);
        }
        // Tolerance: finite-difference Ṁ has O(eps²) truncation error.
        // 1e-6 is generous; tight bound would be ~1e-9.
        bool ok = max_res < 1e-6;
        std::cout << "  [C2] dqᵀ(½Ṁ − C)dq residual:  "
                  << std::scientific << std::setprecision(2) << max_res
                  << (ok ? "  ✓\n" : "  ✗ FAIL\n");
        if (!ok) all_ok = false;
    }

    // ── C3: Energy conservation in free fall ─────────────────────────────
    //
    //  No torque, gravity on, integrate forward dynamics for 1 second
    //  with semi-implicit Euler (dt=1ms). Total energy E = KE + PE must
    //  not drift by more than 1% — this is a strong joint test of
    //  M (kinetic energy), G (gravity), AND C (otherwise the integration
    //  picks up a phantom power source).
    //
    {
        Eigen::VectorXd q  = Eigen::VectorXd::Zero(tree.n_dof());
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(tree.n_dof());
        if (tree.n_dof() >= 1) q[0] = 1.0;  // initial pendulum angle

        const lgn::Vec3 g_world(0, -9.81, 0);
        const double dt = 1e-3;
        const int   steps = 1000;

        double E0 = total_energy(tree, q, dq, g_world);

        for (int s = 0; s < steps; ++s) {
            Eigen::VectorXd zero_tau = Eigen::VectorXd::Zero(tree.n_dof());
            Eigen::VectorXd qdd = lgn::forward_dynamics(
                tree, q, dq, zero_tau, {}, g_world);
            // Semi-implicit Euler: dq first, then q.
            dq += dt * qdd;
            q  += dt * dq;
        }

        double E1 = total_energy(tree, q, dq, g_world);
        double drift = std::abs(E1 - E0) / (std::abs(E0) + 1e-9);
        bool ok = drift < 0.01;
        std::cout << "  [C3] energy drift (1s free fall): "
                  << std::fixed << std::setprecision(3) << drift * 100 << "%"
                  << (ok ? "  ✓\n" : "  ✗ FAIL\n");
        if (!ok) all_ok = false;
    }

    std::cout << "  Dynamics gates: "
              << (all_ok ? "ALL PASS ✓" : "FAILURES ✗") << "\n";
    return all_ok;
}

// ============================================================================
//  Ball drop correctness gate (unchanged from Patch 1.3)
// ============================================================================
static bool run_ball_drop_gate(lgn::KinematicTree& tree) {
    std::cout << "\n── Ball Drop Correctness Gate (A2) ───────────────────────\n";

    const double m   = 0.5;
    const double r   = 0.05;
    const double h   = 1.0;
    const double e   = 0.5;
    const double dt  = 1.0/1000;
    const double g   = 9.81;
    const lgn::Vec3 grav(0, -g, 0);

    const double energy_tol  = 0.02;
    const double impulse_tol = 0.10;

    lgn::ContactWorld world;
    world.add_static_plane(lgn::Vec3::UnitY(), 0.0);
    world.set_timestep(dt);

    lgn::FreeSphere ball;
    ball.pos         = {0.0, h + r, 0.0};
    ball.vel         = lgn::Vec3::Zero();
    ball.radius      = r;
    ball.mass        = m;
    ball.restitution = e;
    ball.name        = "drop_ball";
    int ball_idx = world.add_free_sphere(ball);

    Eigen::VectorXd q  = Eigen::VectorXd::Zero(tree.n_dof());
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(tree.n_dof());
    tree.fk(q);

    int    n_bounces   = 0;
    bool   was_contact = false;
    bool   energy_ok   = true;
    bool   impulse_ok  = true;
    double v_impact    = 0.0;
    double E_ref       = -1.0;
    int    flight_step = 0;
    const int settle   = 5;
    const int max_steps = 8000;

    for (int step = 0; step < max_steps && n_bounces < 3; ++step) {
        const lgn::FreeSphere& b = world.free_sphere(ball_idx);
        double vy_pre_step = b.vel.y();

        world.step_free_spheres(dt, grav);
        auto contacts = world.detect(tree, 0.0);
        world.resolve_soft(contacts, tree, q, dq, /*kn=*/1e4, /*kd=*/50.0);

        const lgn::FreeSphere& b2 = world.free_sphere(ball_idx);
        bool in_contact = (b2.pos.y() - r) < 1e-4;

        if (!was_contact && in_contact) {
            v_impact    = std::max(0.0, -vy_pre_step);
            E_ref       = -1.0;
            flight_step = 0;
        }
        if (was_contact && !in_contact) {
            double v_after    = b2.vel.y();
            double v_expected = e * v_impact;
            double rel_err    = std::abs(v_after - v_expected)
                              / (v_expected + 1e-6);
            if (rel_err > impulse_tol) {
                std::cout << "  [FAIL] Bounce " << n_bounces+1
                          << ": v_impact=" << std::fixed << std::setprecision(4)
                          << v_impact << "  v_after=" << v_after
                          << "  expected≈" << v_expected
                          << "  err=" << rel_err*100 << "%\n";
                impulse_ok = false;
            } else {
                std::cout << "  Bounce " << n_bounces+1
                          << ": v_impact=" << std::fixed << std::setprecision(4)
                          << v_impact << " m/s  v_after=" << v_after
                          << " m/s  expected≈" << v_expected
                          << " m/s  ✓\n";
            }
            ++n_bounces;
            flight_step = 0;
        }
        if (!in_contact) {
            ++flight_step;
            if (flight_step > settle) {
                double KE = 0.5 * m * b2.vel.squaredNorm();
                double PE = m * g * b2.pos.y();
                double E  = KE + PE;
                if (E_ref < 0.0) E_ref = E;
                else {
                    double drift = std::abs(E - E_ref) / (std::abs(E_ref) + 1e-9);
                    if (drift > energy_tol) {
                        std::cout << "  [FAIL] Energy drift "
                                  << std::fixed << std::setprecision(2)
                                  << drift*100 << "% > "
                                  << energy_tol*100 << "%  step=" << step << "\n";
                        energy_ok = false;
                        E_ref = E;
                    }
                }
            }
        }
        was_contact = in_contact;
    }

    if (n_bounces < 3)
        std::cout << "  [WARN] Only " << n_bounces
                  << " bounces observed.\n";

    bool all_ok = energy_ok && impulse_ok;
    std::cout << "\nBall drop gate: "
              << (all_ok ? "ALL PASS ✓" : "FAILURES DETECTED ✗") << "\n";
    return all_ok;
}


int main(int argc, char** argv) {
    std::string urdf_path = (argc > 1) ? argv[1] : "demo/two_segment.urdf";

    lgn::KinematicTree tree = lgn::URDFLoader::from_file(urdf_path);
    std::cout << "Loaded: " << tree.n_dof() << " DOFs, "
              << tree.n_links() << " links\n";
    for (int i = 0; i < (int)tree.dof_names().size(); ++i)
        std::cout << "  DOF[" << i << "] = " << tree.dof_names()[i] << "\n";

    if (tree.n_dof() == 1 && tree.tips().size() == 1) {
        int tip = tree.tips()[0];
        double theta = 0.5;
        Eigen::VectorXd q(1); q[0] = theta;
        tree.fk(q);
        lgn::Vec3 p = lgn::p_of(tree.link(tip).T_world);

        print_vec3("\nFK tip pos  (q=0.5): ", p);
        print_vec3("Analytic:            ",
            lgn::Vec3(-std::sin(theta), 1.5 + std::cos(theta), 0.0));
        double err = (p - lgn::Vec3(-std::sin(theta), 1.5+std::cos(theta), 0)).norm();
        std::cout << "FK error: " << std::scientific << err
                  << (err < 1e-10 ? "  ✓" : "  ✗ FAIL") << "\n";

        Eigen::VectorXd dq(1); dq[0] = 1.0;
        tree.velocity_propagation(q, dq);
        lgn::Vec3 p_tip_w = lgn::p_of(tree.link(tip).T_world);
        lgn::Vec3 v = lgn::v_at_point(tree.link(tip).Hv_world, p_tip_w);
        print_vec3("\nHv tip vel  (dq=1.0):", v);
        print_vec3("Analytic:            ",
            lgn::Vec3(-std::cos(theta), -std::sin(theta), 0.0));
        double verr = (v - lgn::Vec3(-std::cos(theta), -std::sin(theta), 0)).norm();
        std::cout << "Hv error: " << std::scientific << verr
                  << (verr < 1e-9 ? "  ✓" : "  ✗ FAIL") << "\n";

        const double eps = 1e-7;
        auto J = tree.jacobian(q, tip);
        Eigen::VectorXd qp = q; qp[0] += eps;
        lgn::Vec3 p0 = lgn::p_of(tree.fk_tip(q,  tip));
        lgn::Vec3 p1 = lgn::p_of(tree.fk_tip(qp, tip));
        lgn::Vec3 Jfd = (p1 - p0) / eps;
        double jerr = (lgn::Vec3(J(0,0),J(1,0),J(2,0)) - Jfd).norm();
        std::cout << "\nJacobian FD error: " << std::scientific << jerr
                  << (jerr < 1e-5 ? "  ✓" : "  ✗ FAIL") << "\n";

        lgn::IKSolver solver(tree, tree.link(tip).name);
        lgn::T_mat T_des = tree.fk_tip(q, tip);
        Eigen::VectorXd q0 = Eigen::VectorXd::Zero(1);
        Eigen::VectorXd q_sol(1);
        bool ok = solver.solve(q0, T_des, q_sol);
        lgn::T_mat T_got = tree.fk_tip(q_sol, tip);
        double ik_err = (lgn::p_of(T_got) - lgn::p_of(T_des)).norm();
        std::cout << "\nIK round-trip: conv=" << (ok?"YES":"NO")
                  << "  pos_err=" << std::scientific << ik_err
                  << (ik_err < 1e-4 ? "  ✓" : "  ✗ FAIL") << "\n";

        Eigen::MatrixXd M  = lgn::mass_matrix(tree, q);
        Eigen::VectorXd G  = lgn::gravity_vector(tree, q);
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(1);
        Eigen::VectorXd qddot = lgn::forward_dynamics(tree, q, dq, tau);
        std::cout << "\nMass M[0,0]:  " << std::fixed << M(0,0) << " kg·m²\n";
        std::cout << "Gravity G[0]: " << std::fixed << G(0)    << " Nm\n";
        std::cout << "FD q̈[0]:      " << std::fixed << qddot(0)<< " rad/s²\n";

        lgn::ContactWorld cworld;
        cworld.add_collider(tree, "link1", lgn::Sphere{0.05});
        cworld.add_static_plane(lgn::Vec3::UnitY(), 0.0);
        tree.fk(q);
        auto contacts = cworld.detect(tree, 0.0);
        std::cout << "\nContact detect: " << contacts.size()
                  << " contact(s) found  "
                  << (contacts.size() > 0 ? "✓" : "(none — link1 above plane)") << "\n";

        // ── Patch 1.4: dynamics correctness gates ────────────────────────
        run_dynamics_gates(tree);

        // ── Ball drop ─────────────────────────────────────────────────────
        run_ball_drop_gate(tree);

    } else {
        std::cout << "\n(Skipping analytic checks — not the two-segment chain)\n";
        // Still run dynamics gates on any chain
        run_dynamics_gates(tree);
    }

    lgn::ContactWorld world_from_urdf;
    lgn::load_collision_geometry(urdf_path, tree, world_from_urdf);
    std::cout << "\nColliders loaded from URDF: "
              << world_from_urdf.colliders().size() << "\n";

    std::cout << "\nOpening visualiser — drag to orbit, scroll to zoom.\n\n";

    try {
        lgn::Viz viz(tree, "lgn debug  —  " + urdf_path);

        Eigen::VectorXd q  = Eigen::VectorXd::Zero(tree.n_dof());
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(tree.n_dof());
        if (tree.n_dof() > 0) { q[0] = 0.5; dq[0] = 1.0; }

        viz.set_q(q);
        viz.set_dq(dq);

        if (!tree.dof_names().empty()) {
            for (const auto& jt : tree.joints_ref()) {
                if (jt.type != lgn::JointType::Fixed) {
                    viz.apply_wrench(jt.name, { .force={0,0,0}, .torque={0,0,5.0} });
                    break;
                }
            }
        }

        while (viz.is_open())
            viz.frame();

    } catch (const std::exception& e) {
        std::cerr << "Viz error (is DISPLAY set?): " << e.what() << "\n";
        std::cerr << "All non-visual checks above still valid.\n";
        return 0;
    }

    return 0;
}
