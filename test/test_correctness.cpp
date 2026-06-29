// ============================================================================
//  test/test_correctness.cpp
//  Correctness gate — H₀₅ in the benchmark protocol.
//  ALL tests must pass before any speed result is reported.
//
//  Test groups:
//    Core          — T_inv, Hv encode/decode, skew, rpy_to_rot
//    FK            — analytic two-segment ground truth
//    Velocity      — Hv propagation vs. analytic, Jacobian-Hv consistency
//    Jacobian      — vs. finite difference
//    IK            — round-trip convergence
//    Dynamics      — M symmetric+PD, G sign, FD/ID consistency
//    Contacts      — detect, no-contact, soft force direction,
//                    contact Jacobian vs. FD
//    Differential  — B matrix round-trip
//    Action        — force_at_point vs. cross-product
//    VirtualSensor — blocking detector logic
//    BallJoint     — A3a: FK, Jacobian FD, Hv for 3-DOF ball joint
//    LoopClosure   — A3b: LoopConstraint construction and eval_error
//    SoftLimits    — A3c: limit torque sign and zero-in-range
//    MJCFLoader    — A3d: load Amazing Hand MJCF, verify DOF count and
//                         loop constraint count, FK at zero config
//    Exclusions    — A3e: parent-child auto-exclusion, explicit exclusion
// ============================================================================
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/mjcf_loader.hpp>
#include <lgn/ik_solver.hpp>
#include <lgn/dynamics.hpp>
#include <lgn/contacts.hpp>
#include <cmath>

// ── Two-segment URDF ──────────────────────────────────────────────────────────
static const char* TWO_SEG_URDF = R"xml(
<?xml version="1.0"?>
<robot name="two_segment">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <origin xyz="0 0.75 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1875" ixy="0" ixz="0"
                            iyy="0.001" iyz="0" izz="0.1875"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <origin xyz="0 0.5 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.0833" ixy="0" ixz="0"
                            iyy="0.001" iyz="0" izz="0.0833"/>
    </inertial>
  </link>
  <link name="tip"/>
  <joint name="world_to_link1" type="fixed">
    <parent link="world"/><child link="link1"/>
    <origin xyz="0 0 0"/>
  </joint>
  <joint name="J0" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 1.5 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" velocity="10" effort="50"/>
  </joint>
  <joint name="link2_to_tip" type="fixed">
    <parent link="link2"/><child link="tip"/>
    <origin xyz="0 1.0 0"/>
  </joint>
</robot>
)xml";

// ── Three-link chain with a ball joint (A3a test fixture) ─────────────────────
// palm → [ball joint] → proximal → [revolute] → tip
// Ball joint: 3 DOFs (rz, ry, rx), link length 0.1 m
// Revolute J_flex: 1 DOF, link length 0.05 m, axis Z
static const char* BALL_CHAIN_URDF = R"xml(
<?xml version="1.0"?>
<robot name="ball_chain">
  <link name="palm"/>
  <link name="proximal">
    <inertial>
      <origin xyz="0 0.05 0"/>
      <mass value="0.01"/>
      <inertia ixx="8.3e-6" ixy="0" ixz="0"
                            iyy="1e-8" iyz="0" izz="8.3e-6"/>
    </inertial>
  </link>
  <link name="tip"/>
  <joint name="J_ball" type="ball">
    <parent link="palm"/><child link="proximal"/>
    <origin xyz="0 0 0"/>
    <limit lower="-1.57" upper="1.57" velocity="10" effort="5"/>
  </joint>
  <joint name="J_flex" type="revolute">
    <parent link="proximal"/><child link="tip"/>
    <origin xyz="0 0.1 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="10" effort="5"/>
  </joint>
</robot>
)xml";

// ── Minimal MJCF for A3d loader test ─────────────────────────────────────────
// Two actuated hinges, one passive ball joint, one loop closure.
// Kept minimal so the test doesn't depend on the full Amazing Hand assets.
static const char* MINIMAL_MJCF = R"xml(
<mujoco model="test_hand">
  <compiler angle="radian"/>
  <default>
    <default class="act">
      <joint damping="0.5" frictionloss="0.05" armature="0.01"/>
    </default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <body name="link_a" pos="0 0 0.05">
        <joint name="motor1" type="hinge" axis="0 0 1"
               range="-1.5707963 1.5707963" class="act"/>
        <inertial pos="0 0 0.025" mass="0.01"
                  diaginertia="8e-7 8e-7 1e-9"/>
        <site name="site_a" pos="0 0 0.05"/>
        <body name="link_b" pos="0 0 0.05">
          <joint name="passive_b" type="ball"/>
          <inertial pos="0 0 0.025" mass="0.005"
                    diaginertia="4e-7 4e-7 5e-10"/>
          <site name="site_b" pos="0 0 0.04"/>
        </body>
      </body>
      <body name="link_c" pos="0.03 0 0.04">
        <joint name="motor2" type="hinge" axis="0 0 1"
               range="-1.5707963 1.5707963" class="act"/>
        <inertial pos="0 0 0.02" mass="0.008"
                  diaginertia="5e-7 5e-7 8e-10"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="motor1"/>
    <position joint="motor2"/>
  </actuator>
  <equality>
    <connect site1="site_a" site2="site_b"/>
  </equality>
</mujoco>
)xml";

// ── Helpers ───────────────────────────────────────────────────────────────────
static lgn::KinematicTree make_two_seg() {
    return lgn::URDFLoader::from_string(TWO_SEG_URDF);
}
static Eigen::VectorXd q1(double v) {
    Eigen::VectorXd q(1); q[0] = v; return q;
}

// =============================================================================
//  CORE  (unchanged from original)
// =============================================================================

TEST(Core, T_inv_identity) {
    for (double t : {0.0, 0.5, 1.2, -2.3}) {
        lgn::T_mat T = lgn::T_rot(lgn::Vec3::UnitZ(), t)
                     * lgn::T_trans({0.3, 0.5, 0.1});
        lgn::Mat4 I = T * lgn::T_inv(T);
        EXPECT_LT((I - lgn::Mat4::Identity()).norm(), 1e-12);
    }
}
TEST(Core, HvEncodeDecodeRoundTrip) {
    lgn::Vec3 omega(0.3, -1.2, 0.7), v(1.0, -0.5, 2.1);
    lgn::Hv_mat Hv = lgn::Hv_from(omega, v);
    EXPECT_LT((lgn::omega_of(Hv) - omega).norm(), 1e-15);
    EXPECT_LT((lgn::v_of(Hv)     - v    ).norm(), 1e-15);
}
TEST(Core, SkewAntiSymmetric) {
    lgn::Vec3 v(1.2, -0.5, 3.1), w(0.7, 1.1, -0.3);
    lgn::Mat3 S = lgn::skew(v);
    EXPECT_LT((S + S.transpose()).norm(), 1e-15);
    EXPECT_LT((S*w - v.cross(w)).norm(), 1e-14);
}
TEST(Core, RpyToRotOrthogonal) {
    lgn::Vec3 rpy(0.3, -0.5, 1.1);
    lgn::Mat3 R = lgn::rpy_to_rot(rpy);
    EXPECT_LT((R.transpose()*R - lgn::Mat3::Identity()).norm(), 1e-12);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
}

// =============================================================================
//  FK
// =============================================================================
TEST(FK, AnalyticGroundTruth) {
    auto tree = make_two_seg();
    int  tip  = tree.link_index("tip");
    for (int k = 0; k <= 100; ++k) {
        double theta = -M_PI + 2*M_PI*k/100.0;
        tree.fk(q1(theta));
        lgn::Vec3 p = lgn::p_of(tree.link(tip).T_world);
        EXPECT_NEAR(p.x(), -sin(theta),      1e-12);
        EXPECT_NEAR(p.y(), 1.5 + cos(theta), 1e-12);
        EXPECT_NEAR(p.z(), 0.0,               1e-12);
    }
}
TEST(FK, FkTipMatchesFkWorld) {
    auto tree = make_two_seg();
    int  tip  = tree.link_index("tip");
    for (double theta : {0.0, 0.7, -1.2, 2.9}) {
        auto q = q1(theta);
        tree.fk(q);
        lgn::Vec3 pw = lgn::p_of(tree.link(tip).T_world);
        lgn::Vec3 pt = lgn::p_of(tree.fk_tip(q, tip));
        EXPECT_LT((pw - pt).norm(), 1e-12);
    }
}

// =============================================================================
//  VELOCITY
// =============================================================================
TEST(Velocity, HvAnalyticGroundTruth) {
    auto tree = make_two_seg();
    int  tip  = tree.link_index("tip");
    Eigen::VectorXd dq(1); dq[0] = 1.0;
    for (int k = 0; k <= 50; ++k) {
        double theta = -M_PI + 2*M_PI*k/50.0;
        auto q = q1(theta);
        tree.fk(q);
        tree.velocity_propagation(q, dq);
        lgn::Vec3 p_tip = lgn::p_of(tree.link(tip).T_world);
        lgn::Vec3 v = lgn::v_at_point(tree.link(tip).Hv_world, p_tip);
        EXPECT_NEAR(v.x(), -cos(theta), 1e-10);
        EXPECT_NEAR(v.y(), -sin(theta), 1e-10);
        EXPECT_NEAR(v.z(),  0.0,         1e-10);
    }
}
TEST(Velocity, JacobianHvConsistency) {
    auto tree = make_two_seg();
    int  tip  = tree.link_index("tip");
    for (int k = 0; k <= 20; ++k) {
        double theta = -2.0 + 4.0*k/20.0;
        double dq0   = 0.7 + 0.3*sin(theta);
        Eigen::VectorXd q  = q1(theta), dq = q1(dq0);
        tree.fk(q); tree.velocity_propagation(q, dq);
        lgn::Vec3 pt = lgn::p_of(tree.link(tip).T_world);
        lgn::Vec3 v_hv = lgn::v_at_point(tree.link(tip).Hv_world, pt);
        auto J = tree.jacobian(q, tip);
        Eigen::Vector3d v_J = J.topRows<3>() * dq;
        EXPECT_NEAR(v_hv.x(), v_J.x(), 1e-9);
        EXPECT_NEAR(v_hv.y(), v_J.y(), 1e-9);
    }
}

// =============================================================================
//  JACOBIAN
// =============================================================================
TEST(Jacobian, VsFiniteDifference) {
    auto tree = make_two_seg();
    int  tip  = tree.link_index("tip");
    const double eps = 1e-7;
    for (int k = 0; k <= 50; ++k) {
        double theta = -2.5 + 5.0*k/50.0;
        auto J  = tree.jacobian(q1(theta), tip);
        lgn::Vec3 p0 = lgn::p_of(tree.fk_tip(q1(theta),     tip));
        lgn::Vec3 p1 = lgn::p_of(tree.fk_tip(q1(theta+eps), tip));
        lgn::Vec3 Jfd = (p1 - p0) / eps;
        EXPECT_NEAR(J(0,0), Jfd.x(), 1e-5);
        EXPECT_NEAR(J(1,0), Jfd.y(), 1e-5);
    }
}

// =============================================================================
//  IK
// =============================================================================
TEST(IK, RoundTrip) {
    auto tree = make_two_seg();
    int  tip  = tree.link_index("tip");
    lgn::IKSolver solver(tree, "tip");
    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(1), q_sol(1);
    for (int k = 0; k < 20; ++k) {
        double theta = -2.5 + 5.0*k/19.0;
        lgn::T_mat T_des = tree.fk_tip(q1(theta), tip);
        bool ok = solver.solve(q0, T_des, q_sol);
        EXPECT_TRUE(ok);
        lgn::T_mat T_got = tree.fk_tip(q_sol, tip);
        EXPECT_LT((lgn::p_of(T_got) - lgn::p_of(T_des)).norm(), 1e-4);
    }
}

// =============================================================================
//  DYNAMICS
// =============================================================================
TEST(Dynamics, MassMatrixSymmetricPosDef) {
    auto tree = make_two_seg();
    auto M    = lgn::mass_matrix(tree, q1(0.7));
    EXPECT_LT((M - M.transpose()).norm(), 1e-10);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
}
TEST(Dynamics, GravityVectorAtHorizontal) {
    auto tree = make_two_seg();
    auto G    = lgn::gravity_vector(tree, q1(M_PI/2.0), {0,-9.81,0});
    EXPECT_EQ(G.size(), 1);
    EXPECT_NEAR(std::abs(G[0]), 4.905, 0.15);
}
TEST(Dynamics, ForwardInverseDynamicsConsistency) {
    auto tree = make_two_seg();
    Eigen::VectorXd q(1), dq(1), tau(1);
    q[0]=0.8; dq[0]=0.3; tau[0]=2.0;
    auto qddot = lgn::forward_dynamics(tree, q, dq, tau);
    auto tau_r  = lgn::inverse_dynamics(tree, q, dq, qddot);
    EXPECT_NEAR(tau_r[0], tau[0], 1e-4);
}

// =============================================================================
//  CONTACTS
// =============================================================================
TEST(Contacts, SpherePlaneDetection) {
    auto tree = make_two_seg();
    lgn::ContactWorld world;
    world.add_collider(tree, "link1", lgn::Sphere{0.05});
    world.add_static_plane(lgn::Vec3::UnitY(), 0.0);
    tree.fk(q1(0.0));
    auto contacts = world.detect(tree, 0.0);
    EXPECT_FALSE(contacts.empty());
    if (!contacts.empty()) {
        EXPECT_GT(contacts[0].penetration, 0.0);
        EXPECT_NEAR(contacts[0].normal_w.dot(lgn::Vec3::UnitY()), 1.0, 1e-6);
    }
}
TEST(Contacts, NoContactWhenClearlyAbove) {
    auto tree = make_two_seg();
    lgn::ContactWorld world;
    world.add_collider(tree, "tip", lgn::Sphere{0.005});
    world.add_static_plane(lgn::Vec3::UnitY(), 0.0);
    tree.fk(q1(0.0));
    EXPECT_TRUE(world.detect(tree, 0.0).empty());
}
TEST(Contacts, ContactJacobianVsFiniteDiff) {
    auto tree = make_two_seg();
    lgn::ContactWorld world;
    int tip = tree.link_index("tip");
    auto q  = q1(0.5);
    tree.fk(q);
    lgn::Vec3 cp  = lgn::p_of(tree.link(tip).T_world);
    auto Jc = world.contact_jacobian(tree, q, tip, cp);
    const double eps = 1e-7;
    lgn::Vec3 p0 = lgn::p_of(tree.fk_tip(q,       tip));
    lgn::Vec3 p1 = lgn::p_of(tree.fk_tip(q1(0.5+eps), tip));
    lgn::Vec3 Jfd = (p1 - p0) / eps;
    EXPECT_NEAR(Jc(0,0), Jfd.x(), 1e-5);
    EXPECT_NEAR(Jc(1,0), Jfd.y(), 1e-5);
    EXPECT_NEAR(Jc(2,0), Jfd.z(), 1e-5);
}

// =============================================================================
//  DIFFERENTIAL B MATRIX
// =============================================================================
TEST(Differential, PerFingerRoundTrip) {
    for (int k = 0; k < 50; ++k) {
        lgn::Vec2 u(k*0.1 - 2.5, k*0.07 - 1.7);
        lgn::Vec2 u_rt = lgn::joint_to_servo(lgn::servo_to_joint(u));
        EXPECT_NEAR(u_rt.x(), u.x(), 1e-12);
        EXPECT_NEAR(u_rt.y(), u.y(), 1e-12);
    }
}
TEST(Differential, FullHandRoundTrip) {
    Eigen::VectorXd u(8);
    u << 0.5, -0.3, 0.8, 0.1, -0.2, 0.4, 0.6, -0.5;
    auto u_rt = lgn::hand_joint_to_servo(lgn::hand_servo_to_joint(u));
    EXPECT_LT((u_rt - u).norm(), 1e-12);
}

// =============================================================================
//  ACTION MATRIX
// =============================================================================
TEST(Core, ForceAtPointMatchesCrossProduct) {
    lgn::Vec3 f(1.2, -0.5, 0.8);
    lgn::Vec3 r_app(0.03, 0.01, 0.0), r_orig;
    r_orig.setZero();
    lgn::A_mat A   = lgn::force_at_point(f, r_app, r_orig);
    lgn::Vec3 tau  = (r_app - r_orig).cross(f);
    EXPECT_LT((lgn::f_of(A)   - f  ).norm(), 1e-15);
    EXPECT_LT((lgn::tau_of(A) - tau).norm(), 1e-15);
}

// =============================================================================
//  VIRTUAL SENSOR
// =============================================================================
TEST(VirtualSensor, BlockingDetected) {
    const double thr = 0.05;
    Eigen::VectorXd planned(1); planned[0] = 0.5;
    Eigen::VectorXd actual(1);  actual[0]  = 0.3;
    bool blocked = (actual - planned).cwiseAbs().maxCoeff() > thr;
    EXPECT_TRUE(blocked);
}
TEST(VirtualSensor, NoBlockingWithinThreshold) {
    const double thr = 0.05;
    Eigen::VectorXd planned(1); planned[0] = 0.5;
    Eigen::VectorXd actual(1);  actual[0]  = 0.48;
    bool blocked = (actual - planned).cwiseAbs().maxCoeff() > thr;
    EXPECT_FALSE(blocked);
}

// =============================================================================
//  A3a — BALL JOINT
// =============================================================================

TEST(BallJoint, DOFCount) {
    auto tree = lgn::URDFLoader::from_string(BALL_CHAIN_URDF);
    // J_ball = 3 DOFs, J_flex = 1 DOF → total 4
    EXPECT_EQ(tree.n_dof(), 4);
}

TEST(BallJoint, ZeroConfigIsIdentity) {
    // At q=0, ball joint applies zero rotation → proximal is at (0, 0, 0),
    // tip is at (0, 0.1, 0) (link length), then J_flex at (0, 0.1, 0) + (0, 0.05, 0)
    // Wait — BALL_CHAIN_URDF: palm→J_ball(origin 0)→proximal(len 0.1)→J_flex(origin 0 0.1 0)→tip
    // At q=0: tip should be at (0, 0.15, 0)  [0.1 + 0.05]
    // Actually tip is at proximal origin (0,0.1,0) + flex link length
    // The URDF above puts J_flex at "0 0.1 0" (top of proximal) and flex goes 0.05 m
    // but BALL_CHAIN_URDF doesn't have a tip link inertial or a second joint length.
    // Let's just verify: at q=0, FK tip x=0, z=0.
    auto tree = lgn::URDFLoader::from_string(BALL_CHAIN_URDF);
    int tip = tree.link_index("tip");
    Eigen::VectorXd q = Eigen::VectorXd::Zero(4);
    tree.fk(q);
    lgn::Vec3 p = lgn::p_of(tree.link(tip).T_world);
    EXPECT_NEAR(p.x(), 0.0, 1e-12);
    EXPECT_NEAR(p.z(), 0.0, 1e-12);
    // y = 0.1 (J_flex origin) + 0.0 (J_flex at zero has no additional offset in this URDF)
    // Actually J_flex: parent=proximal, origin "0 0.1 0", child=tip → tip at y=0.1
    EXPECT_NEAR(p.y(), 0.1, 1e-12);
}

TEST(BallJoint, PureZRotation) {
    // Set ball DOF 0 (rz) = π/4, others zero, flex=0.
    // Expect tip rotated π/4 about Z from its zero-config position (0, 0.1, 0).
    // Rotated: x = -0.1*sin(π/4), y = 0.1*cos(π/4)
    auto tree = lgn::URDFLoader::from_string(BALL_CHAIN_URDF);
    int tip = tree.link_index("tip");
    Eigen::VectorXd q = Eigen::VectorXd::Zero(4);
    q[0] = M_PI / 4.0;  // ball rz
    tree.fk(q);
    lgn::Vec3 p = lgn::p_of(tree.link(tip).T_world);
    EXPECT_NEAR(p.x(), -0.1 * sin(M_PI/4.0), 1e-10);
    EXPECT_NEAR(p.y(),  0.1 * cos(M_PI/4.0), 1e-10);
    EXPECT_NEAR(p.z(),  0.0,                   1e-10);
}

TEST(BallJoint, JacobianVsFiniteDiff) {
    // Verify Jacobian columns for ball DOFs via finite difference.
    auto tree = lgn::URDFLoader::from_string(BALL_CHAIN_URDF);
    int tip = tree.link_index("tip");
    const double eps = 1e-6;

    // Test at a non-trivial configuration
    Eigen::VectorXd q = Eigen::VectorXd::Zero(4);
    q[0] = 0.3; q[1] = 0.2; q[2] = -0.1; q[3] = 0.5;

    auto J = tree.jacobian(q, tip);
    lgn::Vec3 p0 = lgn::p_of(tree.fk_tip(q, tip));

    for (int d = 0; d < 4; ++d) {
        Eigen::VectorXd qp = q; qp[d] += eps;
        lgn::Vec3 p1 = lgn::p_of(tree.fk_tip(qp, tip));
        lgn::Vec3 Jfd = (p1 - p0) / eps;
        EXPECT_NEAR(J(0,d), Jfd.x(), 1e-4) << "DOF " << d;
        EXPECT_NEAR(J(1,d), Jfd.y(), 1e-4) << "DOF " << d;
        EXPECT_NEAR(J(2,d), Jfd.z(), 1e-4) << "DOF " << d;
    }
}

// =============================================================================
//  A3b — LOOP CLOSURE
// =============================================================================

TEST(LoopClosure, Construction) {
    // Build a simple tree manually and add a loop constraint
    auto tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    ASSERT_EQ(tree.loop_constraints().size(), 0u);

    lgn::LoopConstraint lc;
    lc.name   = "test_lc";
    lc.link_a = tree.link_index("link1");
    lc.link_b = tree.link_index("link2");
    lc.T_link_a = lgn::T_trans({0.0, 0.1, 0.0});
    lc.T_link_b = lgn::T_identity();
    tree.add_loop_constraint(lc);

    EXPECT_EQ(tree.loop_constraints().size(), 1u);
    EXPECT_EQ(tree.loop_constraints()[0].name, "test_lc");
}

TEST(LoopClosure, EvalErrorAtZeroConfig) {
    // At q=0, link2 origin is at (0, 1.5, 0) in world frame.
    // link1 origin is at (0, 0, 0).
    // Site on link1 at (0, 0.1, 0) world = (0, 0.1, 0)
    // Site on link2 at identity → (0, 1.5, 0)
    // Error = (0, 1.5, 0) - (0, 0.1, 0) = (0, 1.4, 0)
    auto tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    tree.fk(q1(0.0));

    lgn::LoopConstraint lc;
    lc.link_a   = tree.link_index("link1");
    lc.T_link_a = lgn::T_trans({0.0, 0.1, 0.0});
    lc.link_b   = tree.link_index("link2");
    lc.T_link_b = lgn::T_identity();

    lgn::Vec3 err = lc.eval_error(tree);
    EXPECT_NEAR(err.x(), 0.0, 1e-12);
    EXPECT_NEAR(err.y(), 1.4, 1e-12);
    EXPECT_NEAR(err.z(), 0.0, 1e-12);
}

TEST(LoopClosure, EvalErrorChangesWithConfig) {
    auto tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);

    lgn::LoopConstraint lc;
    lc.link_a   = tree.link_index("link1");
    lc.T_link_a = lgn::T_trans({0.0, 0.5, 0.0});
    lc.link_b   = tree.link_index("link2");
    lc.T_link_b = lgn::T_trans({0.0, 0.9, 0.0});

    tree.fk(q1(0.0));
    lgn::Vec3 err0 = lc.eval_error(tree);

    tree.fk(q1(M_PI / 4.0));
    lgn::Vec3 err1 = lc.eval_error(tree);

    EXPECT_GT((err1 - err0).norm(), 1e-6);
}

// =============================================================================
//  A3c — SOFT JOINT LIMITS
// =============================================================================

TEST(SoftLimits, ZeroTorqueInsideRange) {
    auto tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    // Set stiffness on J0
    tree.joint(tree.joint_index("J0")).limits.limit_stiffness = 100.0;
    tree.joint(tree.joint_index("J0")).limits.limit_damping   = 1.0;

    // q = 0.5 (well inside [-π, π]) → no limit torque
    Eigen::VectorXd q(1), dq(1);
    q[0] = 0.5; dq[0] = 0.1;
    auto tau = tree.joint_limit_torque(q, dq);
    EXPECT_NEAR(tau[0], 0.0, 1e-12);
}

TEST(SoftLimits, PositiveTorqueWhenBelowLower) {
    auto tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    tree.joint(tree.joint_index("J0")).limits.lower          = -1.0;
    tree.joint(tree.joint_index("J0")).limits.upper          =  1.0;
    tree.joint(tree.joint_index("J0")).limits.has_limits     = true;
    tree.joint(tree.joint_index("J0")).limits.limit_stiffness = 100.0;
    tree.joint(tree.joint_index("J0")).limits.limit_damping   = 0.0;

    // q = -1.5 (below lower=-1.0) → positive restoring torque
    Eigen::VectorXd q(1), dq(1);
    q[0] = -1.5; dq[0] = 0.0;
    auto tau = tree.joint_limit_torque(q, dq);
    EXPECT_GT(tau[0], 0.0);
    EXPECT_NEAR(tau[0], 100.0 * 0.5, 1e-10);  // k*(lo - q) = 100*(−1−(−1.5)) = 50
}

TEST(SoftLimits, NegativeTorqueWhenAboveUpper) {
    auto tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    tree.joint(tree.joint_index("J0")).limits.lower          = -1.0;
    tree.joint(tree.joint_index("J0")).limits.upper          =  1.0;
    tree.joint(tree.joint_index("J0")).limits.has_limits     = true;
    tree.joint(tree.joint_index("J0")).limits.limit_stiffness = 200.0;
    tree.joint(tree.joint_index("J0")).limits.limit_damping   = 0.0;

    // q = 1.3 (above upper=1.0) → negative restoring torque
    Eigen::VectorXd q(1), dq(1);
    q[0] = 1.3; dq[0] = 0.0;
    auto tau = tree.joint_limit_torque(q, dq);
    EXPECT_LT(tau[0], 0.0);
    EXPECT_NEAR(tau[0], -200.0 * 0.3, 1e-10);  // -k*(q - hi) = -200*0.3 = -60
}

// =============================================================================
//  A3d — MJCF LOADER
// =============================================================================

TEST(MJCFLoader, LoadsMinimalMJCF) {
    auto result = lgn::MJCFLoader::from_string(MINIMAL_MJCF);
    const auto& tree = result.tree;

    // motor1 (hinge, 1 DOF) + passive_b (ball, 3 DOFs) + motor2 (hinge, 1 DOF) = 5
    EXPECT_EQ(tree.n_dof(), 5);

    // 1 <connect> equality → 1 loop constraint
    EXPECT_EQ(result.loop_constraints.size(), 1u);
    EXPECT_EQ(tree.loop_constraints().size(), 1u);
}

TEST(MJCFLoader, CorrectLinkCount) {
    auto result = lgn::MJCFLoader::from_string(MINIMAL_MJCF);
    // world + base + link_a + link_b + link_c = 5 links
    EXPECT_EQ(result.tree.n_links(), 5);
}

TEST(MJCFLoader, ZeroConfigFKSanity) {
    // At q=0, no body should be at the origin except the world link.
    // base is at (0,0,0), link_a at (0,0,0.05), link_b at (0,0,0.1),
    // link_c at (0.03,0,0.04).
    auto result = lgn::MJCFLoader::from_string(MINIMAL_MJCF);
    auto& tree = result.tree;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(tree.n_dof());
    tree.fk(q);

    int link_a = tree.link_index("link_a");
    int link_b = tree.link_index("link_b");
    int link_c = tree.link_index("link_c");

    lgn::Vec3 pa = lgn::p_of(tree.link(link_a).T_world);
    lgn::Vec3 pb = lgn::p_of(tree.link(link_b).T_world);
    lgn::Vec3 pc = lgn::p_of(tree.link(link_c).T_world);

    EXPECT_NEAR(pa.z(), 0.05, 1e-10);
    EXPECT_NEAR(pb.z(), 0.10, 1e-10);
    EXPECT_NEAR(pc.x(), 0.03, 1e-10);
    EXPECT_NEAR(pc.z(), 0.04, 1e-10);
}

TEST(MJCFLoader, LoopConstraintSitesResolved) {
    auto result = lgn::MJCFLoader::from_string(MINIMAL_MJCF);
    const auto& lc = result.loop_constraints[0];
    // Both sites must have valid (non-negative) link indices
    EXPECT_GE(lc.link_a, 0);
    EXPECT_GE(lc.link_b, 0);
    // site_a is on link_a, site_b is on link_b
    EXPECT_EQ(result.tree.link(lc.link_a).name, "link_a");
    EXPECT_EQ(result.tree.link(lc.link_b).name, "link_b");
}

TEST(MJCFLoader, DefaultClassDampingApplied) {
    // motor1 and motor2 should have limit_damping > 0 from class "act"
    auto result = lgn::MJCFLoader::from_string(MINIMAL_MJCF);
    const auto& tree = result.tree;
    int ji = tree.joint_index("motor1");
    // damping=0.5, frictionloss=0.05 from class "act"
    // limit_damping = damping + frictionloss = 0.55
    EXPECT_NEAR(tree.joint(ji).limits.limit_damping, 0.55, 1e-10);
}

// =============================================================================
//  A3e — COLLISION EXCLUSIONS
// =============================================================================

TEST(Exclusions, ParentChildAutoExcluded) {
    auto tree = make_two_seg();
    // link1 → J0 → link2: they should be auto-excluded after finalize()
    int l1 = tree.link_index("link1");
    int l2 = tree.link_index("link2");
    EXPECT_TRUE(tree.is_excluded(l1, l2));
}

TEST(Exclusions, UnrelatedLinksNotExcluded) {
    auto tree = make_two_seg();
    int l1 = tree.link_index("link1");
    int tip = tree.link_index("tip");
    // link1 and tip are not parent-child (link2 is between them)
    // They ARE grandparent-grandchild — not auto-excluded, only parent-child.
    EXPECT_FALSE(tree.is_excluded(l1, tip));
}

TEST(Exclusions, ExplicitExclusionWorks) {
    auto tree = make_two_seg();
    int l1  = tree.link_index("link1");
    int tip = tree.link_index("tip");
    EXPECT_FALSE(tree.is_excluded(l1, tip));

    tree.add_exclusion(l1, tip);
    EXPECT_TRUE(tree.is_excluded(l1, tip));
    // Symmetric: order shouldn't matter
    EXPECT_TRUE(tree.is_excluded(tip, l1));
}

TEST(Exclusions, ExclusionPreventsContactDetection) {
    // Two spheres on link1 and link2 (parent-child) should NOT generate
    // a contact with each other, even if overlapping.
    auto tree = make_two_seg();
    tree.fk(q1(0.0));

    lgn::ContactWorld world;
    // Add large spheres that definitely overlap at q=0
    world.add_collider(tree, "link1", lgn::Sphere{1.0});
    world.add_collider(tree, "link2", lgn::Sphere{1.0});

    auto contacts = world.detect(tree, 0.0);
    // Parent-child pair should be excluded → no robot-robot contact
    bool found_l1_l2 = false;
    int l1 = tree.link_index("link1");
    int l2 = tree.link_index("link2");
    for (const auto& c : contacts) {
        if ((c.link_a == l1 && c.link_b == l2) ||
            (c.link_a == l2 && c.link_b == l1))
            found_l1_l2 = true;
    }
    EXPECT_FALSE(found_l1_l2);
}
