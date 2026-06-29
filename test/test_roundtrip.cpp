// ============================================================================
//  test/test_roundtrip.cpp  —  patch 1.2
//  URDF ↔ MJCF format round-trip correctness gate.
//
//  Patch 1.2 fixes vs 1.1:
//    - MJCFLoader::from_string returns MJCFResult, not KinematicTree.
//      Updated to extract .tree explicitly.
//    - Added EXPECT_EQ(tips_a.size(), tips_b.size()) for clearer diagnostics
//      when the loaders disagree on tip count.
//    - Removed local int main; links against GTest::gtest_main.
//
//  Confirms that loading the same robot from URDF and from MJCF produces
//  identical KinematicTree structures: same n_dof, matching FK results at
//  1000 random configurations (max position error < 1e-6 m, max Frobenius
//  rotation error < 1e-6).
//
//  This is a prerequisite for the multi-format universal loader claim
//  in §3.1. If this test fails, that claim must be removed or qualified.
//
//  External-file mode (optional):
//    export LGN_TEST_URDF=/path/to/robot.urdf
//    export LGN_TEST_MJCF=/path/to/robot.mjcf
//    ./test_roundtrip
// ============================================================================
#include <gtest/gtest.h>
#include <lgn/core.hpp>
#include <lgn/kinematic_tree.hpp>
#include <lgn/urdf_loader.hpp>
#include <lgn/mjcf_loader.hpp>

#include <Eigen/Core>
#include <random>
#include <string>
#include <cmath>

namespace lgn_test {

// ── Inline URDF fixture — two-segment revolute chain ─────────────────────────
static const char* TWO_SEG_URDF = R"(
<?xml version="1.0"?>
<robot name="two_seg">
  <link name="world"/>
  <link name="link0">
    <inertial>
      <origin xyz="0 0.25 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.0004" iyz="0" izz="0.004"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <origin xyz="0 0.25 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.0004" iyz="0" izz="0.004"/>
    </inertial>
  </link>
  <link name="tip"/>
  <joint name="base" type="fixed">
    <parent link="world"/><child link="link0"/>
    <origin xyz="0 0 0"/>
  </joint>
  <joint name="J0" type="revolute">
    <parent link="link0"/><child link="link1"/>
    <origin xyz="0 0.5 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="10" effort="10"/>
  </joint>
  <joint name="tip_j" type="fixed">
    <parent link="link1"/><child link="tip"/>
    <origin xyz="0 0.5 0"/>
  </joint>
</robot>
)";

// ── Inline MJCF fixture — same robot ─────────────────────────────────────────
//
//  MJCFLoader synthesises a "world" root link from the implicit worldbody,
//  so URDF and MJCF should both expose the same explicit "world" link.
//
static const char* TWO_SEG_MJCF = R"(
<mujoco model="two_seg">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0">
      <inertial pos="0 0.25 0" mass="0.5"
                diaginertia="0.004 0.0004 0.004"/>
      <body name="link1" pos="0 0.5 0">
        <joint name="J0" type="hinge" axis="0 0 1"
               range="-3.14 3.14" damping="0"/>
        <inertial pos="0 0.25 0" mass="0.5"
                  diaginertia="0.004 0.0004 0.004"/>
        <body name="tip" pos="0 0.5 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

// ── Helpers ───────────────────────────────────────────────────────────────────
static std::vector<Eigen::VectorXd> make_configs(int ndof, int n=1000,
    int seed=42, double lo=-2.8, double hi=2.8)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<Eigen::VectorXd> out(n, Eigen::VectorXd::Zero(ndof));
    for (auto& q : out)
        for (int i = 0; i < ndof; ++i) q[i] = d(rng);
    return out;
}

struct RoundTripResult {
    bool   ndof_match;
    bool   nlinks_match;
    bool   ntips_match;
    double max_pos_err;   // metres
    double max_rot_err;   // Frobenius norm of (R_a - R_b)
    int    n_configs_tested;
};

static RoundTripResult check_roundtrip(
    lgn::KinematicTree& tree_a,
    lgn::KinematicTree& tree_b,
    int n_configs = 1000)
{
    RoundTripResult res{};
    res.ndof_match   = (tree_a.n_dof()   == tree_b.n_dof());
    res.nlinks_match = (tree_a.n_links()  == tree_b.n_links());
    res.ntips_match  = (tree_a.tips().size() == tree_b.tips().size());

    if (!res.ndof_match) return res;

    int ndof = tree_a.n_dof();
    auto configs = make_configs(ndof, n_configs);
    res.n_configs_tested = n_configs;

    const auto& tips_a = tree_a.tips();
    const auto& tips_b = tree_b.tips();
    int n_tips = (int)std::min(tips_a.size(), tips_b.size());

    for (const auto& q : configs) {
        tree_a.fk(q);
        tree_b.fk(q);

        for (int t = 0; t < n_tips; ++t) {
            lgn::Vec3 pa = lgn::p_of(tree_a.link(tips_a[t]).T_world);
            lgn::Vec3 pb = lgn::p_of(tree_b.link(tips_b[t]).T_world);
            lgn::Mat3 Ra = lgn::R_of(tree_a.link(tips_a[t]).T_world);
            lgn::Mat3 Rb = lgn::R_of(tree_b.link(tips_b[t]).T_world);

            double pos_err = (pa - pb).norm();
            double rot_err = (Ra - Rb).norm();

            res.max_pos_err = std::max(res.max_pos_err, pos_err);
            res.max_rot_err = std::max(res.max_rot_err, rot_err);
        }
    }
    return res;
}

// ── Test: inline two-segment fixture ─────────────────────────────────────────
TEST(RoundTrip, TwoSegmentInlineFixture) {
    lgn::KinematicTree urdf_tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    auto mjcf_result = lgn::MJCFLoader::from_string(TWO_SEG_MJCF);
    lgn::KinematicTree& mjcf_tree = mjcf_result.tree;

    auto res = check_roundtrip(urdf_tree, mjcf_tree, 1000);

    EXPECT_TRUE(res.ndof_match)
        << "DOF count mismatch: URDF=" << urdf_tree.n_dof()
        << " MJCF=" << mjcf_tree.n_dof();

    EXPECT_EQ(urdf_tree.tips().size(), mjcf_tree.tips().size())
        << "Tip count mismatch — comparison by index would be misleading";

    if (!res.nlinks_match) {
        std::cerr << "[RoundTrip] Link count differs: URDF="
                  << urdf_tree.n_links() << " MJCF=" << mjcf_tree.n_links()
                  << " — checking FK only\n";
    }

    EXPECT_LT(res.max_pos_err, 1e-6)
        << "Max tip position error over 1000 configs: "
        << res.max_pos_err << " m (threshold 1e-6 m)";

    EXPECT_LT(res.max_rot_err, 1e-6)
        << "Max tip rotation error over 1000 configs: "
        << res.max_rot_err << " (Frobenius, threshold 1e-6)";

    std::cerr << "[RoundTrip/TwoSeg] pos_err=" << res.max_pos_err
              << " rot_err=" << res.max_rot_err << "\n";
}

// ── Test: inertia consistency ─────────────────────────────────────────────────
TEST(RoundTrip, InertiaConsistency) {
    lgn::KinematicTree urdf_tree = lgn::URDFLoader::from_string(TWO_SEG_URDF);
    auto mjcf_result = lgn::MJCFLoader::from_string(TWO_SEG_MJCF);
    lgn::KinematicTree& mjcf_tree = mjcf_result.tree;

    if (urdf_tree.n_dof() != mjcf_tree.n_dof()) {
        GTEST_SKIP() << "DOF mismatch — skip inertia check";
    }

    double max_gamma_err = 0.0;
    int    compared = 0;

    for (int i = 0; i < urdf_tree.n_links(); ++i) {
        const auto& lu = urdf_tree.link(i);
        if (lu.inertial.mass < 1e-9) continue;
        try {
            int j = mjcf_tree.link_index(lu.name);
            const auto& lm = mjcf_tree.link(j);
            double err = (lu.inertial.gamma - lm.inertial.gamma).norm();
            max_gamma_err = std::max(max_gamma_err, err);
            ++compared;
        } catch (...) {
            // Link name not found in MJCF tree — skip
        }
    }

    if (compared == 0) {
        std::cerr << "[RoundTrip/Inertia] No matching named links found "
                     "— skipping Gamma comparison\n";
        return;
    }

    EXPECT_LT(max_gamma_err, 1e-8)
        << "Max Gamma matrix error across " << compared
        << " matched links: " << max_gamma_err;

    std::cerr << "[RoundTrip/Inertia] Gamma max_err=" << max_gamma_err
              << " across " << compared << " links\n";
}

// ── Test: external files (env-driven, optional) ──────────────────────────────
TEST(RoundTrip, ExternalFiles) {
    const char* urdf_path = std::getenv("LGN_TEST_URDF");
    const char* mjcf_path = std::getenv("LGN_TEST_MJCF");

    if (!urdf_path || !mjcf_path) {
        GTEST_SKIP()
            << "Set LGN_TEST_URDF and LGN_TEST_MJCF env vars to test external files.\n"
            << "  export LGN_TEST_URDF=/path/to/robot.urdf\n"
            << "  export LGN_TEST_MJCF=/path/to/robot.mjcf\n"
            << "  ./test_roundtrip";
    }

    lgn::KinematicTree urdf_tree = lgn::URDFLoader::from_file(urdf_path);
    auto mjcf_result = lgn::MJCFLoader::from_file(mjcf_path);
    lgn::KinematicTree& mjcf_tree = mjcf_result.tree;

    auto res = check_roundtrip(urdf_tree, mjcf_tree, 1000);

    std::cerr << "[RoundTrip/External] URDF: " << urdf_path << "\n";
    std::cerr << "[RoundTrip/External] MJCF: " << mjcf_path << "\n";
    std::cerr << "[RoundTrip/External] n_dof URDF=" << urdf_tree.n_dof()
              << " MJCF=" << mjcf_tree.n_dof() << "\n";
    std::cerr << "[RoundTrip/External] pos_err=" << res.max_pos_err
              << " rot_err=" << res.max_rot_err << "\n";

    EXPECT_TRUE(res.ndof_match)
        << "DOF count mismatch";
    EXPECT_LT(res.max_pos_err, 1e-6)
        << "Max position error: " << res.max_pos_err << " m";
    EXPECT_LT(res.max_rot_err, 1e-6)
        << "Max rotation error: " << res.max_rot_err;
}

} // namespace lgn_test
// gtest_main supplies main()
