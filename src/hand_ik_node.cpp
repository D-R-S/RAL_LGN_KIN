// ============================================================================
//  src/hand_ik_node.cpp
//  ROS2 node — one instance per hand (left or right).
//
//  Subscribes:
//    ~/<hand>/fingertip_targets   [geometry_msgs/PoseArray]
//    ~/<hand>/joint_states        [sensor_msgs/JointState]   (warm-start seed)
//    ~/<hand>/servo_states        [sensor_msgs/JointState]   (raw servo angles)
//
//  Publishes:
//    ~/<hand>/ik_solution         [sensor_msgs/JointState]   (joint-space result)
//    ~/<hand>/servo_commands      [sensor_msgs/JointState]   (servo-space commands)
//    /sim/joint_commands          [sensor_msgs/JointState]   (if use_sim:=true)
//
//  Parameters:
//    hand, urdf_path, use_sim, lambda_sq, max_iter, pos_tol, rot_tol,
//    use_differential  (bool, default true for Amazing Hand)
//    servo_topic       (string, topic name for raw servo feedback)
// ============================================================================
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include "lgn/core.hpp"
#include "lgn/kinematic_tree.hpp"
#include "lgn/urdf_loader.hpp"
#include "lgn/ik_solver.hpp"

#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>

using namespace std::chrono_literals;
using JointState = sensor_msgs::msg::JointState;
using PoseArray  = geometry_msgs::msg::PoseArray;
using Pose       = geometry_msgs::msg::Pose;

namespace lgn {

inline T_mat pose_to_T(const Pose& p) {
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x,
                          p.orientation.y, p.orientation.z);
    return T_from_Rp(q.toRotationMatrix(),
                     {p.position.x, p.position.y, p.position.z});
}

// ============================================================================
class HandIKNode : public rclcpp::Node {
public:
    explicit HandIKNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions())
        : rclcpp::Node("hand_ik_node", opts)
    {
        // ── Parameters ──────────────────────────────────────────────────────
        hand_             = declare_parameter("hand",             std::string("right"));
        urdf_path_        = declare_parameter("urdf_path",        std::string(""));
        use_sim_          = declare_parameter("use_sim",          false);
        use_differential_ = declare_parameter("use_differential", true);

        IKParams ikp;
        ikp.lambda_sq  = declare_parameter("lambda_sq",  1e-4);
        ikp.max_iter   = declare_parameter("max_iter",   50);
        ikp.pos_tol    = declare_parameter("pos_tol",    1e-4);
        ikp.rot_tol    = declare_parameter("rot_tol",    1e-3);
        ikp.step_limit = declare_parameter("step_limit", 0.2);
        ik_params_     = ikp;

        // ── Load URDF ────────────────────────────────────────────────────────
        load_urdf();

        hand_solver_ = std::make_unique<HandIKSolver>(*tree_, ik_params_);
        q_current_   = Eigen::VectorXd::Zero(tree_->n_dof());

        // ── Precompute differential B matrices (one per finger) ──────────────
        // Assumes 4 fingers, 2 DOF each, DOFs ordered [f0_flex, f0_abd, ...]
        // Adjust finger_dof_map_ if your URDF has a different DOF ordering.
        if (use_differential_) {
            B_     = differential_B();
            B_inv_ = differential_B_inv();
            servo_cmd_.resize(tree_->n_dof());
            servo_cmd_.setZero();
        }

        // ── Subscriptions ────────────────────────────────────────────────────
        std::string ns = "/" + hand_;

        sub_targets_ = create_subscription<PoseArray>(
            ns + "/fingertip_targets", 10,
            [this](PoseArray::SharedPtr msg){ on_targets(msg); });

        sub_joints_ = create_subscription<JointState>(
            ns + "/joint_states", 10,
            [this](JointState::SharedPtr msg){ on_joint_states(msg); });

        // Raw servo feedback — convert to joint space for warm start
        if (use_differential_) {
            sub_servos_ = create_subscription<JointState>(
                ns + "/servo_states", 10,
                [this](JointState::SharedPtr msg){ on_servo_states(msg); });
        }

        // ── Publishers ────────────────────────────────────────────────────────
        pub_solution_      = create_publisher<JointState>(ns + "/ik_solution",    10);
        pub_servo_commands_= create_publisher<JointState>(ns + "/servo_commands", 10);
        if (use_sim_)
            pub_sim_ = create_publisher<JointState>("/sim/joint_commands", 10);

        RCLCPP_INFO(get_logger(),
            "HandIKNode ready — hand: %s, DOFs: %d, tips: %zu, differential: %s",
            hand_.c_str(), tree_->n_dof(), tree_->tips().size(),
            use_differential_ ? "YES" : "NO");
    }

private:
    void load_urdf() {
        if (!urdf_path_.empty()) {
            tree_ = std::make_unique<KinematicTree>(
                URDFLoader::from_file(urdf_path_));
            RCLCPP_INFO(get_logger(), "Loaded URDF: %s", urdf_path_.c_str());
            return;
        }
        auto pc = std::make_shared<rclcpp::SyncParametersClient>(
            this, "/robot_state_publisher");
        if (pc->wait_for_service(3s)) {
            auto ps = pc->get_parameters({"robot_description"});
            if (!ps.empty() && !ps[0].as_string().empty()) {
                tree_ = std::make_unique<KinematicTree>(
                    URDFLoader::from_string(ps[0].as_string()));
                RCLCPP_INFO(get_logger(), "Loaded URDF from robot_description");
                return;
            }
        }
        throw std::runtime_error("HandIKNode: no URDF found");
    }

    // ── Callback: fingertip target poses ─────────────────────────────────────
    void on_targets(const PoseArray::SharedPtr msg) {
        auto t0 = now();
        const auto& tips = tree_->tips();

        if (msg->poses.size() != tips.size()) {
            RCLCPP_WARN(get_logger(),
                "PoseArray size %zu != tips %zu — skipping",
                msg->poses.size(), tips.size());
            return;
        }

        // Build target map
        std::unordered_map<std::string, T_mat> targets;
        for (size_t i = 0; i < tips.size(); ++i)
            targets[tree_->link(tips[i]).name] = pose_to_T(msg->poses[i]);

        // IK solve (parallel per finger)
        std::unordered_map<std::string, bool> conv;
        Eigen::VectorXd q_sol = hand_solver_->solve_all(q_current_, targets, &conv);
        q_current_ = q_sol;

        double ms = (now() - t0).nanoseconds() * 1e-6;

        // Publish joint-space solution
        publish_joint_state(pub_solution_, q_sol,
                             msg->header.stamp, hand_ + "_palm");

        // If differential hand: convert joint → servo and publish commands
        if (use_differential_) {
            Eigen::VectorXd u = hand_joint_to_servo(q_sol);
            publish_joint_state(pub_servo_commands_, u,
                                 msg->header.stamp, hand_ + "_servo");
        }

        if (use_sim_ && pub_sim_)
            publish_joint_state(pub_sim_, q_sol,
                                 msg->header.stamp, hand_ + "_palm");

        // Convergence summary
        bool all_conv = true;
        for (auto& [n,c] : conv) all_conv &= c;
        RCLCPP_DEBUG(get_logger(),
            "IK %.2f ms — converged: %s", ms, all_conv ? "YES" : "NO");
    }

    // ── Callback: joint state (warm start from joint-space feedback) ──────────
    void on_joint_states(const JointState::SharedPtr msg) {
        const auto& names = tree_->dof_names();
        for (size_t i = 0; i < msg->name.size(); ++i)
            for (int d = 0; d < (int)names.size(); ++d)
                if (msg->name[i] == names[d]) { q_current_[d] = msg->position[i]; break; }
    }

    // ── Callback: servo states → joint space warm start ───────────────────────
    // Converts raw servo angles through differential B matrix before storing.
    void on_servo_states(const JointState::SharedPtr msg) {
        if ((int)msg->position.size() != tree_->n_dof()) return;
        Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(
            msg->position.data(), msg->position.size());
        q_current_ = hand_servo_to_joint(u);
    }

    // ── Publish helper ────────────────────────────────────────────────────────
    void publish_joint_state(
        rclcpp::Publisher<JointState>::SharedPtr& pub,
        const Eigen::VectorXd& q,
        const rclcpp::Time& stamp,
        const std::string& frame_id)
    {
        JointState js;
        js.header.stamp    = stamp;
        js.header.frame_id = frame_id;
        const auto& names  = tree_->dof_names();
        for (int i = 0; i < (int)names.size(); ++i) {
            js.name.push_back(names[i]);
            js.position.push_back(q[i]);
        }
        pub->publish(js);
    }

    // ── Members ────────────────────────────────────────────────────────────────
    std::string  hand_;
    std::string  urdf_path_;
    bool         use_sim_{false};
    bool         use_differential_{true};
    IKParams     ik_params_;
    Mat2         B_, B_inv_;
    Eigen::VectorXd servo_cmd_;

    std::unique_ptr<KinematicTree>  tree_;
    std::unique_ptr<HandIKSolver>   hand_solver_;
    Eigen::VectorXd                 q_current_;

    rclcpp::Subscription<PoseArray>::SharedPtr  sub_targets_;
    rclcpp::Subscription<JointState>::SharedPtr sub_joints_;
    rclcpp::Subscription<JointState>::SharedPtr sub_servos_;
    rclcpp::Publisher<JointState>::SharedPtr    pub_solution_;
    rclcpp::Publisher<JointState>::SharedPtr    pub_servo_commands_;
    rclcpp::Publisher<JointState>::SharedPtr    pub_sim_;
};

} // namespace lgn

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<lgn::HandIKNode>());
    rclcpp::shutdown();
    return 0;
}
