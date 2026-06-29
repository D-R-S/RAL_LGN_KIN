// ============================================================================
//  src/virtual_sensor_node.cpp
//
//  The sim observer: runs the lgn dynamics + contact model in real-time
//  alongside the actual hardware, producing virtual sensor readings.
//
//  Core idea:
//    Real hardware gives:   q_actual (servo positions → joint angles)
//    IK planner gives:      q_planned (what was commanded)
//    Discrepancy:           Δq = q_actual - q_planned
//                           → finger blocked → contact occurring
//    Sim model gives:       contact force, point, normal
//                           → virtual tactile reading
//
//  Subscribes:
//    /<hand>/ik_solution         [JointState]  — planned joint angles
//    /<hand>/joint_states        [JointState]  — actual joint angles (from servo fb)
//
//  Publishes:
//    /<hand>/virtual_contacts    [lgn_hand_ik/VirtualContactArray]
//    /<hand>/contact_detected    [std_msgs/Bool]   — any contact?
//    /<hand>/contact_force_est   [geometry_msgs/WrenchStamped] per finger
//
//  Parameters:
//    hand, urdf_path, contact_kn, contact_kd, block_threshold,
//    gravity_x, gravity_y, gravity_z
// ============================================================================
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/bool.hpp>

#include "lgn/core.hpp"
#include "lgn/kinematic_tree.hpp"
#include "lgn/urdf_loader.hpp"
#include "lgn/dynamics.hpp"
#include "lgn/contacts.hpp"
#include "lgn/collision_loader.hpp"

#include <chrono>
#include <string>
#include <memory>

using namespace std::chrono_literals;
using JointState    = sensor_msgs::msg::JointState;
using WrenchStamped = geometry_msgs::msg::WrenchStamped;
using BoolMsg       = std_msgs::msg::Bool;

namespace lgn {

class VirtualSensorNode : public rclcpp::Node {
public:
    explicit VirtualSensorNode(const rclcpp::NodeOptions& opts = {})
        : rclcpp::Node("virtual_sensor_node", opts)
    {
        hand_       = declare_parameter("hand",        std::string("right"));
        urdf_path_  = declare_parameter("urdf_path",   std::string(""));
        kn_         = declare_parameter("contact_kn",  5e4);
        kd_         = declare_parameter("contact_kd",  100.0);
        block_thr_  = declare_parameter("block_threshold", 0.05); // rad
        Vec3 g;
        g.x() = declare_parameter("gravity_x", 0.0);
        g.y() = declare_parameter("gravity_y", -9.81);
        g.z() = declare_parameter("gravity_z", 0.0);
        gravity_ = g;

        load_urdf();

        q_planned_ = Eigen::VectorXd::Zero(tree_->n_dof());
        q_actual_  = Eigen::VectorXd::Zero(tree_->n_dof());
        dq_        = Eigen::VectorXd::Zero(tree_->n_dof());

        std::string ns = "/" + hand_;

        sub_planned_ = create_subscription<JointState>(
            ns + "/ik_solution", 10,
            [this](JointState::SharedPtr m){ on_planned(m); });

        sub_actual_ = create_subscription<JointState>(
            ns + "/joint_states", 10,
            [this](JointState::SharedPtr m){ on_actual(m); });

        pub_contact_detected_ = create_publisher<BoolMsg>(
            ns + "/contact_detected", 10);
        pub_wrench_ = create_publisher<WrenchStamped>(
            ns + "/contact_force_est", 10);

        // Run observer at 100 Hz (10 ms)
        timer_ = create_wall_timer(10ms, [this](){ tick(); });

        RCLCPP_INFO(get_logger(),
            "VirtualSensorNode ready — hand: %s, DOFs: %d",
            hand_.c_str(), tree_->n_dof());
    }

private:
    void load_urdf() {
        if (!urdf_path_.empty()) {
            tree_ = std::make_unique<KinematicTree>(
                URDFLoader::from_file(urdf_path_));
            // Populate ContactWorld from URDF <collision> elements
            load_collision_geometry(urdf_path_, *tree_, contact_world_);
            // Ground plane at y=0 (always present)
            contact_world_.add_static_plane(Vec3::UnitY(), 0.0);
            RCLCPP_INFO(get_logger(), "Loaded URDF + collision geometry");
        } else {
            throw std::runtime_error(
                "VirtualSensorNode: urdf_path param required");
        }
    }

    void on_planned(const JointState::SharedPtr m) {
        fill_q(m, q_planned_);
    }

    void on_actual(const JointState::SharedPtr m) {
        auto now = this->now();
        if (last_actual_stamp_.nanoseconds() > 0) {
            double dt = (now - last_actual_stamp_).seconds();
            if (dt > 0 && dt < 0.5) {
                Eigen::VectorXd q_new = q_actual_;
                fill_q(m, q_new);
                dq_ = (q_new - q_actual_) / dt;
                q_actual_ = q_new;
            }
        } else {
            fill_q(m, q_actual_);
        }
        last_actual_stamp_ = now;
    }

    void tick() {
        // ── Step 1: detect joint blocking (planned vs actual) ─────────────────
        Eigen::VectorXd delta = q_actual_ - q_planned_;
        bool any_blocked = delta.cwiseAbs().maxCoeff() > block_thr_;

        // ── Step 2: run FK on actual joint state ──────────────────────────────
        tree_->fk(q_actual_);

        // ── Step 3: detect contacts using sim model ───────────────────────────
        auto contacts = contact_world_.detect(*tree_, 0.0);

        // ── Step 4: soft contact resolution → virtual force estimates ─────────
        Eigen::VectorXd tau_zero = Eigen::VectorXd::Zero(tree_->n_dof());
        Eigen::VectorXd lambda_Jc = contact_world_.resolve_soft(
            contacts, *tree_, q_actual_, dq_, kn_, kd_);

        // ── Step 5: virtual sensor readings ───────────────────────────────────
        auto readings = contact_world_.virtual_sensor_readings(
            contacts, *tree_, lambda_Jc, q_actual_);

        // ── Step 6: publish ───────────────────────────────────────────────────
        auto stamp = this->now();

        BoolMsg bm;
        bm.data = any_blocked || !contacts.empty();
        pub_contact_detected_->publish(bm);

        // Publish net estimated wrench (sum over all contacts)
        WrenchStamped ws;
        ws.header.stamp    = stamp;
        ws.header.frame_id = hand_ + "_palm";
        for (const auto& r : readings) {
            ws.wrench.force.x += r.force_world.x();
            ws.wrench.force.y += r.force_world.y();
            ws.wrench.force.z += r.force_world.z();
        }
        pub_wrench_->publish(ws);

        // Log at 1 Hz
        static int tick_count = 0;
        if (++tick_count % 100 == 0) {
            RCLCPP_DEBUG(get_logger(),
                "contacts: %zu  blocked: %s  max_delta: %.3f rad",
                contacts.size(),
                any_blocked ? "YES" : "no",
                delta.cwiseAbs().maxCoeff());
        }
    }

    void fill_q(const JointState::SharedPtr m, Eigen::VectorXd& q) {
        const auto& names = tree_->dof_names();
        for (size_t i = 0; i < m->name.size(); ++i)
            for (int d = 0; d < (int)names.size(); ++d)
                if (m->name[i] == names[d])
                    { q[d] = m->position[i]; break; }
    }

    std::string  hand_, urdf_path_;
    double kn_, kd_, block_thr_;
    Vec3   gravity_;

    std::unique_ptr<KinematicTree> tree_;
    ContactWorld                   contact_world_;

    Eigen::VectorXd q_planned_, q_actual_, dq_;
    rclcpp::Time    last_actual_stamp_;

    rclcpp::Subscription<JointState>::SharedPtr sub_planned_, sub_actual_;
    rclcpp::Publisher<BoolMsg>::SharedPtr       pub_contact_detected_;
    rclcpp::Publisher<WrenchStamped>::SharedPtr pub_wrench_;
    rclcpp::TimerBase::SharedPtr               timer_;
};

} // namespace lgn

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<lgn::VirtualSensorNode>());
    rclcpp::shutdown();
    return 0;
}
