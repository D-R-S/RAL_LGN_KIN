from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────────
    hand_arg = DeclareLaunchArgument(
        'hand',
        default_value='right',
        description='Which hand to run: right or left')

    urdf_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value='',
        description='Absolute path to hand URDF file. '
                    'If empty, falls back to /robot_state_publisher '
                    'robot_description parameter.')

    sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='If true, also publish to /sim/joint_commands')

    diff_arg = DeclareLaunchArgument(
        'use_differential',
        default_value='true',
        description='Enable differential B-matrix servo mapping '
                    '(set false for non-Amazing-Hand URDFs)')

    lambda_arg = DeclareLaunchArgument(
        'lambda_sq',
        default_value='0.0001',
        description='DLS damping lambda squared')

    iter_arg = DeclareLaunchArgument(
        'max_iter',
        default_value='50',
        description='Maximum IK iterations per solve')

    pos_tol_arg = DeclareLaunchArgument(
        'pos_tol',
        default_value='0.0001',
        description='Position convergence tolerance [m]')

    rot_tol_arg = DeclareLaunchArgument(
        'rot_tol',
        default_value='0.001',
        description='Rotation convergence tolerance [rad]')

    # ── IK node ───────────────────────────────────────────────────────────────
    ik_node = Node(
        package='lgn_hand_ik',
        executable='hand_ik_node',
        name=['hand_ik_', LaunchConfiguration('hand')],
        output='screen',
        parameters=[{
            'hand':             LaunchConfiguration('hand'),
            'urdf_path':        LaunchConfiguration('urdf_path'),
            'use_sim':          LaunchConfiguration('use_sim'),
            'use_differential': LaunchConfiguration('use_differential'),
            'lambda_sq':        LaunchConfiguration('lambda_sq'),
            'max_iter':         LaunchConfiguration('max_iter'),
            'pos_tol':          LaunchConfiguration('pos_tol'),
            'rot_tol':          LaunchConfiguration('rot_tol'),
        }]
    )

    return LaunchDescription([
        hand_arg,
        urdf_arg,
        sim_arg,
        diff_arg,
        lambda_arg,
        iter_arg,
        pos_tol_arg,
        rot_tol_arg,
        ik_node,
    ])

