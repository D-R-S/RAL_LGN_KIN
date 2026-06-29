"""
launch/full_hand.launch.py

Convenience launcher: starts both hand_ik_node AND virtual_sensor_node
for one hand in a single launch command.

Usage:
    ros2 launch lgn_hand_ik full_hand.launch.py \
        hand:=right \
        urdf_path:=/path/to/amazing_hand_right.urdf

    ros2 launch lgn_hand_ik full_hand.launch.py \
        hand:=left \
        urdf_path:=/path/to/amazing_hand_left.urdf
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    hand_arg = DeclareLaunchArgument(
        'hand', default_value='right',
        description='Which hand: right or left')

    urdf_arg = DeclareLaunchArgument(
        'urdf_path', default_value='',
        description='Absolute path to hand URDF')

    sim_arg = DeclareLaunchArgument(
        'use_sim', default_value='false',
        description='Mirror IK solution to /sim/joint_commands')

    pkg_share = FindPackageShare('lgn_hand_ik')

    ik_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'hand_ik.launch.py'])),
        launch_arguments={
            'hand':      LaunchConfiguration('hand'),
            'urdf_path': LaunchConfiguration('urdf_path'),
            'use_sim':   LaunchConfiguration('use_sim'),
        }.items()
    )

    vs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'virtual_sensor.launch.py'])),
        launch_arguments={
            'hand':      LaunchConfiguration('hand'),
            'urdf_path': LaunchConfiguration('urdf_path'),
        }.items()
    )

    return LaunchDescription([
        hand_arg, urdf_arg, sim_arg,
        ik_launch,
        vs_launch,
    ])

