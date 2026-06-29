from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────────
    hand_arg = DeclareLaunchArgument(
        'hand',
        default_value='right',
        description='Which hand: right or left')

    urdf_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value='',
        description='Absolute path to hand URDF (must include <collision> elements)')

    kn_arg = DeclareLaunchArgument(
        'contact_kn',
        default_value='50000.0',
        description='Soft contact spring stiffness [N/m]')

    kd_arg = DeclareLaunchArgument(
        'contact_kd',
        default_value='100.0',
        description='Soft contact damping [Ns/m]')

    thr_arg = DeclareLaunchArgument(
        'block_threshold',
        default_value='0.05',
        description='Joint blocking detection threshold [rad]')

    gx_arg = DeclareLaunchArgument('gravity_x', default_value='0.0')
    gy_arg = DeclareLaunchArgument('gravity_y', default_value='-9.81')
    gz_arg = DeclareLaunchArgument('gravity_z', default_value='0.0')

    # ── Virtual sensor node ───────────────────────────────────────────────────
    vs_node = Node(
        package='lgn_hand_ik',
        executable='virtual_sensor_node',
        name=['virtual_sensor_', LaunchConfiguration('hand')],
        output='screen',
        parameters=[{
            'hand':            LaunchConfiguration('hand'),
            'urdf_path':       LaunchConfiguration('urdf_path'),
            'contact_kn':      LaunchConfiguration('contact_kn'),
            'contact_kd':      LaunchConfiguration('contact_kd'),
            'block_threshold': LaunchConfiguration('block_threshold'),
            'gravity_x':       LaunchConfiguration('gravity_x'),
            'gravity_y':       LaunchConfiguration('gravity_y'),
            'gravity_z':       LaunchConfiguration('gravity_z'),
        }]
    )

    return LaunchDescription([
        hand_arg, urdf_arg,
        kn_arg, kd_arg, thr_arg,
        gx_arg, gy_arg, gz_arg,
        vs_node,
    ])

