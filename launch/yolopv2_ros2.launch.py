import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        # YOLOPv2 の推論ノード
        launch_ros.actions.Node(
            package='yolopv2_ros2',
            executable='yolopv2_inference_node',
            name='yolopv2_inference',
            output='screen'
        ),
        
        # 俯瞰変換ノード
        launch_ros.actions.Node(
            package='yolopv2_ros2',
            executable='bird_eye_view_node',
            name='bird_eye_view',
            output='screen'
        ),

        # パスパブリッシャーノード
        launch_ros.actions.Node(
            package='yolopv2_ros2',
            executable='path_publisher_node',
            name='path_publisher',
            output='screen'
        ),

        # パス追従ノード
        launch_ros.actions.Node(
            package='yolopv2_ros2',
            executable='follow_node',
            name='follow_node',
            output='screen'
        )
    ])
