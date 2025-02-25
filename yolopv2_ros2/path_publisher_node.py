import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from scipy.interpolate import interp1d, CubicSpline


class PathPublisherNode(Node):
    def __init__(self):
        super().__init__('path_publisher_node')

        # サブスクライブ
        self.subscription_pc = self.create_subscription(
            PointCloud2,
            '/yolopv2/pointcloud2/bird_eye_view',
            self.pc_callback,
            10)

        self.subscription_pose = self.create_subscription(
            PoseStamped,
            '/pose',
            self.pose_callback,
            10)

        # パブリッシュ
        self.publisher_path = self.create_publisher(Path, '/path', 10)

        self.current_pose = None  # 最新のロボット位置
        self.latest_pc_msg = None  # 最新のPointCloud2メッセージを保存

        self.get_logger().info('PathPublisherNode initialized.')

    def pose_callback(self, msg):
        """ ロボットの現在位置を取得 """
        self.current_pose = msg.pose
        if self.latest_pc_msg:
            self.process_pointcloud(self.latest_pc_msg)  # 保存されたPointCloud2を処理
            self.latest_pc_msg = None  # 処理後にクリア

    def pc_callback(self, msg):
        """ PointCloud2 を受信して経路を生成 """
        if self.current_pose is None:
            self.get_logger().warn('Pose not received yet. Storing latest PointCloud2 message.')
            self.latest_pc_msg = msg  # 最新のPointCloud2を保存
            return
        
        self.process_pointcloud(msg)  # 直接処理

    def process_pointcloud(self, msg):
        """ PointCloud2 メッセージを処理して、指定された x のポイントで補間し、経路を生成 """
        if self.current_pose is None:
            self.get_logger().warn('Pose not available. Skipping path generation.')
            return

        # PointCloud2 から (x, y) 座標を取得
        points = list(pc2.read_points(msg, field_names=("x", "y"), skip_nans=True))
        self.get_logger().info(f'Received {len(points)} points from PointCloud2')

        if not points:
            self.get_logger().warn('No points received in PointCloud2.')
            return

        # x を 0, 3, 5 のポイントに制限
        target_x_vals = [1, 2, 3, 4, 5]
        x_to_y = {x: [] for x in target_x_vals}

        for x, y in points:
            if x in x_to_y:
                x_to_y[x].append(y)

        # 各 x の y の中央値を取得
        waypoints = []
        for x in target_x_vals:
            if x_to_y[x]:
                y_median = np.median(x_to_y[x])
                waypoints.append((x, y_median))

        # x でソート
        waypoints.sort()

        if len(waypoints) < 2:
            self.get_logger().warn('Insufficient path points for interpolation. Skipping path update.')
            return

        # スプライン補間
        x_vals, y_vals = zip(*waypoints)
        spline = CubicSpline(x_vals, y_vals, bc_type='natural')

        x_fine = np.linspace(min(x_vals), max(x_vals), num=50)  # 補間点を50点に減らす
        y_fine = spline(x_fine)

        # Path メッセージ作成
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        start_pose = PoseStamped()
        start_pose.header = path_msg.header
        start_pose.pose.position.x = 0.0
        start_pose.pose.position.y = 0.0
        start_pose.pose.position.z = 0.0
        start_pose.pose.orientation.w = 1.0
        path_msg.poses.append(start_pose)

        for x, y in zip(x_fine, y_fine):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.publisher_path.publish(path_msg)
        self.get_logger().info(f'Published Path with {len(path_msg.poses)} points.')


def main(args=None):
    rclpy.init(args=args)
    node = PathPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()