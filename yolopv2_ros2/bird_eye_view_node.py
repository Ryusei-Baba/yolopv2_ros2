import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge
import numpy as np
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped


class BirdEyeViewNode(Node):
    def __init__(self):
        super().__init__('bird_eye_view_node')

        self.bridge = CvBridge()
        self.subscription_ll = self.create_subscription(Image, '/yolopv2/image/ll_seg_mask', self.ll_callback, 10)
        self.publisher_bev = self.create_publisher(PointCloud2, '/yolopv2/pointcloud2/bird_eye_view', 10)
        self.publisher_pose = self.create_publisher(PoseStamped, '/pose', 10)  # ★Poseのパブリッシャー追加★

        # 画像 → ロボット座標系変換のパラメータ
        self.src_pts = np.float32([[100, 180], [380, 180], [-480, 300], [960, 300]])
        self.dst_pts = np.float32([[3.0, 2.5], [3.0, -2.5], [-1.5, 2.5], [-1.5, -2.5]])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        self.get_logger().info('BirdEyeViewNode initialized.')

        # タイマーで定期的にPoseを送信
        self.timer = self.create_timer(0.1, self.publish_pose)  # 0.1秒ごとに実行

        # 仮のロボット位置（テスト用）
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.get_logger().info('BirdEyeViewNode initialized.')

    def ll_callback(self, msg):
        self.get_logger().debug('Received image message.')
        ll_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # **赤色の抽出**
        red_channel = ll_image[:, :, 2]
        _, binary_mask = cv2.threshold(red_channel, 150, 255, cv2.THRESH_BINARY)

        # **赤いピクセルの座標取得**
        points = cv2.findNonZero(binary_mask)
        if points is None:
            self.get_logger().warn('No red regions detected. Skipping processing.')
            return

        points = points.reshape(-1, 2)

        # **ピクセル座標をロボット座標系に変換**
        pixel_coords = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
        transformed = cv2.perspectiveTransform(pixel_coords[:, :2].reshape(-1, 1, 2), self.M)

        # **PointCloud2 フィルタ処理**
        transformed_dict = {}

        for p in transformed:
            x, y = p[0][0], p[0][1]
            x = round(x)  # ★ x を整数に変換（四捨五入）

            if x not in transformed_dict:
                transformed_dict[x] = {'pos': None, 'neg': None}

            # **y >= 0 の場合、y が最も 0 に近いものを保存**
            if y >= 0:
                if transformed_dict[x]['pos'] is None or abs(y) < abs(transformed_dict[x]['pos'][1]):
                    transformed_dict[x]['pos'] = (x, y, 0.0)

            # **y < 0 の場合、y が最も 0 に近いものを保存**
            else:
                if transformed_dict[x]['neg'] is None or abs(y) < abs(transformed_dict[x]['neg'][1]):
                    transformed_dict[x]['neg'] = (x, y, 0.0)

        # **フィルタされたポイントをリスト化**
        filtered_points = []
        for x in transformed_dict:
            if transformed_dict[x]['pos']:  # y > 0 の代表点
                filtered_points.append(transformed_dict[x]['pos'])
            if transformed_dict[x]['neg']:  # y < 0 の代表点
                filtered_points.append(transformed_dict[x]['neg'])


        # **PointCloud2 メッセージ作成**
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        cloud_msg = pc2.create_cloud_xyz32(header, filtered_points)

        # **パブリッシュ**
        self.publisher_bev.publish(cloud_msg)
        self.get_logger().info(f'Published filtered PointCloud2 with {len(filtered_points)} points.')

    def publish_pose(self):
        """ 仮のPoseをパブリッシュ """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = self.robot_x
        pose_msg.pose.position.y = self.robot_y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0  # 向きを仮に設定

        self.publisher_pose.publish(pose_msg)
        self.get_logger().info(f'Published Pose: x={self.robot_x}, y={self.robot_y}')


def main(args=None):
    rclpy.init(args=args)
    node = BirdEyeViewNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
