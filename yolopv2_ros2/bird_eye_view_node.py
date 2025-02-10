import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class BirdEyeViewNode(Node):
    def __init__(self):
        super().__init__('bird_eye_view_node')

        # CvBridgeのインスタンス
        self.bridge = CvBridge()

        # 画像をサブスクライブ
        self.subscription_da = self.create_subscription(Image, '/yolopv2/image/da_seg_mask', self.da_callback, 10)
        self.subscription_ll = self.create_subscription(Image, '/yolopv2/image/ll_seg_mask', self.ll_callback, 10)

        # 俯瞰図のパブリッシャー
        self.publisher_bev = self.create_publisher(Image, '/yolopv2/image/bird_eye_view', 10)

        # 画像データのバッファ
        self.da_image = None
        self.ll_image = None

    def da_callback(self, msg):
        self.da_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_publish()

    def ll_callback(self, msg):
        self.ll_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_and_publish()

    def process_and_publish(self):
        if self.da_image is None or self.ll_image is None:
            return  # どちらかの画像がまだ受信されていない場合は処理しない

        # 画像サイズを取得
        height, width = self.da_image.shape[:2]

        # 画像を統合
        combined_image = cv2.addWeighted(self.da_image, 0.5, self.ll_image, 0.5, 0)


        # 俯瞰変換前の座標 (元の画像上の4点)
        src_pts = np.float32([
            [0, height*0.7],               # 左下
            [width * 0.3, height * 0.6],   # 左上
            [width * 0.7, height * 0.6],   # 右上
            [width, height*0.7]            # 右下
        ])

        # 俯瞰変換後の座標 (出力画像上の4点)
        dst_pts = np.float32([
            [width*0.4, height],   # 俯瞰後の左下
            [width * 0.4, 0],      # 俯瞰後の左上
            [width * 0.6, 0],      # 俯瞰後の右上
            [width*0.6, height]    # 俯瞰後の右下
        ])


        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 俯瞰変換を適用
        bev_image = cv2.warpPerspective(combined_image, M, (width, height))

        # 変換した画像をパブリッシュ
        bev_msg = self.bridge.cv2_to_imgmsg(bev_image, encoding='bgr8')
        self.publisher_bev.publish(bev_msg)
        self.get_logger().info('Published Bird Eye View Image')

def main(args=None):
    rclpy.init(args=args)
    node = BirdEyeViewNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()