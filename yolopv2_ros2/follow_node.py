import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

class PIDControlNode(Node):
    def __init__(self):
        super().__init__('pid_control_node')

        # /path から経路を受信
        self.subscription = self.create_subscription(
            Path,
            '/path',
            self.path_callback,
            10)
        
        # /cmd_vel で速度指令をパブリッシュ
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 経路データ
        self.path_points = []

        # PID制御パラメータ
        self.kp = 1.0  # 比例ゲイン
        self.ki = 0.01  # 積分ゲイン
        self.kd = 0.1  # 微分ゲイン
        self.max_steering_angle = 1.0  # 最大ステアリング角

        # 速度設定
        self.linear_velocity = 8.0  # m/s（直進速度を固定）

        # PID制御用変数
        self.previous_error = 0.0
        self.integral_error = 0.0

        # タイマーで定期的に制御実行
        self.timer = self.create_timer(0.1, self.control_loop)

    def path_callback(self, msg):
        """
        /path トピックから受信した経路を処理
        """
        self.path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.get_logger().info(f"Received path with {len(self.path_points)} points.")

    def control_loop(self):
        """
        PID制御による角速度制御
        """
        if len(self.path_points) < 2:
            return  # 経路がない or 1点だけなら何もしない

        # 次の目標点を取得
        lookahead_index = min(5, len(self.path_points) - 1)  # 5つ先のポイントを目標に
        target_x, target_y = self.path_points[lookahead_index]

        # 現在のロボットの進行方向は (1,0)（x軸正方向）
        # 目標点への角度を求める
        target_angle = math.atan2(target_y, target_x)  # (0,0) から見た角度
        heading_error = self.normalize_angle(target_angle)  # -π ～ π に正規化

        # PID制御計算
        self.integral_error += heading_error * 0.1  # 積分項（dt=0.1秒）
        derivative_error = (heading_error - self.previous_error) / 0.1  # 微分項（dt=0.1秒）

        angular_velocity = (
            self.kp * heading_error +
            self.ki * self.integral_error +
            self.kd * derivative_error
        )
        angular_velocity = max(-self.max_steering_angle, min(self.max_steering_angle, angular_velocity))  # 制限

        # 誤差更新
        self.previous_error = heading_error

        # 速度コマンドを送信
        self.publish_cmd_vel(self.linear_velocity, angular_velocity)

    def normalize_angle(self, angle):
        """
        角度を -π ～ π の範囲に正規化
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def publish_cmd_vel(self, linear, angular):
        """
        /cmd_vel に速度指令をパブリッシュ
        """
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear
        cmd_vel_msg.angular.z = angular
        self.cmd_vel_publisher.publish(cmd_vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PIDControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()