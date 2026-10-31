import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class StereoCameraNode(Node):

    def __init__(self):
        super().__init__('stereo_camera_node')

        self.declare_parameter('device', '/dev/video0')
        device = self.get_parameter('device').value

        self.bridge = CvBridge()

        self.left_pub = self.create_publisher(
            Image, '/stereo/left/image_raw', 10
        )

        self.right_pub = self.create_publisher(
            Image, '/stereo/right/image_raw', 10
        )

        self.combined_pub = self.create_publisher(
            Image, '/stereo/combined/image_raw', 10
        )

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG')
        )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(
                f'Could not open camera: {device}'
            )

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.get_logger().info(
            f'Camera opened: {width} x {height}'
        )

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.capture_frame
        )

    def capture_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning(
                'Failed to capture frame'
            )
            return

        height, width = frame.shape[:2]

        if width % 2 != 0:
            self.get_logger().error(
                'Stereo frame width must be even'
            )
            return

        midpoint = width // 2

        left = frame[:, :midpoint]
        right = frame[:, midpoint:]

        stamp = self.get_clock().now().to_msg()

        left_msg = self.bridge.cv2_to_imgmsg(
            left, encoding='bgr8'
        )
        left_msg.header.stamp = stamp
        left_msg.header.frame_id = 'stereo_left'

        right_msg = self.bridge.cv2_to_imgmsg(
            right, encoding='bgr8'
        )
        right_msg.header.stamp = stamp
        right_msg.header.frame_id = 'stereo_right'

        combined_msg = self.bridge.cv2_to_imgmsg(
            frame, encoding='bgr8'
        )
        combined_msg.header.stamp = stamp
        combined_msg.header.frame_id = 'stereo_combined'

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.combined_pub.publish(combined_msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = StereoCameraNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()