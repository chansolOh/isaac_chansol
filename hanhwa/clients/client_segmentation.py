import sys
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from detr_msg.srv import DetrSrv




class DetrClient(Node):
    def __init__(self):
        super().__init__('detr_client')
        self.cli = self.create_client(DetrSrv, 'detr_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /detr_service ...')
        self.cv_bridge = CvBridge()

    def send(self, img, timestamp):
        req = DetrSrv.Request()
        img_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="bgr8")
        req.input_image = img_msg
        req.timestamp = timestamp

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None:
            raise RuntimeError("service send failed (no response)")
        return res.response


def main():
    rclpy.init()
    node = DetrClient()

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    try:
        out = node.send(image_path)
        print(out)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
