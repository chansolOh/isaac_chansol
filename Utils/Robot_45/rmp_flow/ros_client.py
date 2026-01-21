import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from vla_msgs.msg import ActionMatrix
import numpy as np
import threading
import time

class VLAClient(Node):
    def __init__(self):
        super().__init__('vla_client')

        # Publish → inference 서버로 전달
        self.pub_full  = self.create_publisher(Image, '/full_image', 10)
        self.pub_wrist = self.create_publisher(Image, '/wrist_image', 10)
        self.pub_state = self.create_publisher(Float32MultiArray, '/state', 10)
        self.pub_task  = self.create_publisher(String, '/task_desc', 10)

        # Subscribe → 서버에서 보내는 결과 받기
        self.sub_result = self.create_subscription(
            ActionMatrix, '/action_matrix', self.result_cb, 10)
        
        self.shared_full  = None
        self.shared_wrist = None
        self.shared_state = None
        self.shared_task  = None

        # 결과 저장
        self.result_ready = False
        self.latest_result = None

        # threading.Thread(target=self._publish_loop, daemon=True).start()

    def update_inputs(self, full_np, wrist_np, state_list, task_text):
        self.shared_full  = full_np
        self.shared_wrist = wrist_np
        self.shared_state = [float(x) for x in state_list]  # float32 보장
        self.shared_task  = task_text

    def _publish_loop(self):
        """IsaacSim loop와 독립적으로 30Hz로 publish."""
        while True:
            if self.shared_full is not None:

                # --- full_image ---
                full_msg = Image()
                full_msg.height = self.shared_full.shape[0]
                full_msg.width  = self.shared_full.shape[1]
                full_msg.encoding = "rgb8"
                full_msg.data = self.shared_full.tobytes()

                # --- wrist_image ---
                wrist_msg = Image()
                wrist_msg.height = self.shared_wrist.shape[0]
                wrist_msg.width  = self.shared_wrist.shape[1]
                wrist_msg.encoding = "rgb8"
                wrist_msg.data = self.shared_wrist.tobytes()

                state_msg = Float32MultiArray(data=self.shared_state)
                task_msg = String(data=self.shared_task)

                self.pub_full.publish(full_msg)
                self.pub_wrist.publish(wrist_msg)
                self.pub_state.publish(state_msg)
                self.pub_task.publish(task_msg)
                time.sleep(0.1)


    def result_cb(self, msg):
        self.latest_result = msg.data
        self.result_ready = True
        self.get_logger().info("새 inference 결과 도착!")

def main():
    rclpy.init()
    node = VLAClient()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    import threading
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
