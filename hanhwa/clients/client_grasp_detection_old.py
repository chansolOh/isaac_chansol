import os
import numpy as np
import cv2
import math

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from grasping_test_uon_interfaces.srv  import HanhwaDemo

os.environ["ROS_DOMAIN_ID"] = "19"
os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

class HRDemoGraspPredClient(Node):
    def __init__(self):
        super().__init__('hr_demo_grasp_client')
        
        self.bridge = CvBridge()
        
        self.cli = self.create_client(HanhwaDemo, 'hr_demo_grasp_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for grasping_predict service...')
        self.req = HanhwaDemo.Request()

    def send(self, gripper_type:str, img_rgb:np.ndarray, img_depth:np.ndarray, img_mask:np.ndarray, centroids:list):
        img_rgb_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
        img_depth_msg = self.bridge.cv2_to_imgmsg(img_depth, encoding="32FC1")
        img_mask_msg = self.bridge.cv2_to_imgmsg(img_mask, encoding="mono8")
        
        self.req.gripper_type = gripper_type
        self.req.img_rgb = img_rgb_msg
        self.req.img_depth = img_depth_msg
        self.req.img_mask = img_mask_msg
        self.req.centroids = centroids
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        
        result = future.result()
        
        centroid = result.centroid
        grasp_angle_deg = result.angle_deg
        grasp_width = result.width
        display_grasp_msg = result.display
        display_grasp = self.bridge.imgmsg_to_cv2(display_grasp_msg, desired_encoding="rgb8")
        
        
        return centroid, grasp_angle_deg, grasp_width, display_grasp
    
    
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
    ]
    
    
    
    
def main():
    rclpy.init()
    node = HRDemoGraspPredClient()
    
    # tmp image ======================================
    h, w = 64, 64
    rgb = np.random.randint(0, 256, (h, w, 4), np.uint8)  # BGRA
    img_rgb_msg_tmp = Image()
    img_rgb_msg_tmp.height = h
    img_rgb_msg_tmp.width  = w
    img_rgb_msg_tmp.encoding = "bgra8"
    img_rgb_msg_tmp.step = w * 4
    img_rgb_msg_tmp.data = rgb.tobytes()
    
    depth = np.random.randint(0, 5000, (h, w), np.uint16)
    img_depth_msg_tmp = Image()
    img_depth_msg_tmp.height = h
    img_depth_msg_tmp.width  = w
    img_depth_msg_tmp.encoding = "16UC1"
    img_depth_msg_tmp.step = w * 2
    img_depth_msg_tmp.data = depth.tobytes()
    # ===========================================

    # send
    grasp_angle_deg, grasp_width, display_grasp = node.send(gripper_type="ROBOTIS_RH-P12-RN", img_rgb=img_rgb_msg_tmp, img_depth=img_depth_msg_tmp, centroids=[50, 50])
    # gripper_type = "ROBOTIS_RH-P12-RN" or "DH3"

    # post processing
    grasp_angle_rad = np.deg2rad(grasp_angle_deg)
    
    orientation_target = [0., math.sin(grasp_angle_rad / 2.0), 0., math.cos(grasp_angle_rad / 2.0)] # qx, qy, qz, qw
    orientation_base = [0.5, 0.5, 0.5, 0.5]
    orientation = quat_mul(orientation_base, orientation_target) # qx, qy, qz, qw

    print("=====grasp predict results=====")
    print(f"  angle   : {grasp_angle_deg:.2f} deg   ({grasp_angle_rad:.4f} rad)")
    print(f"  width   : {grasp_width} / ???")
    
    # visualization
    cv2.imshow("display", display_grasp)
        
    print("[INFO] press 'q' to close windows...")
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()