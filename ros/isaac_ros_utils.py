import omni
import omni.syntheticdata._syntheticdata as sd
import omni.replicator.core as rep
import omni.graph.core as og
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils import extensions
extensions.enable_extension("isaacsim.ros2.bridge")
import omni.kit.app
omni.kit.app.get_app().update()

import numpy as np



def publish_rgb(camera: Camera, freq):
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return

def add_camera_to_ros(camera_prim_path: str, name :str , freq: float = 20, resolution:tuple = (224,224) ) -> Camera:
    """Adds a camera to ROS2 publishing.

    Args:
        camera_prim_path (str): [description]
        freq (float): [description]

    Returns:
        Camera: [description]
    """
    camera = Camera(
        prim_path=camera_prim_path,
        name=name,
        frequency=freq,
        resolution=resolution,)
    camera.initialize()
    publish_rgb(camera, freq)
    return camera





##### 너무 느려서 포기
# class VLAClient(Node):
#     def __init__(self):
#         super().__init__('vla_client')
#         self.cli = self.create_client(VLAInference, 'vla_inference')
#         self.result = None

#         while not self.cli.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting...')

#     def send_request(self, full_img, wrist_img, joints, desc):
#         ### 그나마 cv2 로 변환해서 전송하는게 빠름
#         _,encoded = cv2.imencode('.jpg', full_img)
#         full_msg = Image()
#         full_msg.data = encoded.tobytes()

        #### 정석방법은 진짜 개느림
        # full_msg = Image()
        # full_msg.height = full_img.shape[0]
        # full_msg.width  = full_img.shape[1]
        # full_msg.encoding = "rgb8"
        # full_msg.data = memoryview(full_img.reshape(-1))

        # # --- wrist_image ---
        # wrist_msg = Image()
        # wrist_msg.height = wrist_img.shape[0]
        # wrist_msg.width  = wrist_img.shape[1]
        # wrist_msg.encoding = "rgb8"
        # wrist_msg.data = wrist_img.tobytes()


        # req = VLAInference.Request()
        # req.full_image = full_msg
        # req.wrist_image = wrist_msg
        # req.joints = joints
        # req.description = desc
        # self.result = self.cli.call_async(req)
        # print(self.result)
        # return 



# action = np.zeros((8,7))
# def sub_result(msg):
#     global action
#     action_tmp = np.array(msg.data).reshape((8,7))

#     if np.any(action != action_tmp):
#         action = action_tmp



# rclpy.init()
# state_pub_node = rclpy.create_node('state_pub')
# state_pub = state_pub_node.create_publisher(Float32MultiArray, '/state', 10)
# task_pub_node = rclpy.create_node('task_pub')
# task_pub = task_pub_node.create_publisher(String, '/task_desc', 10)

# action_sub_node = rclpy.create_node('action_sub')
# action_sub = action_sub_node.create_subscription(
#             ActionMatrix, '/action_matrix', sub_result, 10)


# executor = MultiThreadedExecutor()
# executor.add_node(state_pub_node)
# executor.add_node(task_pub_node)
# executor.add_node(action_sub_node)

# thread = threading.Thread(target=executor.spin, daemon=True)
# thread.start()

# task = String(data='pick up the black bowl between the plate and the ramekin and place it on the plate')
# task = String(data='pick up the plate')


