# import os
# ISAACSIM_BIN = "/home/uon/ochansol/isaac_code/isaac_chansol/.venv/bin/isaacsim"
# JAZZY_LIB = "/home/uon/ochansol/isaac_code/isaac_chansol/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"

# env = os.environ.copy()

# # ROS2 Bridge용 환경
# env["ROS_DISTRO"] = "jazzy"
# env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

# # LD_LIBRARY_PATH 안전하게 추가
# env["LD_LIBRARY_PATH"] = (
#     f"{env['LD_LIBRARY_PATH']}:{JAZZY_LIB}"
#     if "LD_LIBRARY_PATH" in env
#     else JAZZY_LIB
# )

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
from isaacsim.core.api import World
import numpy as np


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.Robot_45.robot_policy import My_Robot_Task as Robot_task
from Utils.Robot_45.robot_configs import ROBOT_CONFIGS

# from isaacsim.core.utils import extensions
# extensions.enable_extension("isaacsim.ros2.bridge")


import rclpy
from rclpy.node import Node
from ik_isaac_interfaces.srv import IkSolve



class IK_isaac:
    def __init__(self):


        self.my_world = World(stage_units_in_meters=1.0,
                        physics_dt  = 0.01,
                        rendering_dt = 0.01)

        self.Robot_Cfg = ROBOT_CONFIGS["Doosan_M1013"]()
        self.my_robot_task = Robot_task(robot_config=self.Robot_Cfg, name="robot_task" )
        self.my_world.add_task(self.my_robot_task)
        self.my_world.reset()

        self.my_robot = self.my_robot_task._robot


    def get_ik(self, target_pos = np.array([0.3,0.3,0.3]), target_ori = np.array([0,0,0]), return_traj=False):
        if type(target_pos) == type([]):
            target_pos = np.array(target_pos)
        if type(target_ori) == type([]):
            target_ori = np.array(target_ori)


        target_joint_positions = self.my_robot_task.compute_ik_traj(target_position = target_pos,
                                            target_orientation = np.array([180,0,0]) + target_ori,
                                            frame_name = "Robotiq_2f140_open",
                                            return_traj=False
                                            )
        deg = np.round(np.array(target_joint_positions)/np.pi*180,3)

        return deg
    




class IkServer(Node):
    def __init__(self):
        super().__init__('ik_server')
        self.srv = self.create_service(IkSolve, 'isaac_ik_server', self.callback)
        self.get_logger().info('IK Server ready')

    def callback(self, req, res):
        joint_result = self.ik_isaac.get_ik(target_pos = [req.position.x, req.position.y, req.position.z],
                             target_ori = [req.rotation.x, req.rotation.y, req.rotation.z],
                             return_traj = req.return_traj
                             )
        res.joint_result = joint_result
        return res
    

        
def main():
    rclpy.init()
    node = IkServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
