import numpy as np
from isaacsim.core.api import World

from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, euler_to_rot_matrix, quat_to_rot_matrix, quat_to_euler_angles


import omni.isaac.core.prims as Prims
import omni.usd
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.types import ArticulationAction

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, '../python'))

# from .basic_ik.robot_task import Robot_Task
from .robot_policy import My_Robot_Task
from .robot_configs import ROBOT_CONFIGS

from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from ..general_utils import mat_utils


my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

class BasicIk:
    def __init__(self, chunk_size=8, action_size = 7):
        Robot_Cfg = ROBOT_CONFIGS["Robotis_OMY_VLA_Demo"]()
        self.robot_task = My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" )

        my_world.add_task(self.robot_task)
        my_world.reset()
 
        self.robot = self.robot_task._robot
        self.robot.set_world_pose(position = [0,0,0])

        self.action_chunk_arr = np.array([])
        self.chunk_size = chunk_size
        self.action_size = action_size
        self.action_scale = 0#0.005
        self.ensemble_size = 8
        self.ensemble_weights_scale = 1
        self.ensemble_weights = np.array([-self.ensemble_weights_scale*i for i in range(self.ensemble_size)])
        self.gripper_close_pos = np.array([0.0, 0.0])
        self.gripper_open_pos = self.robot_task.joint_closed_positions

        self.target_offset = np.array([0.5, 0.0, -0.374])

        self.actions = None
        self.set_scene()

    def set_scene(self):
        self.target_prim = VisualCuboid(
            prim_path="/World/target_prim",
            name="visual_cube",
            position=np.array([-0.211 , -0.011, 1.174]) + self.target_offset,
            orientation=euler_angles_to_quat(np.array([180,0,-180]), degrees=True),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.0, 1.0, 0.0])   # 초록
        )
        self.target_prim.set_visibility(False)

    def set_target_prim(self, action):
        ### target_position
        tp = action[:3] * self.action_scale
        tp = self.target_pos + np.array([tp[0], tp[1], tp[2]])
        ### target_orientation
        toq = mat_utils.quat_mul(self.target_ori,mat_utils.axis_angle_to_quat(action[3:6]))

        ## only yaw rotation
        # to = np.array([np.pi, 0, to[2]])

        self.apply_gripper_action(action[-1])
        print(np.round(action[-1],4))

        self.target_prim.set_world_pose(
            position =  tp, 
            orientation =toq
            )
    def joint_control(self, joint_positions):
        # self.apply_gripper_action(joint_positions[-1])

        self.actions = ArticulationAction(
                        joint_indices=[0,1,2,3,4,5,7] ,  ####  joint name으로 index 찾아오기
                        joint_positions = joint_positions)
      
        

    def apply_gripper_action(self, gripper_action):

        # action =  0.04 - ( (self.gripper_open_pos - self.gripper_close_pos) * (gripper_action +1)/2 + self.gripper_close_pos )
        # self.robot.apply_action(
        #     ArticulationAction(
        #         joint_indices=[self.robot_task.gripper_joint_index] , #### joint name으로 index 찾아오기
        #         joint_positions = action
        #     ) 
        # )
        action =  0.04 - ( (self.gripper_open_pos - self.gripper_close_pos) * (gripper_action +1)/2 + self.gripper_close_pos )
        self.robot.apply_action(
            ArticulationAction(
                joint_indices=[6,7] , #### joint name으로 index 찾아오기
                joint_positions = action
            ) 
        )

    def ee_pose_control(self):

        target_pos = self.target_prim.get_world_pose()[0] - self.robot.get_world_pose()[0]
        target_ori = self.target_prim.get_world_pose()[1]

        target_joint_positions,_ = self.robot_task.compute_ik(target_position = target_pos,
                            target_orientation =quat_to_euler_angles(target_ori, degrees=True), # x,y,z 순서로 회전
                            frame_name = self.robot_task.tcp_prim_name,
                            warm_start=self.robot.get_joint_positions()[self.robot_task.manipulator_joint_index]
                            )

        self.actions = ArticulationAction(
                                joint_indices=self.robot_task.manipulator_joint_index ,  ####  joint name으로 index 찾아오기
                                joint_positions = target_joint_positions)
        
    def action_step(self):
        if self.actions is not None:
            self.robot.apply_action(self.actions)


    def get_state(self, action_type="ee_pose"):
        if action_type == "ee_pose":
            self.target_pos, self.target_ori = self.target_prim.get_world_pose()
            self.target_pos_r = self.target_pos - self.target_offset
            self.target_ori_rpy = quat_to_euler_angles(self.target_ori, degrees=False) + np.array([np.pi*2,0,-np.pi])
            self.target_ori_quat = euler_angles_to_quat(self.target_ori_rpy, degrees=False)
            self.target_axis_angle = mat_utils.quat_to_axis_angle(self.target_ori_quat[[1,2,3,0]])


            gripper_state = self.robot.gripper.get_joint_positions()
            total = np.hstack((self.target_pos_r, self.target_axis_angle, gripper_state))
            state = total.tolist()
            return state
        elif action_type == "joint":
            joint_positions = self.robot_task.get_joint_positions()
            state = joint_positions.tolist()
            return state
    
    def set_action(self, action, action_type = "ee_pose", action_chunk = False):
        if action_chunk == False:
            if action_type == "ee_pose":
                self.set_target_prim(action)
                self.ee_pose_control()
            elif action_type == "joint":
                self.joint_control(action)
        else:
            if len(self.action_chunk_arr) == 0:
                self.action_chunk_arr = action[None,:]
            elif len(self.action_chunk_arr) < self.ensemble_size:
                self.action_chunk_arr = np.vstack((self.action_chunk_arr, action[None,:]))
            else:
                self.action_chunk_arr = np.vstack((self.action_chunk_arr[1:], action[None,:]))
                temporal_ensemble_action = self.ensemble_weights.dot( self.action_chunk_arr[[i for i in range(self.ensemble_size)],[-i-1 for i in range(self.ensemble_size)]] ) / self.ensemble_weights.sum()
                if action_type == "ee_pose":
                    self.set_target_prim(temporal_ensemble_action)
                    self.ee_pose_control()
                elif action_type == "joint":
                    self.joint_control(temporal_ensemble_action)

    def reset(self):
        self.robot_task.reset()
