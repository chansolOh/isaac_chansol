# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks

from omni.isaac.core.utils.stage import add_reference_to_stage

from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.prims import SingleArticulation

from isaacsim.robot.manipulators.grippers import ParallelGripper
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.core.prims import XFormPrim
import omni.usd
from pxr import UsdGeom, Gf
import torch
from isaacsim.core.utils.types import ArticulationAction

from .robot_configs import RobotConfig
from ..general_utils import mat_utils

# Inheriting from the base class Follow Target

def dpi_arr(deg_arr, target_deg_arr):
    result_arr = target_deg_arr.copy()
    over_idx = (deg_arr - target_deg_arr) >180
    result_arr[over_idx] +=360
    under_idx = (deg_arr - target_deg_arr) <-180
    result_arr[under_idx] -=360
    return result_arr

class My_Robot_Task(tasks.BaseTask):
    def __init__(
        self,
        robot_config: RobotConfig,
        name: str = "My_Robot",

        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, offset=offset)

        self.cfg = robot_config
        for k, v in vars(self.cfg).items():
            setattr(self, k, v)

        self.stage = omni.usd.get_context().get_stage()
    

        self.joint_states = None
        return

    
    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)


        self._robot = self.set_robot()
        self.kinematics_solver = self.set_solver()
        self.joint_names = self.kinematics_solver.get_joint_names()
        self.robot_prim = self.stage.GetPrimAtPath(self.prim_path)
        

        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return
    
    def set_robot(self) :
        add_reference_to_stage(usd_path=self.asset_path, prim_path=self.prim_path)
        
        if self.gripper_joint_prim_names==None:
            manipulator = SingleArticulation(
                prim_path=self.prim_path,
                name=self.name,
                position = self.robot_pos,
                orientation = self.robot_ori,
                scale = self.robot_scale,
            )

        else:
            gripper = ParallelGripper(
                end_effector_prim_path=os.path.join(self.prim_path, self.ee_link_path),
                joint_prim_names=self.gripper_joint_prim_names,
                joint_opened_positions=self.joint_opened_positions,
                joint_closed_positions=self.joint_closed_positions,
                action_deltas=np.array([-0., 0.]),
            )

            manipulator = SingleManipulator(
                prim_path=self.prim_path,
                name=self.name,
                end_effector_prim_path=os.path.join(self.prim_path, self.ee_link_path),
                gripper=gripper,
                position = self.robot_pos ,
                orientation = self.robot_ori ,
                scale = self.robot_scale ,
            )
 
        joints_default_positions = np.zeros(self.total_joint_num)
        manipulator.set_joints_default_state(positions=torch.tensor(joints_default_positions, dtype=torch.float32))

        return manipulator


    def set_world_pose(self, position, rotation):
        if self.robot_prim.HasAttribute("xformOp:translate"):
            self.robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(position))
        return
    

    @property
    def get_robot_name(self):
        return self._robot.name

    def set_solver(self):
        kinematics_solver = LulaKinematicsSolver(
                            robot_description_path=self.description_path,
                            urdf_path = self.urdf_path
        )
        return kinematics_solver

    def compute_ik(self,
        target_position : Optional[list],
        target_orientation : Optional[list], 
        frame_name : str = None,
        warm_start : np.ndarray = np.array([0.3,0.3,0.3,0.3,0.3,0.3])
    ):
        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        if type(target_orientation) == list:
            target_orientation = np.array(target_orientation)
        ik = self.kinematics_solver.compute_inverse_kinematics(
            frame_name = frame_name,
            target_position = target_position ,
            target_orientation = euler_angles_to_quat( target_orientation/180*np.pi),
            warm_start=warm_start
            )
        return ik
    
    def compute_ik_traj(self,
        target_position : Optional[list],
        target_orientation : Optional[list], 
        frame_name : str ,
        # warm_start : np.ndarray,
        return_traj: bool = False
        ):

        init_pos, init_rot = self.compute_fk(
            frame_name = frame_name,
        )
        warm_start = self.get_joint_positions()

        target_orientation = dpi_arr(init_rot, np.array(target_orientation))

        pos_dist = np.linalg.norm(target_position - init_pos)
        pos_step = 0.01  # 1cm
        pos_num = int(np.ceil(pos_dist / pos_step)) + 1
        rot_dist = np.linalg.norm(target_orientation - init_rot)
        rot_step = 1.0  # 1 degree
        rot_num = int(np.ceil(rot_dist / rot_step)) + 1
        num = max(pos_num, rot_num)
        points = np.linspace(init_pos, target_position, num=num)
        degs = np.linspace(init_rot, target_orientation, num=num)

        # import pdb; pdb.set_trace()
        joint_position_traj = []

        for i in range(num):
            target_position = points[i]
            target_orientation = degs[i]
            joint_positions, _ = self.compute_ik(
                target_position = target_position,
                target_orientation = target_orientation,
                frame_name = frame_name,
                warm_start = warm_start)
            joint_positions = dpi_arr(warm_start/np.pi*180, joint_positions/np.pi*180)/180*np.pi
            warm_start = joint_positions
            joint_position_traj.append(joint_positions)
        # import matplotlib.pyplot as plt
        # plt.plot(joint_position_traj)
        # plt.show()
        if return_traj:
            return np.array(joint_position_traj)
        
        return joint_positions


    def compute_fk(self, 
        frame_name:str, 
        ):

        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        pos, rot_mat = self.kinematics_solver.compute_forward_kinematics(frame_name =frame_name, joint_positions = self.get_joint_positions())

        rot = mat_utils.mat_to_euler(rot_mat)
        return pos, rot
    

    
    def get_joint_positions(self):
        return self._robot.get_joint_positions()[:len(self.joint_names)]

    def pre_step(self, current_time_step_index, current_time):
        return
    
    def post_reset(self):
        
        return
    