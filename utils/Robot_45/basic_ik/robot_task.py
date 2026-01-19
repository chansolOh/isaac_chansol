# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import json
from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks

from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
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


# Inheriting from the base class Follow Target
class Robot_Task(tasks.BaseTask):
    def __init__(
        self,
        config_dir_path : str,
        name: str = "Default_name",
        prim_path :str = "/World/Robot",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, offset=offset)
        self.prim_path = prim_path
        self.stage = omni.usd.get_context().get_stage()
    
        self.config_dir_path = config_dir_path  
        with open(os.path.join(self.config_dir_path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        # self.world = World(stage_units_in_meters=1.0)
        self.manipulator_joint_index = self.config["manipulator_config"]["joint_index"]
        self.gripper_joint_index = self.config["gripper_config"]["joint_index"]

        return
    
    def set_up_scene(self, scene: Scene) -> None:

        super().set_up_scene(scene)

        self._robot = self.set_robot()
        self.kinematics_solver = self.set_solver()
        self.robot_prim = self.stage.GetPrimAtPath(self.prim_path)
        

        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return
    
    def set_robot(self) -> SingleManipulator:

        gripper_config = self.config["gripper_config"]

        add_reference_to_stage(usd_path=self.config["usd_path"], prim_path=self.prim_path)

        gripper = ParallelGripper(
            end_effector_prim_path=f"{self.prim_path}/{gripper_config['end_effector_prim_path']}",
            # end_effector_prim_path="/World/Doosan_M1013/robotiq_arg2f_base_link",
            joint_prim_names=gripper_config["joint_prim_names"],
            joint_opened_positions=np.array(gripper_config["joint_opened_positions"]),
            joint_closed_positions=np.array(gripper_config["joint_closed_positions"]),
            action_deltas=np.array([-0., 0.]),
        )

        manipulator = SingleManipulator(
            prim_path=self.prim_path,
            name="robot",
            end_effector_prim_path=f"{self.prim_path}/{gripper_config['end_effector_prim_path']}",
            gripper=gripper,
        )
        joints_default_positions = np.zeros(self.config["joint_total_num"])
        # joints_default_positions[6] = 0
        # joints_default_positions[7] = 0
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
                            robot_description_path=self.config["manipulator_config"]["robot_description_path"],
                            urdf_path = self.config["manipulator_config"]["urdf_path"],
        )

        return kinematics_solver


    def compute_ik(self,
        target_position : Optional[list],
        target_orientation : Optional[list], 
        frame_name : str = None,
        warm_start : np.ndarray = np.array([0]*6)
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

    def compute_fk(self, 
        frame_name:str, 
        joint_positions : Optional[np.ndarray] 
        ):
        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        fk = self.kinematics_solver.compute_forward_kinematics(frame_name =frame_name, joint_positions = joint_positions)
        print("fk : ", fk)
        return fk


    def pre_step(self, current_time_step_index, current_time):
        return
    
    def post_reset(self):
        
        return