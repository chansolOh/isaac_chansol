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
from isaacsim.core.prims import XFormPrim
import omni.usd
from pxr import UsdGeom, Gf
import torch
from isaacsim.core.utils.types import ArticulationAction

from .robot_configs import RobotConfig
from ..general_utils import mat_utils


from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)
from isaacsim.core.api import World
import omni.replicator.core as rep


# Inheriting from the base class Follow Target

def dpi_arr(deg_arr, target_deg_arr):
    result_arr = target_deg_arr.copy()
    over_idx = (deg_arr - target_deg_arr) >180
    result_arr[over_idx] +=360
    under_idx = (deg_arr - target_deg_arr) <-180
    result_arr[under_idx] -=360
    return result_arr

def detect_joint_jumps(q: np.ndarray, thr: float):
    """
    q: (T, J) joint array
    thr: 한 스텝에서 허용하는 최대 변화량 (rad 또는 deg 기준 일관되게)
    return:
      jumps_idx: 급변이 발생한 '스텝 인덱스' (t: q[t-1] -> q[t])
      jump_joints: 각 점프에서 어떤 joint가 튀었는지 (list of arrays)
      dq: (T-1, J) 스텝 차이
    """
    dq = np.diff(q, axis=0)                  # (T-1, J)
    mask = np.abs(dq) > thr                  # (T-1, J)
    jumps_idx = np.where(mask.any(axis=1))[0] + 1  # q[t]가 튄 t
    if len(jumps_idx) > 0:
        jump = True
    else:
        jump = False
    return jump

class My_Robot_Task(tasks.BaseTask):
    def __init__(
        self,
        robot_config: RobotConfig,
        name: str = "My_Robot",
        idle_joint: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, offset=offset)

        self.cfg = robot_config
        for k, v in vars(self.cfg).items():
            setattr(self, k, v)
        self.world = World()
        self.stage = omni.usd.get_context().get_stage()

        self.physics_dt = self.world.get_physics_dt()
        self.idle_joint = idle_joint
        return

    
    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)


        self._robot = self.set_robot()
        self.kinematics_solver = self.set_solver()
        self.c_space_trajectory_generator, self.taskspace_trajectory_generator = self.set_motion_gen_solver()
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
                prim_path=self.prim_path if self.extra_prim_path is None else self.extra_prim_path,
                name=self.name,
                position = self.robot_pos,
                orientation = self.robot_ori,
                scale = self.robot_scale,
            )

        else:
            self.gripper = ParallelGripper(
                end_effector_prim_path=os.path.join(self.prim_path if self.extra_prim_path is None else self.extra_prim_path, self.ee_link_path),
                joint_prim_names=self.gripper_joint_prim_names,
                joint_opened_positions=self.joint_opened_positions,
                joint_closed_positions=self.joint_closed_positions,
                action_deltas=np.array([-0., 0.]),
            )

            manipulator = SingleManipulator(
                prim_path=self.prim_path,
                name=self.name,
                end_effector_prim_path=os.path.join(self.prim_path if self.extra_prim_path is None else self.extra_prim_path, self.ee_link_path),
                gripper=self.gripper,
                position = self.robot_pos ,
                orientation = self.robot_ori ,
                scale = self.robot_scale ,
            )
 
        joints_default_positions = np.zeros(self.total_joint_num) if self.idle_joint is None else self.idle_joint
        manipulator.set_joints_default_state(positions=torch.tensor(joints_default_positions, dtype=torch.float32))

        return manipulator


    def set_world_pose(self, position, rotation):
        if self.robot_prim.HasAttribute("xformOp:translate"):
            self.robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(position))
        return
    

    @property
    def get_robot_name(self):
        return self._robot.name
    def set_semantic_labels(self):
        rep.utils._set_semantics_legacy(self.robot_prim, [("class", "robot")])

    def set_solver(self):
        kinematics_solver = LulaKinematicsSolver(
                            robot_description_path=self.description_path,
                            urdf_path = self.urdf_path
        )
        return kinematics_solver
    def set_motion_gen_solver(self):
        c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path = self.description_path,
            urdf_path = self.urdf_path
        )

        taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path = self.description_path,
            urdf_path = self.urdf_path
        )
        return c_space_trajectory_generator, taskspace_trajectory_generator

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
        init_joint_state : Optional[np.ndarray] = None,
        return_traj: bool = False
        ):

        init_pos, init_rot = self.compute_fk(
            frame_name = frame_name,
            joint_positions = init_joint_state
        )
        if init_joint_state is not None:
            warm_start = init_joint_state
        else:
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
        # plt.plot(np.array(joint_position_traj)/np.pi*180)
        # plt.show()
        if return_traj:
            if detect_joint_jumps(np.array(joint_position_traj)/np.pi*180, thr=25):
                print("error: Joint jump detected in IK trajectory!")
                return self.get_joint_positions()[None,:]
            
            return np.array(joint_position_traj)
        
        return joint_positions


    def compute_fk(self, 
        frame_name:str, 
        joint_positions: Optional[np.ndarray] = None
        ):

        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        pos, rot_mat = self.kinematics_solver.compute_forward_kinematics(frame_name =frame_name, joint_positions = self.get_joint_positions() if joint_positions is None else joint_positions)

        rot = mat_utils.mat_to_euler(rot_mat)
        return pos, rot
    
    def apply_action(self, joint_indices, joint_positions):
        self._robot.apply_action(ArticulationAction(
                            joint_indices=joint_indices ,  ####  joint name으로 index 찾아오기
                            joint_positions = joint_positions))
        return

    
    def get_joint_positions(self):
        return self._robot.get_joint_positions()[:len(self.joint_names)]

    def pre_step(self, current_time_step_index, current_time):
        return
    
    def post_reset(self):
        # SingleArticulation.post_reset(self._robot)
        return
    def trajectory_gen_cspace(self, joint_array,time_array=None, physics_dt = None):
        if time_array is None:
            ## optimal time trajectory
            trajectory_time = self.c_space_trajectory_generator.compute_c_space_trajectory(joint_array)
        else:
            ## timestamped trajectory
            trajectory_time = self.c_space_trajectory_generator.compute_timestamped_c_space_trajectory(joint_array,time_array)


        articulation_trajectory = ArticulationTrajectory(self._robot, trajectory_time, self.physics_dt if physics_dt is None else physics_dt)
        action_sequence=articulation_trajectory.get_action_sequence()

        return action_sequence
    

    def trajectory_gen_taskspace(self, position_array, orientation_array, ee_prim_name):

        trajectory_time = self.taskspace_trajectory_generator.compute_task_space_trajectory_from_points(position_array, orientation_array, ee_prim_name)
        
        articulation_trajectory = ArticulationTrajectory(self._robot, trajectory_time, self.physics_dt)
        action_sequence=articulation_trajectory.get_action_sequence()

        return action_sequence
    
    def rrt_plan_to_traj_actions(self,plan, physics_dt = 0.02):
        joint_waypoints = []
        for action in plan:
            joint_waypoints.append(action.joint_positions)
        return self.trajectory_gen_cspace(np.asarray(joint_waypoints), physics_dt=physics_dt)

    def rrt_plan_to_taskspace(self, plan, ee_prim_name):
        position_array = []
        orientation_array = []
        for action in plan:
            pos, euler = self.compute_fk(frame_name=ee_prim_name, joint_positions=action)
            position_array.append(pos)
            orientation_array.append(mat_utils.euler_to_quat(euler, degrees=True))
        return np.array(position_array), np.array(orientation_array)

