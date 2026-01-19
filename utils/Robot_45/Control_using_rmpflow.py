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

from .rmp_flow.rmpflow_controller import RMPFlowController
from .rmp_flow.follow_target import FollowTarget



my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()






class Rmpflow_Robot:
    def __init__(self, chunk_size=8, action_size = 7):

        env_prim_path = "/World/Robot"
        add_reference_to_stage(usd_path = "/nas/ochansol/isaac/USD/robots/manipulator/Franka_panda/Franka.usd", prim_path = env_prim_path)


        my_task = FollowTarget(name="follow_target_task", robot_prim_path = env_prim_path)
        my_world.add_task(my_task)
        my_world.reset()
        task_params = my_world.get_task("follow_target_task").get_params()
        franka_name = task_params["robot_name"]["value"]
        self.target_name = task_params["target_name"]["value"]
        self.robot = my_world.scene.get_object(franka_name)
        self.robot.set_world_pose(position = [0,0,0.50214])
        self.rmp_controller = RMPFlowController(name="target_follower_controller", robot_name = "Franka", robot_articulation=self.robot)
        self.articulation_controller = self.robot.get_articulation_controller()

        self.target_offset = np.array([0.5, 0.0, -0.374])
        self.target_init_pos = np.array([-0.211, -0.011, 1.174]) + self.target_offset

        target_prim = stage.GetPrimAtPath("/World/TargetCube")
        imageable = UsdGeom.Imageable(target_prim)
        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        target_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(self.target_init_pos[0], self.target_init_pos[1], self.target_init_pos[2]))
        self.target_Prim = Prims.XFormPrim(name ="target_cube", prim_path="/World/TargetCube")

        self.action_chunk_arr = np.array([])
        self.chunk_size = chunk_size
        self.action_size = action_size
        self.action_scale = 0.008
        self.ensemble_size = 4
        self.ensemble_weights_scale = 0.5

        self.ensemble_weights = np.array([-self.ensemble_weights_scale*i for i in range(self.ensemble_size)])
        self.gripper_close_pos = np.array([0.0, 0.0])
        self.gripper_open_pos = np.array([0.04, 0.04])


    def set_target_prim(self, action):
        ### target_position
        tp = action[:3] * self.action_scale
        tp = self.target_pos + np.array([tp[0], tp[1], tp[2]])

        ### target_orientation
        to = self.target_ori_r + action[3:6] * self.action_scale *10
        to = np.array([np.pi, 0, to[2]])

        self.apply_gripper_action(action[-1])
        print(np.round(action[-1],4))

        self.target_Prim.set_world_pose(
            position =  tp, 
            # orientation = euler_angles_to_quat(to)
            )
    def apply_gripper_action(self, gripper_action):

        action =  0.04-( (self.gripper_open_pos - self.gripper_close_pos) * (gripper_action +1)/2 + self.gripper_close_pos )
        self.articulation_controller.apply_action(
            ArticulationAction(
                joint_indices=[7,8] ,
                joint_positions = action
            ) 
        )

    def apply_action(self):

        observations = my_world.get_observations()
        actions = self.rmp_controller.forward(
            target_end_effector_position=observations[self.target_name]["position"],
            target_end_effector_orientation=observations[self.target_name]["orientation"],
        )
        self.articulation_controller.apply_action(actions)

    def get_state(self):
        self.target_pos, self.target_ori = self.target_Prim.get_world_pose()
        self.target_pos_r = self.target_pos - self.target_offset
        self.target_ori_r = np.array(quat_to_euler_angles(self.target_ori))

        target_rpy = np.array([self.target_ori_r[0]+np.pi*2, self.target_ori_r[1], self.target_ori_r[2]-np.pi])
        gripper_state = self.robot.gripper.get_joint_positions()[-2:]
        total = np.hstack((self.target_pos_r, target_rpy, gripper_state))
        state = total.tolist()
        return state
    
    def set_action(self, action):
        if len(self.action_chunk_arr) == 0:
            self.action_chunk_arr = action[None,:]
        elif len(self.action_chunk_arr) < self.ensemble_size:
            self.action_chunk_arr = np.vstack((self.action_chunk_arr, action[None,:]))
        else:
            self.action_chunk_arr = np.vstack((self.action_chunk_arr[1:], action[None,:]))
            temporal_ensemble_action = self.ensemble_weights.dot( self.action_chunk_arr[[i for i in range(self.ensemble_size)],[-i-1 for i in range(self.ensemble_size)]] ) / self.ensemble_weights.sum()
            self.set_target_prim(temporal_ensemble_action)

    def resset(self):
        self.rmp_controller.reset()


