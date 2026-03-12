import sys
from pathlib import Path

import yaml
from yaml import SafeLoader as Loader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Utils.Robot_45 import robot_configs
from socket_utils.client import TcpClient

Robot_name = "Robotis_OMY"
Robot_Cfg = robot_configs.ROBOT_CONFIGS[Robot_name]()

client = TcpClient("192.168.0.137", 9111, name="chansol")
client.connect()

robot_config = {
    "Robot_name": Robot_name,
    "yml_path": Robot_Cfg.curobo_yml_path
}
block =client.send("set_robot_config", robot_config, blocking=True, recv_timeout_sec=500)





from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World

import numpy as np
import os

import carb
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import XFormPrim
import omni.isaac.core.prims as Prims
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path

from isaacsim.robot_motion.motion_generation import (
    LulaCSpaceTrajectoryGenerator,
    LulaTaskSpaceTrajectoryGenerator,
    LulaKinematicsSolver,
    ArticulationTrajectory
)

import lula
import omni.isaac.core.utils.prims as prim_utils


from Utils.isaac_utils_51 import scan_rep, rep_utils
from Utils.general_utils import mat_utils
from Utils.Robot_45 import robot_policy

from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects.cuboid import VisualCuboid
import copy

my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)


my_robot_task = robot_policy.My_Robot_Task(
    robot_config=Robot_Cfg, 
    name="robot_task",
    # idle_joint=np.array([0,-90,0,0,0,0])/180*np.pi )
    idle_joint=np.array([0,0,0,0,0,0,0,0,0,0])/180*np.pi )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim

light_1 = prim_utils.create_prim(
    "/World/Light_1",
    "SphereLight",
    position=np.array([0, 0, 20.0]),
    attributes={
        "inputs:radius": 0.01,
        "inputs:intensity": 5e3,
        "inputs:color": (255, 250, 245),
        "inputs:exposure" : 12,
    }
)


world_config = {
    "mesh": {
        "base_scene": {
            "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
            # "file_path": "scene/nvblox/srl_ur10_bins.obj",
            "file_path": "/nas/ochansol/isaac/USD/robots/manipulator/UR5e/srl_ur10_bins.obj",
        },
    },
    "cuboid": {
        "table": {
            "dims": [5.0, 5.0, 0.2],  # x, y, z
            "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
        "table2": {
            "dims": [.1, .1, 0.1],  # x, y, z
            "pose": [-0.3, 0.15, 0.3, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
        "table3": {
            "dims": [.1, .1, 0.1],  # x, y, z
            "pose": [0.3, -0.15, 0.4, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}

prim_dict = copy.deepcopy(world_config)

for key in world_config:
    if key == "mesh":
        for mesh_name, mesh_info in world_config[key].items():
            pose = mesh_info["pose"]
            file_path = mesh_info["file_path"]

            pos = pose[:3]
            quat = pose[3:]
            mesh_prim = add_reference_to_stage(
                prim_path=f"/World/mesh/{mesh_name}", 
                usd_path=file_path)
            if mesh_prim.HasAttribute("xformOp:rotateX:unitsResolve"):
                mesh_prim.RemoveProperty("xformOp:rotateX:unitsResolve")
            if mesh_prim.HasAttribute("xformOp:rotateY:unitsResolve"):
                mesh_prim.RemoveProperty("xformOp:rotateY:unitsResolve")
            if mesh_prim.HasAttribute("xformOp:rotateZ:unitsResolve"):
                mesh_prim.RemoveProperty("xformOp:rotateZ:unitsResolve")
            if mesh_prim.HasAttribute("xformOp:scale:unitsResolve"):
                mesh_prim.RemoveProperty("xformOp:scale:unitsResolve")
            mesh_prim.GetAttribute("xformOp:translate").Set(rep_utils.np_to_GfVec3f(pos))
            mesh_prim.GetAttribute("xformOp:orient").Set(rep_utils.np_to_GfQuatf(quat))
            prim_dict[key][mesh_name]["prim"] = mesh_prim
    elif key == "cuboid":
        for cuboid_name, cuboid_info in world_config["cuboid"].items():
            dims = cuboid_info["dims"]
            pose = cuboid_info["pose"]
            pos = pose[:3]
            quat = pose[3:]
            cuboid_prim = VisualCuboid(
                prim_path=f"/World/cuboid/{cuboid_name}",
                position=pos,
                orientation=quat,
                scale=dims
            )
            prim_dict[key][cuboid_name]["prim"] = cuboid_prim


target_cube = VisualCuboid("/World/target_cube", position = [0.2,0.2,0.5], size = 1.0, scale = [0.1,0.1,0.1])

client.send("set_world_config", world_config)


goal_pos=np.array([[-0.3, 0.3, 0.3],
                   [-0.3, -0.3, 0.3],
                   [0.3, -0.3, 0.3],
                   [0.4, 0.6, 0.3],
                   [0.2,0.7,0.1]])
goal_rot = np.array([mat_utils.euler_to_quat(i,degrees=True) for i in [[179,0,0]]*len(goal_pos)])

for idx, pos in enumerate(goal_pos):
    VisualCuboid(f"/World/viz_{idx}", position = pos, size = 1.0, scale = np.array([0.05,0.05,0.05]))

action_idx = 0
stage_idx = 0
goal_idx = 0
while True:
    my_world.step(render=True)


    if my_world.is_playing():
        if stage_idx ==0:
            goal_pos = target_cube.get_world_pose()[0]
            goal_rot = target_cube.get_world_pose()[1]
            res =client.send("get_traj",{
                                    # "goal_pos": goal_pos[goal_idx].tolist(),
                                    # "goal_rot": goal_rot[goal_idx].tolist(),
                                    "goal_pos": goal_pos.tolist(),
                                    "goal_rot": goal_rot.tolist(),
                                    "joint_state": my_robot_task.get_joint_positions()[:6].tolist(),
                                    "joint_names": my_robot_task.joint_names[:6]
                                })
            
            result = res["success"]
            if not result:
                print("Failed to get trajectory from server.")
                print("goal_pos:", goal_pos[goal_idx],
                      "goal_rot:", goal_rot[goal_idx],
                     "joint_state:", my_robot_task.get_joint_positions()[:6],
                     "joint_names", my_robot_task.joint_names[:6]                    
                    )
                stage_idx = 2
                continue


            traj = np.array(res["trajectory"])
            traj = np.hstack((traj,np.zeros((len(traj),2))))
            actions = my_robot_task.trajectory_gen_cspace(traj, physics_dt=0.05)#, time_array = np.array([0,5,10,15,20]))
            print("Received trajectory from server.")
            stage_idx += 1

        elif stage_idx == 1:
            # my_robot.apply_action(ArticulationAction(
            #                         joint_indices=[0,1,2,3,4,5] ,
            #                         joint_positions = traj[action_idx]) )
            my_robot.apply_action(actions[action_idx])

            if my_world.current_time > (action_idx+1)*0.01:
                action_idx += 1
                if action_idx >= len(actions):
                    action_idx = 0
                    stage_idx += 1

        elif stage_idx == 2:
            goal_idx += 1
            stage_idx = 0
            if goal_idx >= len(goal_pos):
                goal_idx = 0
            




        # if my_world.current_time > (stage_idx+1)*3.0:
        #     stage_idx += 1
        #     if stage_idx >= len(goal_pos):
        #         stage_idx = 0
