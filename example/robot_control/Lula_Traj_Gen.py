
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

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.isaac_utils_51 import scan_rep, rep_utils
from Utils.general_utils import mat_utils
from Utils.Robot_45 import robot_configs, robot_policy



my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)

# Robot_Cfg = robot_configs.ROBOT_CONFIGS["UR10"]()
Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" )
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




c_space_points = np.array([
    [0, 0, 0, 0, 0, 0,0,0],
    [0.2, -0.2, 0.2, 0, 0.2, 0,0,0],
    [0.4, -0.4, 0.4, 0, 0.4, 0,0,0],
            ])
task_space_position_targets = np.array([
    [0.2, 0., 0.2],
    [0.0, 0., 0.2],
    [-0.2, 0., 0.2],
    ])
task_space_orientation_targets = np.tile(mat_utils.euler_to_quat(np.array([0,0,0]), degrees=True),(3,1))

action_idx = 0
# actions = my_robot_task.trajectory_gen(c_space_points)#, time_array = np.array([0,5,10,15,20]))

actions = my_robot_task.trajectory_gen_taskspace(task_space_position_targets, task_space_orientation_targets, ee_prim_name="OMY_grasp_joint")
# actions = my_robot_task.trajectory_gen_taskspace(task_space_position_targets, task_space_orientation_targets, ee_prim_name="ee_link")
my_world.reset()
my_world.pause()

while True:
    my_world.step(render=True)


    if my_world.is_playing():

        my_robot.apply_action(actions[action_idx])
        if my_world.current_time > (action_idx+1)*0.01:
            action_idx += 1
            if action_idx >= len(actions):
                action_idx = 0

