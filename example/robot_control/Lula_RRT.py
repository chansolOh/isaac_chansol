
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

from isaacsim.core.api.objects.cuboid import VisualCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.distance_metrics import rotational_distance_angle

from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import interface_config_loader



import lula
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf
import omni

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
stage = omni.usd.get_context().get_stage()
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

target_path = "/World/target_cube"

target_cube = UsdGeom.Cube.Define(stage, target_path)
target_prim = UsdGeom.Xformable(target_cube.GetPrim())
# target_prim.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.1))

target = XFormPrim(target_path)
target.set_default_state(np.array([.45, .5, .7]),euler_angles_to_quats([3*np.pi/4, 0, np.pi]))

obstacle = VisualCuboid("/World/Wall", position = np.array([.3,.6,.6]), size = 1.0, scale = np.array([.1,.4,.4]))


#Initialize an RRT object
# rrt = RRT(
#     robot_description_path = "/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",#rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
#     urdf_path = "/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur10/ur10_robot.urdf",#rmp_config_dir + "/franka/lula_franka_gen.urdf",
#     rrt_config_path = "/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur10/planner_config.yaml",
#     end_effector_frame_name = "ee_link"
# )
rrt = RRT(
    robot_description_path = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_RRT.yaml",#rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
    urdf_path = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",#rmp_config_dir + "/franka/lula_franka_gen.urdf",
    rrt_config_path = "/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/omy/planner_config.yaml",
    end_effector_frame_name = "OMY_grasp_joint"
)

# RRT for supported robots can also be loaded with a simpler equivalent:
# rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
# rrt = RRT(**rrt_confg)

rrt.add_obstacle(obstacle)

# Set the maximum number of iterations of RRT to prevent it from blocking Isaac Sim for
# too long.
rrt.set_max_iterations(5000)

# Use the PathPlannerVisualizer wrapper to generate a trajectory of ArticulationActions
path_planner_visualizer = PathPlannerVisualizer(my_robot, rrt)





target_translation = np.zeros(3)
target_rotation = np.eye(3)
frame_counter = 0
plan = None

my_world.reset()
my_world.pause()
while True:
    my_world.step(render=True)



    if my_world.is_playing():
        current_target_translation, current_target_orientation = target.get_world_poses()
        current_target_translation = current_target_translation[0]
        current_target_orientation= current_target_orientation[0]

        current_target_rotation = quats_to_rot_matrices(current_target_orientation)

        translation_distance = np.linalg.norm(target_translation - current_target_translation)
        rotation_distance = rotational_distance_angle(current_target_rotation, target_rotation)
        target_moved = translation_distance > 0.01 or rotation_distance > 0.01
        if (frame_counter % 60 == 0 and np.any(target_moved)):
            # Replan every 60 frames if the target has moved
            rrt.set_end_effector_target(current_target_translation, current_target_orientation)
            rrt.update_world()
            plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)

            target_translation = current_target_translation
            target_rotation = current_target_rotation

        if plan:
            action = plan.pop(0)
            my_robot.apply_action(action)


