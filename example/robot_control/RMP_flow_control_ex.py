
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import cuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy

from isaacsim.robot_motion.motion_generation.lula import RmpFlow
from pxr import Usd, UsdGeom

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.isaac_utils_51 import scan_rep, rep_utils
from Utils.general_utils import mat_utils
from Utils.Robot_45 import robot_configs, robot_policy
import Utils.isaac_utils_51.rep_utils as csr

from Utils.isaac_utils_51.debug_tools import debug_draw_lines, debug_draw_obb, debug_draw_points, debug_draw_clear



my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)
Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" ,
                idle_joint=np.array([0,-32,25,43,92,0,0,0,0,0])/180*np.pi 
                )
my_world.add_task(my_robot_task)
my_world.reset()
stage = omni.usd.get_context().get_stage()
robot_name = my_robot_task.get_robot_name
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim
env_prim = add_reference_to_stage(prim_path = "/World/env", usd_path ="/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd")


rmp_config = {
    "end_effector_frame_name": "OMY_grasp_joint",
    "maximum_substep_size" : 0.00334,
    "ignore_robot_state_updates" : False,
    "robot_description_path" : Robot_Cfg.rrt_description_path,
    "urdf_path": Robot_Cfg.urdf_path,
    "rmpflow_config_path" : Robot_Cfg.rmpflow_config_path
    
}


# Initialize an RmpFlow object
rmpflow = RmpFlow(**rmp_config)
physics_dt = 1 / 60.0
articulation_rmpflow = ArticulationMotionPolicy(my_robot, rmpflow, physics_dt)
articulation_controller = my_robot.get_articulation_controller()

# Make a target to follow
target_cube = cuboid.VisualCuboid(
    "/World/target", position=np.array([0.5, 0, 0.5]), color=np.array([1.0, 0, 0]), size=0.1
)

# Make an obstacle to avoid
obstacle = cuboid.VisualCuboid(
    "/World/obstacle", position=np.array([0.8, 0, 0.5]), color=np.array([0, 1.0, 0]), size=0.1
)
rmpflow.add_obstacle(obstacle)



my_world.reset()
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False

        target_orientation = target_cube.get_world_pose()[1]

        rmpflow.set_end_effector_target(
            target_position=target_cube.get_world_pose()[0], target_orientation=target_orientation
        )

        rmpflow.update_world()

        actions = articulation_rmpflow.get_next_articulation_action()
        articulation_controller.apply_action(actions)

simulation_app.close()
