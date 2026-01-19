import numpy as np
import matplotlib.pyplot as plt


from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World

from isaacsim.core.utils.stage import add_reference_to_stage


import omni.isaac.core.prims as Prims
import omni.usd
from pxr import Usd, UsdGeom, Gf

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.Robot_45 import Control_using_rmpflow as rmp_control
from utils.Robot_45 import Control_using_basic_ik as basic_ik

my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()




env_prim_path = "/World/open_vla_oft"
add_reference_to_stage(usd_path = "/nas/ochansol/VLA/USD/openvla_oft/panda_robot_env.usd", prim_path = env_prim_path)

Robot_inst = rmp_control.Rmpflow_Robot( chunk_size=8, action_size = 7)
# Robot_inst = basic_ik.BasicIk( chunk_size=8, action_size = 7)


full_img_cam_path = "/World/open_vla_oft/full_image_cam"
wrist_img_cam_path = "/World/Robot/panda_hand/wrist_camera"



description = 'pick up the black bowl between the plate and the ramekin and place it on the plate'




Robot_inst.robot.gripper.open()

vla_flag = False
action_flag = True
reset_needed = False

ot = 0
i=0
while simulation_app.is_running():
    my_world.step(render=True)
    sim_t = my_world.current_time

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
      


        state = Robot_inst.get_state()
 


        #### action
        if reset_needed:
            my_world.reset()
            Robot_inst.reset()
            reset_needed = False

        Robot_inst.apply_action()