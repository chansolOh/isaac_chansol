
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

from isaacsim.core.api.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils
import omni

import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrim

import json
import os
import matplotlib.pyplot as plt



import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.isaac_utils_51 import scan_rep, rep_utils
from Utils.general_utils import mat_utils
from Utils.Robot_45 import robot_configs, robot_policy





object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]
root_path = "/nas/ochansol/isaac"




my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)

stage = omni.usd.get_context().get_stage()
GroundPlane(prim_path="/World/GroundPlane", z_position=0)
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
light_2 = prim_utils.create_prim(
    "/World/Light_2",
    "SphereLight",
    position=np.array([0, 0.79, 1.57]),
    attributes={
        "inputs:radius": 0.25,
        "inputs:intensity": 5e3,
        "inputs:color": (255, 250, 245),
        "inputs:exposure" : -4,
    }
)

Robot_Cfg = robot_configs.ROBOT_CONFIGS["Doosan_M1013"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task",
                                           idle_joint = np.array([
                                               [  94.43 ,  -34.29 , 113.58  , -4.01 , 100.53, 1.71]
                                               ])/180.0*np.pi )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim








physics_scene_conf={
    # 'physxScene:enableGPUDynamics': 1, # True
    # 'physxScene:broadphaseType' : "GPU",
    # 'physxScene:collisionSystem' : "PCM",
    
    # 'physxScene:timeStepsPerSecond' : 1000,
    'physxScene:minPositionIterationCount' : 30,
    'physxScene:minVelocityIterationCount' : 20,
    "physics:gravityMagnitude":35,
    # "physxScene:updateType":"Asynchronous",
}
for key in physics_scene_conf.keys():
    stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
        
        
target_prim_path = "/World/target_xform2"
target_xprim = XFormPrim(
    prim_path=target_prim_path,
    name="my_xform2",
    position=np.array([0.158, -0.08, 0.43]),
    orientation=mat_utils.euler_to_quat(np.array([180.0, 0.0, 0.0]), degrees=True),  # quat (w, x, y, z) 형태가 보통
)
my_world.scene.add(target_xprim)



traj_idx = 0
i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.001

my_world.stop()




while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and stop_flag:
        traj_idx = 0
        i=0
        state=0
        ik_first_flag=True
        obj_reset_flag = True
        stop_flag = False
        my_world.reset()
        my_world.pause()

    if my_world.is_playing():

        # import pdb; pdb.set_trace()
        stop_flag=True
        if my_world.current_time_step_index <= 1:
            my_world.reset() 
        i += 1
        

        if state==0:
            if ik_first_flag:
                target_pos, target_orientation = target_xprim.get_world_pose()
                target_orientation = mat_utils.quat_to_euler(np.array(target_orientation), degrees=True)

   

                target_joint_positions = my_robot_task.compute_ik_traj(target_position = target_pos,
                                            target_orientation = target_orientation,
                                            frame_name = "Robotiq_2f140_open",
                                            return_traj=True
                                            )
                traj_idx = 0
                ik_first_flag =False

                # print(target_pos)
                print(target_joint_positions/np.pi*180)
                # my_world.pause()

            target_joint = target_joint_positions[traj_idx]
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                  joint_positions = target_joint) )
            if i%5==0:
                if traj_idx < len(target_joint_positions)-1:
                    traj_idx +=1

            # if np.mean(joint_err)<joint_err_th:
            #     ik_first_flag = True
                # state+=1
 
        

        if i >= 400  :
            # state+=1
            i=0
            ik_first_flag = True
            traj_idx = 0
            # obj_reset_flag = True
        if state>=5:
            state=0

        # if target_idx >= gamja_rep.count:
        #     target_idx =0

simulation_app.close()