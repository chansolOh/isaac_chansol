# import os

# def setup_ros2_env_inside():
#     os.environ["ROS_DISTRO"] = "jazzy"
#     os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

#     jazzy_lib = os.path.expanduser(
#         "~/ochansol/isaac_code/isaac_chansol/.venv/lib/python3.11/site-packages/"
#         "isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
#     )
#     ld = os.environ.get("LD_LIBRARY_PATH", "")
#     if jazzy_lib not in ld.split(":"):
#         os.environ["LD_LIBRARY_PATH"] = (ld + ":" + jazzy_lib).strip(":")

# setup_ros2_env_inside()



from isaacsim import SimulationApp

# from isaac_chansol.example.test.image_serve_test import Robot_inst

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

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

from Utils.Robot_45 import Control_using_rmpflow as rmp_control
from Utils.Robot_45 import Control_using_basic_ik as basic_ik
from socket_utils.vla_socket.vla_client import VLAClient

import Utils.isaac_utils_51.rep_utils as csr
import Utils.isaac_utils_51.scan_rep as scan_rep
import Utils.isaac_utils_51.light_set as light


from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import numpy as np
from Utils.Robot_45 import robot_configs, robot_policy


from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")

my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)
stage = omni.usd.get_context().get_stage()




# Robot_inst = rmp_control.Rmpflow_Robot( chunk_size=8, action_size = 7)
my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)
Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" ,
                idle_joint=np.array([0,-32,25,43,92,0,0,0,0,0])/180*np.pi 
                )
my_world.add_task(my_robot_task)
my_world.reset()
my_robot = my_robot_task._robot
env_prim = add_reference_to_stage(prim_path = "/World/env", usd_path ="/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd")

################### camera setup ####################
full_cam_path = f"{str(env_prim.GetPrimPath())}/demo/full_camera"
wrist_cam_path = f"{my_robot_task.prim_path}/OMY/link6/wrist_camera"


full_res=(1280,720)
wrist_res=(848,480)
full_camera = Camera(
    prim_path=full_cam_path,
    name="cam_top",
    frequency=30,
    resolution=full_res,)

wrist_camera = Camera(
    prim_path=wrist_cam_path,
    name="cam_wrist",
    frequency=30,
    resolution=wrist_res,)

full_camera.initialize()
wrist_camera.initialize()
my_world.reset()

render_product_full = full_camera._render_product
render_product_wrist = wrist_camera._render_product

annotator_full = rep.AnnotatorRegistry.get_annotator("rgb")
annotator_full.attach([render_product_full])

annotator_wrist = rep.AnnotatorRegistry.get_annotator("rgb")
annotator_wrist.attach([render_product_wrist])
#################################


################ object setup ###############


obj_root_path = "/nas/ochansol/3d_model/scan_etc"
sampled_model_dict={
    "apple":{
        "name":"apple",
        "path": "/nas/ochansol/3d_model/scan_etc/apple_test/apple.usd",
        "size_rank": 0,
        "scale" : [0.1,0.1,0.1]
    },
    "box_magenta":{
        "name":"box_magenta",
        "path": "/nas/ochansol/3d_model/VLA/custom_box_12_12_08_magenta/custom_box_12_12_08_magenta.usd",
        "size_rank": 0,
        "scale" : [1,1,1]
    }
}


obj_rep_all_list = []
for key in sampled_model_dict:
    model_attr = sampled_model_dict[key]
    print("model_attr : ", model_attr["name"])
    scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                            class_name = model_attr["name"],
                            size = model_attr["size_rank"],
                            scale = model_attr.get("scale", [0.1,0.1,0.1])
                            )
    sampled_model_dict[key]["rep"] = scan_obj
    obj_rep_all_list.append(scan_obj)

for OBJ in obj_rep_all_list:
    print("set collider for : ", OBJ.class_name)
    OBJ.set_rigidbody_collider()
    # OBJ.remove_collider()
    OBJ.set_physics_material(
        dynamic_friction=0.25,
        static_friction=0.4,
        restitution=0.1
    )


my_world.reset()
#########################################

physics_scene_conf={
    # 'physxScene:enableGPUDynamics': 1, # True
    # 'physxScene:broadphaseType' : "GPU",
    # 'physxScene:collisionSystem' : "PCM",
    
    # 'physxScene:timeStepsPerSecond' : 1000,
    'physxScene:minPositionIterationCount' : 5,
    'physxScene:minVelocityIterationCount' : 5,
    # "physics:gravityMagnitude":35,
    # "physxScene:updateType":"Asynchronous",
}
for key in physics_scene_conf.keys():
    stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
        
platform_area_prims = csr.find_target_name(env_prim,["Mesh"],"platform_area")
platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == "demo"][0]

platform_path = platform_area_prims.GetPath().__str__()
platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])

my_world.reset()

platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_rep.set_tf(platform_tf)
platform_rep.set_scale(platform_scale)

csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False, rotation=False)




SERVER_IP = "127.0.0.1"
PORT = 1823
client = VLAClient(SERVER_IP, PORT)
client.connect()


description = 'put the apple in the box'


vla_flag = False
action_flag = True
reset_needed = False
stop_flag = True

ot = 0
i=0

import time
SEND_HZ = 25
send_period = 1.0 / SEND_HZ
next_send_time = time.perf_counter()
client.start_infer_thread(hz=SEND_HZ)

while simulation_app.is_running():
    my_world.step(render=True)
    sim_t = my_world.current_time

    if my_world.is_stopped() and stop_flag:
        i=0
        state=0
        ik_first_flag=True
        obj_reset_flag = True
        stop_flag = False
        record_flag = False
        my_world.reset()

        csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False, rotation=False)




    if my_world.is_playing():
      
        stop_flag=True
        now = time.perf_counter()
        if now < next_send_time:
            continue
        next_send_time += send_period



        state = my_robot_task.get_joint_positions()[[0,1,2,3,4,5,7]].tolist()  # joint state + gripper state
        # state = np.array(Robot_inst.get_state(action_type="joint"))[[0,1,2,3,4,5,-1]].tolist()  # joint state + gripper state
        full_rgb = annotator_full.get_data()
        wrist_rgb = annotator_wrist.get_data()

        client.push(
            images_bgr={
                "full": full_rgb, 
                "wrist": wrist_rgb
                }, 

            obs={
                "joint_state": state
                }, 

            action_type="joint")  # 이미지 1장
        
        if client.action is not None:
            my_robot.apply_action(ArticulationAction(
                        joint_indices=[0,1,2,3,4,5,7] ,  ####  joint name으로 index 찾아오기
                        joint_positions = client.action))
            # Robot_inst.set_action(client.action, action_type="joint", action_chunk=False)


        #### action
        if reset_needed:
            my_world.reset()
            # Robot_inst.reset()
            reset_needed = False

        # Robot_inst.action_step()