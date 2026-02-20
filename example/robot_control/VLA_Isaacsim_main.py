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

from Utils.Robot_45 import Control_using_rmpflow as rmp_control
from Utils.Robot_45 import Control_using_basic_ik as basic_ik
from socket_utils.vla_socket.vla_client import VLAClient

import Utils.isaac_utils_51.rep_utils as csr
import Utils.isaac_utils_51.scan_rep as scan_rep
import Utils.isaac_utils_51.light_set as light


from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import numpy as np

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")

my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)
stage = omni.usd.get_context().get_stage()




# Robot_inst = rmp_control.Rmpflow_Robot( chunk_size=8, action_size = 7)
Robot_inst = basic_ik.BasicIk( chunk_size=8, action_size = 7)
env_prim = stage.GetPrimAtPath(Robot_inst.robot_task.prim_path)



################### camera setup ####################
full_img_cam_path = "/World/Robot/demo/full_camera"
wrist_img_cam_path = "/World/Robot/OMY_custom_no_delay/OMY/link6/wrist_camera"

full_res=(1280,720)
wrist_res=(848,480)
full_camera = Camera(
    prim_path=full_img_cam_path,
    name="cam_top",
    frequency=30,
    resolution=full_res,)

wrist_camera = Camera(
    prim_path=wrist_img_cam_path,
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
        "path": os.path.join(obj_root_path, "apple/edited/apple.usd"),
        "size_rank": 0,
    },
    # "paprika":{
    #     "name":"paprika",
    #     "path": os.path.join(obj_root_path, "paprika/edited/paprika.usd"),
    #     "size_rank": 0,
    # },
    # "potato":{
    #     "name":"potato",
    #     "path": os.path.join(obj_root_path, "potato/edited/potato.usd"),
    #     "size_rank": 0,
    # },    
}

# box_path_list = [os.path.join(Robot_inst.robot_task.prim_path,i) for i in ["custom_box_12_12_08_blue", "custom_box_12_12_08_yellow","custom_box_12_12_08_magenta"]]

box_path_list = [os.path.join(Robot_inst.robot_task.prim_path,i) for i in ["custom_box_12_12_08_magenta"]]
box_rep_list = []
for box_path in box_path_list:
    box_rep = scan_rep.Scan_Rep(
        prim_path = box_path,
        class_name = box_path.split("/")[-1],
        scale=[1,1,1],
        )
    box_rep_list.append(box_rep)

obj_rep_all_list = [] + box_rep_list
for key in sampled_model_dict:
    model_attr = sampled_model_dict[key]
    print("model_attr : ", model_attr["name"])
    scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                            class_name = model_attr["name"],
                            size = model_attr["size_rank"],)
    obj_rep_all_list.append(scan_obj)


for OBJ in obj_rep_all_list:
    print("set collider for : ", OBJ.class_name)
    OBJ.set_rigidbody_collider()
    # OBJ.remove_collider()
    OBJ.set_physics_material(
        dynamic_friction=0.25,
        static_friction=0.4,
        restitution=0.0
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




        state = np.array(Robot_inst.get_state(action_type="joint"))[[0,1,2,3,4,5,-1]].tolist()  # joint state + gripper state
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
            Robot_inst.set_action(client.action, action_type="joint", action_chunk=False)


        #### action
        if reset_needed:
            my_world.reset()
            Robot_inst.reset()
            reset_needed = False

        Robot_inst.action_step()