
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np


from isaacsim.core.api.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils
import omni
import carb
from isaacsim.util.debug_draw import _debug_draw

import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera


import json
import os
import matplotlib.pyplot as plt



import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaac_chansol.isaac_utils_45 import scan_rep, rep_utils
from isaac_chansol.general_utils import mat_utils
from isaac_chansol.Robot_45 import robot_configs, robot_policy




def debug_draw_obb(obb):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_lines(
        [carb.Float3(i) for i in obb[[0,1,2,2,0,4,5,5,7,6,7,6]]] , 
        [carb.Float3(i) for i in obb[[1,3,3,0,4,5,1,7,6,4,3,2]]] , 
        [carb.ColorRgba(1.0,0.0,0.0,1.0)]*12,
        [1]*12 )
    return draw

def debug_draw_points(points,size = 3, color=[1,0,0]):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_points(
        [carb.Float3(i) for i in points] , 
        [carb.ColorRgba(color[0],color[1],color[2],1.0)]*len(points),
        [size]*len(points) )
    return draw

def debug_draw_clear():
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()


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

Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim



cam_model_conf_path = "/nas/ochansol/camera_params/azure_kinect_conf.json"
with open(cam_model_conf_path, 'r') as f:
    cam_model_conf = json.load(f)


# ((fx,_,cx),(_,fy,cy),(_,_,_))= cam_model_conf["intrinsic_matrix"]



full_res=(1920,1080)
cx,cy = full_res[0]/2, full_res[1]/2
focal_length = cx

full_cam_path = f"{my_robot_task.prim_path}/cam_top2"

# full_camera = Camera(
#     prim_path=full_cam_path,
#     name="cam_top2",
#     frequency=25,
#     resolution=full_res,)

# full_camera.initialize()

# render_product_full = full_camera._render_product

# instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
# instance_seg_annotator.attach([render_product_full])
# depth_plane_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
# depth_plane_annotator.attach([render_product_full])









# object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]

# model_list = []
# for path in object_path_list:
#     with open(os.path.join(path, "objects_conf.json"),'r'  ) as f:
#         model_list += json.load(f)

# sampled_model_dict = {}
# for model_attr in model_list:
#     sampled_model_dict[model_attr["name"]] = model_attr


# # target_obj_name = np.random.choice(list(sampled_model_dict.keys()),1,replace=False)
# target_obj_name = list(sampled_model_dict.keys())[0]
# model_attr = sampled_model_dict[target_obj_name]
# target_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
#                         class_name = model_attr["name"],
#                         size = model_attr["size_rank"],)

# target_obj.set_rigidbody_collider()
# # target_obj.set_contact_sensor()
# target_obj.set_physics_material(
#     dynamic_friction=0.25,
#     static_friction=0.4,
#     restitution=0.0
# )


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
        
        




i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.001

my_world.stop()


# target_obj.set_pose(position = np.array([0.2,0.0,1.2]),
#                     rotation = np.array([0,0,0]) )



# world_base_tf   = rep_utils.gf_mat_to_np( rep_utils.find_parents_tf(stage.GetPrimAtPath(f"{my_robot_task.prim_path}/world_base") , include_self=True)    )
# robot_tf        = rep_utils.gf_mat_to_np( rep_utils.find_parents_tf(stage.GetPrimAtPath(my_robot.prim_path)))
# cam_tf          = rep_utils.gf_mat_to_np( rep_utils.find_parents_tf(stage.GetPrimAtPath(full_cam_path), include_self=True))
# robot_to_cam_tf = mat_utils.mat_dot([ np.linalg.inv(robot_tf), cam_tf , mat_utils.rot_x(180) ])
# robot_rot_tf_inv = np.linalg.inv( np.linalg.inv(world_base_tf).dot(robot_tf) )
# np.save("/nas/ochansol/etc/cam_tf2.npy", cam_tf)
# np.save("/nas/ochansol/etc/robot_tf2.npy", robot_tf)
# np.save("/nas/ochansol/etc/world_base_tf2.npy", world_base_tf)
# np.save("/nas/ochansol/etc/robot_rot_tf_inv2.npy", robot_rot_tf_inv)

# /home/cubox/ochansol/isaac_code/python/VLA_data_collect/OMY_dual_arm



while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and stop_flag:
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


        # if state==0:
        #     my_robot.apply_action(ArticulationAction(
        #                             joint_indices=[0,1,2,3,4,5] ,
        #                           joint_positions = [0,0,0,0,0,0]) )
        if state==0:

            start_angles = np.array([0.7888, 0.6655, 1.5500, 0.04141, 0.90969, -0.8074]) 



            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                  joint_positions = start_angles) )


        if state==1:
            if ik_first_flag:
                start_angles = np.array([0.7888, 0.6655, 1.5500, 0.04141, 0.90969, -0.8074,0,0]) / 180*np.pi
                target_pos = np.array([ 0.38076228, -0.26901221, -0.03123425])
                target_orientation = np.array([40.87287696, 23.58468836, 19.09839406])

                target_joint_positions = my_robot_task.compute_ik_traj(target_position = target_pos,
                                            target_orientation = target_orientation,
                                            frame_name = "OMY_grasp_joint",
                                            warm_start=start_angles
                                            )

                
                target_joint_positions = np.hstack((target_joint_positions[:6], 
                                                    np.array([0,0])))
                ik_first_flag =False

                print(target_pos)
                # my_world.pause()


            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5,6,7] ,
                                  joint_positions = target_joint_positions) )
            joint_states = my_robot.get_joint_positions()[:8]
            joint_err = np.abs(joint_states - target_joint_positions)
            # if np.mean(joint_err)<joint_err_th:
            #     ik_first_flag = True
                # state+=1
 
        

        if i >= 300  :
            state+=1
            i=0
            ik_first_flag = True
            # obj_reset_flag = True
        if state>=5:
            state=0

        # if target_idx >= gamja_rep.count:
        #     target_idx =0

simulation_app.close()