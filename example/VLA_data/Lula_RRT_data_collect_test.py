
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
from isaacsim.core.utils.types import ArticulationAction

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")

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
import Utils.isaac_utils_51.rep_utils as csr

from Utils.isaac_utils_51.debug_tools import debug_draw_lines, debug_draw_obb, debug_draw_points, debug_draw_clear

from scipy.spatial.transform import Rotation as R
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("zero vector")
    return v / n


def rotation_between_vectors_to_quat(a, b):
    """
    a를 b로 보내는 회전을 quaternion [x, y, z, w] 로 반환
    """
    a = normalize(a)
    b = normalize(b)

    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if np.isclose(dot, 1.0):
        return np.array([0.0, 0.0, 0.0, 1.0])  # identity quat

    if np.isclose(dot, -1.0):
        # 180도 회전: a와 수직인 축 하나 선택
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis = normalize(axis)
        return R.from_rotvec(axis * np.pi).as_quat()

    axis = normalize(cross)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle).as_quat()



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

platform_area_prims = csr.find_target_name(env_prim,["Mesh"],"platform_area")
platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == "demo"][0]

platform_path = platform_area_prims.GetPath().__str__()
platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])

my_world.reset()

platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_rep.set_tf(platform_tf)
platform_rep.set_scale(platform_scale)

# csr.scatter_in_platform_area_spread(platform_rep, obj_rep_all_list, fixed_first = False, rotation=[["x","y","z"],["z"]])

picking_rep = sampled_model_dict["apple"]["rep"]
place_rep = sampled_model_dict["box_magenta"]["rep"]
box_obb = place_rep.get_init_obb()
box_x, box_y, box_z = box_obb.max(0)- box_obb.min(0)
obstacle = VisualCuboid("/World/Wall", 
                        position = np.array([0,0,0]), 
                        size = 1, 
                        scale = np.array([box_x,box_y, box_z]),
                        visible=False)

#Initialize an RRT object
rrt = RRT(
    robot_description_path = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_RRT.yaml",#rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
    urdf_path = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",#rmp_config_dir + "/franka/lula_franka_gen.urdf",
    rrt_config_path = "/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/omy/planner_config.yaml",
    end_effector_frame_name = "OMY_grasp_joint"
)

rrt.add_obstacle(obstacle)
rrt.set_max_iterations(8000)
path_planner_visualizer = PathPlannerVisualizer(my_robot, rrt)



frame_counter = 0
plan = None
count = 0
stage = 0
current_time = 0
rrt_flag = True
rrt_atempt_count = 0
compute_target_flag = True
apple_pos_buffer = []

stop_flag = False
target_pos = np.zeros(3)
target_quat = np.zeros(4)

cam_vec = np.array([0,-1,0])


my_world.reset()
my_world.pause()


csr.scatter_in_platform_area_spread(platform_rep, obj_rep_all_list, fixed_first = False, rotation=[["x","y","z"],["z"]])
while True:
    my_world.step(render=True)
    if rrt_atempt_count > 5:
        my_world.stop()

    if my_world.is_stopped() and stop_flag:
        i=0
        stage=0
        obj_reset_flag = True
        stop_flag = False
        plan = None
        rrt_flag = True
        rrt_atempt_count = 0
        frame_counter = 0
        compute_target_flag = True

        my_world.reset()
        # my_world.pause()
        csr.scatter_in_platform_area_spread(platform_rep, obj_rep_all_list, fixed_first = False, rotation=[["x","y","z"],["z"]])



    if my_world.is_playing():
        stop_flag=True

        current_time = my_world.current_time

        # import pdb; pdb.set_trace()
        picking_pose = picking_rep.get_world_pose()
        place_pose = place_rep.get_world_pose()
        picking_pos, picking_quat = np.array(picking_pose["translation"]), np.array(picking_pose["rotation"])
        place_pos, place_quat = np.array(place_pose["translation"]), np.array(place_pose["rotation"])
        obstacle.set_world_pose(position=place_pos,orientation=place_quat)
        ## go to picking point
        print("stage : ", stage)


        if stage ==0:
            # if compute_target_flag:
            #     ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            #     target_pos, target_euler = ee_pos+np.random.uniform([-0.01, -0.05, -0.15],[0.03,0.05, -0.05 ]), ee_euler + np.random.uniform(-6,6, size=3)
            #     compute_target_flag = False

            #     my_robot.apply_action(ArticulationAction(
            #                         joint_indices=[6,7] ,
            #                       joint_positions = [0,0]) )
            #     joint_positions = my_robot_task.compute_ik_traj(target_position = target_pos,
            #                                 target_orientation = target_euler,
            #                                 frame_name = "OMY_grasp_joint",
            #                                     )
            #     my_robot.apply_action(ArticulationAction(
            #                         joint_indices=[0,1,2,3,4,5] ,
            #                       joint_positions = joint_positions[:6]) )




            if len(apple_pos_buffer)<30:
                apple_pos_buffer.append(picking_pos)
            else:
                apple_pos_buffer.pop(0)
                apple_pos_buffer.append(picking_pos)

                if np.std(apple_pos_buffer, axis=0).mean()<0.0001:
                    stage += 1
                    rrt_flag = True
                    plan = None
                    compute_target_flag = True
                    apple_pos_buffer = []
                    rrt_atempt_count=0
                # print("apple pos buffer std : ", np.std(apple_pos_buffer, axis=0))





        if stage ==1:
            ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            if compute_target_flag:
                target_pos, target_quat = picking_pos, picking_quat

                target_ori_vec = target_pos - ee_pos
                pre_step_pos = ee_pos + target_ori_vec*0.3

                ee_r = R.from_euler('xyz', ee_euler, degrees=True)
                cam_vec_rotated = ee_r.apply(cam_vec)
                pre_step_quat = rotation_between_vectors_to_quat(normalize(cam_vec_rotated), normalize(target_ori_vec))
                pre_step_quat = (R.from_quat(pre_step_quat) * ee_r).as_quat()[[3,0,1,2]]
                compute_target_flag = False
                view_pos = pre_step_pos
                view_quat = pre_step_quat

            if rrt_flag:
                my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [0,0]) )
                rrt.set_end_effector_target(pre_step_pos, pre_step_quat)
                rrt.update_world()
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                if plan: actions = my_robot_task.rrt_plan_to_traj_actions(plan, physics_dt=0.02)
                else : actions = []
                rrt_flag = False

            pos_crit = np.abs(np.array(ee_pos) - pre_step_pos).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - pre_step_quat).sum()
            print("pos_crit : ", pos_crit, "ori_crit : ", ori_crit)
            if pos_crit<0.005 and ori_crit < 0.05:
                stage += 1
                rrt_flag = True
                plan = None
                compute_target_flag = True
                rrt_atempt_count=0

            if actions:
                action = actions.pop(0)
                my_robot.apply_action(action)
            else:
                rrt_flag = True
                rrt_atempt_count += 1








        if stage ==2:
            ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")

            
            # debug_draw_clear()
            # debug_draw_lines(np.array([ee_pos, target_pos]))
            # debug_draw_lines(np.array([ee_pos, ee_pos + cam_vec_rotated]), color=(0,1,0))
            
            if compute_target_flag:
                # target_pos, target_quat = picking_pos+np.array([0,0,0.05]), mat_utils.euler_to_quat(np.array([90,0,90]), degrees=True)
                # compute_target_flag = False

                target_pos = picking_pos
                target_quat = mat_utils.euler_to_quat(np.array([90,0,90]), degrees=True)
                plan_list = []
                candi_yaw_list = []
                rrt.set_max_iterations(80)
                for candi_yaw in range(90-90,90+90,5):
                    rrt.set_end_effector_target(target_pos, mat_utils.euler_to_quat(np.array([90,0,candi_yaw]), degrees=True))
                    rrt.update_world()
                    plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                    if plan:
                        plan_list.append(plan)
                        candi_yaw_list.append(candi_yaw)
                if plan_list:
                    # best_plan_idx = np.argmin(np.abs(90-np.array(candi_yaw_list)))
                    target_quat = mat_utils.euler_to_quat(np.array([90,0,np.random.choice(candi_yaw_list)]), degrees=True)
                    target_pos = picking_pos + np.array([0,0,0.05])
                    # print("target_yaw : ", candi_yaw_list[best_plan_idx])
                rrt.set_max_iterations(8000)
                compute_target_flag = False
            if rrt_flag:
                my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [0,0]) )
                rrt.set_end_effector_target(target_pos, target_quat)
                rrt.update_world()
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                if plan: actions = my_robot_task.rrt_plan_to_traj_actions(plan, physics_dt=0.02)
                else : actions = []
                rrt_flag = False

            pos_crit = np.abs(np.array(ee_pos) - target_pos).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - target_quat).sum()
            print("pos_crit : ", pos_crit, "ori_crit : ", ori_crit)
            if pos_crit<0.005 and ori_crit < 0.03:
                stage += 1
                rrt_flag = True
                plan = None
                compute_target_flag = True
                rrt_atempt_count=0

            if actions:
                action = actions.pop(0)
                my_robot.apply_action(action)
            else:
                rrt_flag = True
                rrt_atempt_count += 1
        









        if stage ==3:
            ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")

            
            # debug_draw_clear()
            # debug_draw_lines(np.array([ee_pos, target_pos]))
            # debug_draw_lines(np.array([ee_pos, ee_pos + cam_vec_rotated]), color=(0,1,0))
            if compute_target_flag:
                target_pos = picking_pos 

            if rrt_flag:
                my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [0,0]) )
                rrt.set_end_effector_target(target_pos, target_quat)
                rrt.update_world()
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                if plan: actions = my_robot_task.rrt_plan_to_traj_actions(plan, physics_dt=0.02)
                else : actions = []
                rrt_flag = False

            pos_crit = np.abs(np.array(ee_pos) - target_pos).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - target_quat).sum()
            print("pos_crit : ", pos_crit, "ori_crit : ", ori_crit)
            if pos_crit<0.005 and ori_crit < 0.03:
                stage += 1
                rrt_flag = True
                plan = None
                compute_target_flag = True
                rrt_atempt_count=0

            if actions:
                action = actions.pop(0)
                my_robot.apply_action(action)
            else:
                rrt_flag = True
                rrt_atempt_count += 1


            






        ## gripper close
        elif stage ==4:
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [np.pi/4,np.pi/4]) )
            gripper_joint_idx = my_robot.get_dof_index("rh_r1_joint")
            gripper_joint_effort = my_robot.get_measured_joint_efforts(joint_indices=np.array([gripper_joint_idx]))
            if gripper_joint_effort > 0.5:
                stage += 1
                rrt_flag = True
                plan = None
                init_diff_grasp_gripper = np.abs(picking_pos - ee_pos)
            # print("gripper_joint_effort : ", gripper_joint_effort   )
            




        elif stage ==5:
            ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            diff_grasp_gripper = np.abs(picking_pos - ee_pos)
            if np.abs(init_diff_grasp_gripper - diff_grasp_gripper).sum() > 0.02:
                my_world.stop()
                print("init_diff_grasp_gripper : ", init_diff_grasp_gripper, "diff_grasp_gripper : ", diff_grasp_gripper)
                print("grasp failed, retrying...")
                continue




            if compute_target_flag:
                target_pos = (place_pos+picking_pos)/2 + np.array([0,0,0.1]) 
                # target_quat = mat_utils.euler_to_quat(np.array([90,0,90]), degrees=True)    
                compute_target_flag = False

            if rrt_flag:
                rrt.set_end_effector_target(target_pos, target_quat)
                rrt.update_world()
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                if plan: actions = my_robot_task.rrt_plan_to_traj_actions(plan, physics_dt=0.02)
                else : actions = []
                rrt_flag = False

            pos_crit = np.abs(np.array(ee_pos) - target_pos).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - target_quat).sum()
            # print("pos_crit : ", pos_crit, "ori_crit : ", ori_crit)
            if pos_crit<0.005 and ori_crit < 0.03:
                stage += 1
                rrt_flag = True
                plan = None
                compute_target_flag = True
                rrt_atempt_count=0
                grasp_steady_buffer = []

            if actions:
                action = actions.pop(0)
                my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [np.pi/4,np.pi/4]) )
                my_robot.apply_action(action)
            else:
                rrt_flag = True
                rrt_atempt_count += 1
                print("rrt attempt count : ", rrt_atempt_count)





        elif stage ==6:
            ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            diff_grasp_gripper = np.abs(picking_pos - ee_pos)
            if np.abs(init_diff_grasp_gripper - diff_grasp_gripper).sum() > 0.02:
                my_world.stop()
                print("init_diff_grasp_gripper : ", init_diff_grasp_gripper, "diff_grasp_gripper : ", diff_grasp_gripper)
                print("grasp failed, retrying...")
                continue

            if compute_target_flag:
                target_pos = place_pos + np.array([0,0,0.08])
                # target_quat = mat_utils.euler_to_quat(np.array([90,0,90]), degrees=True) 
                compute_target_flag = False

            if rrt_flag:
                rrt.set_end_effector_target(target_pos, target_quat)
                rrt.update_world()
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                if plan: actions = my_robot_task.rrt_plan_to_traj_actions(plan, physics_dt=0.02)
                else : actions = []
                rrt_flag = False

            pos_crit = np.abs(np.array(ee_pos) - target_pos).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - target_quat).sum()
            print("pos_crit : ", pos_crit, "ori_crit : ", ori_crit)
            if pos_crit<0.005 and ori_crit < 0.03:
                stage += 1
                rrt_flag = True
                plan = None
                compute_target_flag = True
                rrt_atempt_count=0

            if actions:
                action = actions.pop(0)
                my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [np.pi/4,np.pi/4]) )
                my_robot.apply_action(action)
            else:
                rrt_flag = True
                rrt_atempt_count += 1






        ## gripper open
        elif stage ==7:
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [0.0,0.0]) )
            
            gripper_joints = my_robot_task.get_joint_positions()[[6,7]]
            if np.sum(gripper_joints) < 0.001:
                stage += 1
                rrt_flag = True
                plan = None
            


        elif stage == 8:
            ee_pos , ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            if rrt_flag:
                my_robot.apply_action(ArticulationAction(
                                    joint_indices=[6,7] ,
                                  joint_positions = [0,0]) )
                rrt.set_end_effector_target(view_pos, view_quat)
                rrt.update_world()
                plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
                if plan: actions = my_robot_task.rrt_plan_to_traj_actions(plan, physics_dt=0.02)
                else : actions = []
                rrt_flag = False

            pos_crit = np.abs(np.array(ee_pos) - view_pos).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - view_quat).sum()
            print("pos_crit : ", pos_crit, "ori_crit : ", ori_crit)
            if pos_crit<0.005 and ori_crit < 0.03:
                stage += 1
                rrt_flag = True
                plan = None
                compute_target_flag = True
                rrt_atempt_count=0

            if actions:
                action = actions.pop(0)
                my_robot.apply_action(action)
            else:
                rrt_flag = True
                rrt_atempt_count += 1



        elif stage == 9:
            center_diff = np.abs(picking_pos - place_pos).sum()
            if center_diff < 0.03:
                print("success! center_diff : ", center_diff)
            else:
                print("failed... center_diff : ", center_diff)
        




        frame_counter += 1

