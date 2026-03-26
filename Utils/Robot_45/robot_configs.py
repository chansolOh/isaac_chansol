# robot_config.py
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ..general_utils import mat_utils

@dataclass(frozen=True)
class RobotConfig:
    prim_path: str
    asset_path: str
    urdf_path: str
    description_path: str

    gripper_joint_prim_names: tuple[str, ...]
    total_joint_num: int

    curobo_yml_path : Optional[str] = None
    rmpflow_config_path : Optional[str] = None
    rrt_description_path : Optional[str] = None

    joint_opened_positions: Optional[np.ndarray] = None
    joint_closed_positions: Optional[np.ndarray] = None

    ee_link_path: Optional[str] = None
    robot_pos: Optional[np.ndarray] = None
    robot_ori: Optional[np.ndarray] = None
    robot_scale: Optional[np.ndarray] = None
    tcp_prim_name : Optional[str] = None
    extra_prim_path : Optional[str] = None

def robotis_omy_sim2real() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        extra_prim_path = "/World/Robot/OMY_custom_no_delay",
        ee_link_path="OMY/link6",
        asset_path =        "/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_no_delay.usd",

        urdf_path=          "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
        description_path=   "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.yaml",
        curobo_yml_path=    "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/Robotis_OMY_curobo.yml",
        
        gripper_joint_prim_names=("rh_r1_joint", "rh_l1"),
        total_joint_num=10,
        joint_opened_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_closed_positions=np.array([1.04, 1.04], dtype=np.float32),
        tcp_prim_name = "grasp_point"
        ## OMY_grasp_joint ( grasp inverse link name
    )

def robotis_omy_vla_demo() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        extra_prim_path = "/World/Robot/OMY_custom_no_delay",
        ee_link_path="OMY/link6",
        asset_path =        "/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis.usd",

        urdf_path=          "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
        description_path=   "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.yaml",
        
        gripper_joint_prim_names=("rh_r1_joint", "rh_l1"),
        total_joint_num=10,
        joint_opened_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_closed_positions=np.array([1.04, 1.04], dtype=np.float32),
        tcp_prim_name = "grasp_point"
        ## OMY_grasp_joint ( grasp inverse link name
    )


def robotis_omy_dual_arms() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        asset_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/dual_arms/OMY_dual.usd",

        urdf_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_dual.urdf",
        description_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.yaml",

        # ee_link_path="OMY_custom/OMY/link6",
        gripper_joint_prim_names=None,#("rh_r1_joint", "rh_l1"),
        total_joint_num=10,
        joint_opened_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_closed_positions=np.array([1.04, 1.04], dtype=np.float32),
    )

def robotis_omy() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        asset_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/OMY_custom.usd",

        urdf_path=          "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
        description_path=   "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_new.yaml",

        rrt_description_path = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_RRT.yaml",
        curobo_yml_path=    "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/Robotis_OMY_curobo.yml",
        rmpflow_config_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/omy_rmpflow_config.yaml",
        
        # ee_link_path="OMY/link6",
        gripper_joint_prim_names=None,#("rh_r1_joint", "rh_l1"),
        total_joint_num=10,
        joint_opened_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_closed_positions=np.array([1.04, 1.04], dtype=np.float32),
        ## OMY_grasp_joint ( grasp inverse link name
    )

def robotis_omy_no_delay() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        asset_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/OMY_custom_no_delay_new.usd",

        urdf_path=          "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
        description_path=   "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.yaml",
        curobo_yml_path=    "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/Robotis_OMY_curobo.yml",
        
        # ee_link_path="OMY/link6",
        gripper_joint_prim_names=None,#("rh_r1_joint", "rh_l1"),
        total_joint_num=10,
        joint_opened_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_closed_positions=np.array([1.04, 1.04], dtype=np.float32),
        ## OMY_grasp_joint ( grasp inverse link name
    )

def doosan_m1013() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        asset_path="/nas/ochansol/isaac/USD/robots/manipulator/Doosan_M1013/Doosan_M1013_2026/Doosan_M1013_org.usd",

        urdf_path="/nas/ochansol/isaac/USD/robots/manipulator/Doosan_M1013/Doosan_M1013_2026/Doosan_M1013_2026.urdf",
        description_path="/nas/ochansol/isaac/USD/robots/manipulator/Doosan_M1013/Doosan_M1013_2026/Doosan_M1013_2026.yaml",
        
        gripper_joint_prim_names=None,
        total_joint_num=6,
        # robot_ori=mat_utils.euler_to_quat(np.array([0.0, 0.0, 180.0]), degrees=True),
        ## Robotiq_2f140_open  ( grasp inverse link name)
    )


def ur10() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        asset_path="/nas/ochansol/isaac/USD/robots/manipulator/UR10/ur10.usd",

        urdf_path="/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur10/ur10_robot.urdf",
        description_path="/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
        curobo_yml_path = "/nas/ochansol/isaac/USD/robots/manipulator/UR10/ur10e_curobo.yml",

        gripper_joint_prim_names=None,
        total_joint_num=6,
        # robot_ori=mat_utils.euler_to_quat(np.array([0.0, 0.0, 180.0]), degrees=True),
        ## Robotiq_2f140_open  ( grasp inverse link name)
    )

def ur5e() -> RobotConfig:
    return RobotConfig(
        prim_path="/World/Robot",
        asset_path="/nas/ochansol/isaac/USD/robots/manipulator/UR5e/ur5e.usd",

        urdf_path="/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur5e/ur5e.urdf",
        description_path="/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/universal_robots/ur5e/rmpflow/ur5e_robot_description.yaml",
        
        gripper_joint_prim_names=None,
        total_joint_num=6,
        # robot_ori=mat_utils.euler_to_quat(np.array([0.0, 0.0, 180.0]), degrees=True),
        ## Robotiq_2f140_open  ( grasp inverse link name)
    )


ROBOT_CONFIGS = {
    "Robotis_OMY_Dual_Arms": robotis_omy_dual_arms,
    "Robotis_OMY": robotis_omy,
    "Robotis_OMY_no_delay": robotis_omy_no_delay,
    "Doosan_M1013": doosan_m1013,
    "Robotis_OMY_Sim2Real": robotis_omy_sim2real,
    "Robotis_OMY_VLA_Demo": robotis_omy_vla_demo,
    "UR10" : ur10,
    "UR5e" : ur5e
}