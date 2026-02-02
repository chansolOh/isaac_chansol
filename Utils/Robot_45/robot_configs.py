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

    # ee_link_path: str
    gripper_joint_prim_names: tuple[str, ...]
    total_joint_num: int

    joint_opened_positions: Optional[np.ndarray] = None
    joint_closed_positions: Optional[np.ndarray] = None


    robot_pos: Optional[np.ndarray] = None
    robot_ori: Optional[np.ndarray] = None
    robot_scale: Optional[np.ndarray] = None



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

        urdf_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
        description_path="/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.yaml",
        
        # ee_link_path="OMY/link6",
        gripper_joint_prim_names=("rh_r1_joint", "rh_l1"),
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


ROBOT_CONFIGS = {
    "Robotis_OMY_Dual_Arms": robotis_omy_dual_arms,
    "Robotis_OMY": robotis_omy,
    "Doosan_M1013": doosan_m1013,
}