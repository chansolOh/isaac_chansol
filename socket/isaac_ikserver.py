

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
from isaacsim.core.api import World
import numpy as np


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.Robot_45.robot_policy import My_Robot_Task as Robot_task
from Utils.Robot_45.robot_configs import ROBOT_CONFIGS



from typing import Any, Dict, Callable, Optional, Set, Tuple
from typing_extensions import override


from server import TcpServer

class IK_isaac:
    def __init__(self):


        self.my_world = World(stage_units_in_meters=1.0,
                        physics_dt  = 0.01,
                        rendering_dt = 0.01)

        self.Robot_Cfg = ROBOT_CONFIGS["Doosan_M1013"]()
        self.my_robot_task = Robot_task(robot_config=self.Robot_Cfg, name="robot_task" )
        self.my_world.add_task(self.my_robot_task)
        self.my_world.reset()

        self.my_robot = self.my_robot_task._robot


    def get_ik(self, target_pos = np.array([0.3,0.3,0.3]), target_ori = np.array([0,0,0]), frame_name="Robotiq_2f140_open", return_traj=False):
        if type(target_pos) == type([]):
            target_pos = np.array(target_pos)
        if type(target_ori) == type([]):
            target_ori = np.array(target_ori)


        target_joint_positions = self.my_robot_task.compute_ik_traj(target_position = target_pos,
                                            target_orientation = np.array([180,0,0]) + target_ori,
                                            frame_name = frame_name,
                                            return_traj=return_traj
                                            )
        deg = np.round(np.array(target_joint_positions)/np.pi*180,3)

        return deg
    



class IsaacIkServer(TcpServer):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        allowed_names: Optional[Set[str]] = None,
        handler: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        super().__init__(host, port, allowed_names, handler)
        self.ik_isaac = IK_isaac()

    @override
    def default_handler(self, request: Dict[str, Any]) -> Any:
        op = request.get("op")
        data = request.get("data")

        if op == "get_ik":
            target_pos = np.array(data["target_pos"])
            target_ori = np.array(data["target_ori"])
            frame_name = data.get("frame_name", "Robotiq_2f140_open")
            return_traj = data.get("return_traj", False)
            joint_positions = self.ik_isaac.get_ik(target_pos=target_pos, 
                                 target_ori=target_ori, 
                                 frame_name=frame_name,
                                 return_traj=return_traj
                                 )

            return {"joint_positions": joint_positions.tolist()}
        


if __name__ == "__main__":
    allowed = {
        "/chansol/robot",
        "chansol"
    }
    server = IsaacIkServer(host="192.168.0.137", port=9111, allowed_names=allowed)
    server.start()
