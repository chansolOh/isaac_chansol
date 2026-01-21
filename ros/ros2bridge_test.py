import os
ISAACSIM_BIN = "/home/uon/ochansol/isaac_code/isaac_chansol/.venv/bin/isaacsim"
JAZZY_LIB = "/home/uon/ochansol/isaac_code/isaac_chansol/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"

env = os.environ.copy()

# ROS2 Bridge용 환경
env["ROS_DISTRO"] = "jazzy"
env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

# # LD_LIBRARY_PATH 안전하게 추가
# env["LD_LIBRARY_PATH"] = (
#     f"{env['LD_LIBRARY_PATH']}:{JAZZY_LIB}"
#     if "LD_LIBRARY_PATH" in env
#     else JAZZY_LIB
# )

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World
import numpy as np


from isaacsim.core.utils import extensions
extensions.enable_extension("isaacsim.ros2.bridge")




import omni.graph.core as og

GRAPH_PATH = "/World/ActionGraph_ROS2_ServiceServer"
keys = og.Controller.Keys

# 노드 타입(5.1 문서 기준)
TYPE_TICK     = "omni.graph.action.OnPlaybackTick"
TYPE_CONTEXT  = "isaacsim.ros2.bridge.ROS2Context"
TYPE_REQ      = "isaacsim.ros2.bridge.OgnROS2ServiceServerRequest"
TYPE_RESP     = "isaacsim.ros2.bridge.OgnROS2ServiceServerResponse"

SERVICE_NAME = "/service_name"  # ROS에서 이 이름으로 call
PKG, SUBFOLDER, NAME = "std_srvs", "srv", "SetBool"

og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("tick",   TYPE_TICK),
            ("ctx",    TYPE_CONTEXT),
            ("req",    TYPE_REQ),
            ("resp",   TYPE_RESP),
        ],
        keys.SET_VALUES: [
            # request/response 노드에 동일한 service type 지정 (필수) :contentReference[oaicite:3]{index=3}
            ("req.inputs:serviceName", SERVICE_NAME),
            ("req.inputs:messagePackage", PKG),
            ("req.inputs:messageSubfolder", SUBFOLDER),
            ("req.inputs:messageName", NAME),

            ("resp.inputs:messagePackage", PKG),
            ("resp.inputs:messageSubfolder", SUBFOLDER),
            ("resp.inputs:messageName", NAME),

            # SetBool 응답값(이 입력 포트들은 서비스 타입 지정 후 동적으로 생김) :contentReference[oaicite:4]{index=4}
            ("resp.inputs:success", True),
            ("resp.inputs:message", "Hello from Isaac Sim ROS2 Bridge!"),
        ],
        keys.CONNECT: [
            # tick → req 실행 (매 프레임 요청 수신 체크)
            ("tick.outputs:tick", "req.inputs:execIn"),

            # context → req/resp
            ("ctx.outputs:context", "req.inputs:context"),
            ("ctx.outputs:context", "resp.inputs:context"),

            # req <-> resp 연결 규칙 (문서 그대로) :contentReference[oaicite:5]{index=5}
            ("req.outputs:serverHandle", "resp.inputs:serverHandle"),
            ("req.outputs:onReceived",   "resp.inputs:onReceived"),
        ],
    }
)

print("✅ Action Graph created:", GRAPH_PATH)
print("Now press ▶ Play, then call the service from ROS2.")
