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
from socket_utils.client import TcpClient
from ros import isaac_ros_utils
my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()




# Robot_inst = rmp_control.Rmpflow_Robot( chunk_size=8, action_size = 7)
Robot_inst = basic_ik.BasicIk( chunk_size=8, action_size = 7)


full_img_cam_path = "/World/Robot/demo/full_camera"
wrist_img_cam_path = "/World/Robot/OMY_custom_no_delay/OMY/link6/wrist_camera"

import threading
import queue
import struct
import numpy as np
import cv2
import socket
from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep

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



def send_bytes(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def encode_to_jpg_bytes(img: np.ndarray, jpeg_quality: int = 80, input_format="rgba") -> bytes | None:
    if img is None:
        return None
    img = np.asarray(img)
    if img.size == 0:
        return None

    # dtype normalize
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            mx = float(np.nanmax(img)) if img.size else 0.0
            if mx <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # format -> BGR for JPEG
    if img.ndim == 3 and img.shape[-1] == 4:
        # RGBA or BGRA
        if input_format.lower() == "rgba":
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif input_format.lower() == "bgra":
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            # auto treat as RGBA
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.ndim == 3 and img.shape[-1] == 3:
        # RGB or BGR
        if input_format.lower() == "rgb":
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img  # assume BGR
    else:
        return None

    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return enc.tobytes()
class FrameSender(threading.Thread):
    def __init__(
        self,
        conn: socket.socket,
        *,
        jpeg_quality: int = 80,
        input_format="rgba",
        send_hz: float = None,   # None이면 들어오는 즉시 전송
    ):
        super().__init__(daemon=True)
        self.conn = conn
        self.jpeg_quality = jpeg_quality
        self.input_format = input_format
        self.send_hz = send_hz

        self._latest = None
        self._lock = threading.Lock()

        self.stop_evt = threading.Event()
        self.last_err = None

    # 🔥 최신 프레임만 덮어쓰기
    def push(self, frame: np.ndarray):
        with self._lock:
            self._latest = frame

    def _pop_latest(self):
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    def run(self):
        try:
            if self.send_hz is not None:
                period = 1.0 / self.send_hz
                next_t = time.perf_counter()
            else:
                period = None

            while not self.stop_evt.is_set():

                # 🔹 30Hz 같은 전송 주기 제어
                if period is not None:
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                        continue
                    next_t += period

                frame = self._pop_latest()
                if frame is None:
                    # 보낼 프레임 없으면 잠깐 쉼 (CPU 과사용 방지)
                    time.sleep(0.001)
                    continue

                jpg = encode_to_jpg_bytes(
                    frame,
                    jpeg_quality=self.jpeg_quality,
                    input_format=self.input_format,
                )

                if jpg is None:
                    continue

                send_bytes(self.conn, jpg)

        except Exception as e:
            self.last_err = e
        finally:
            try:
                self.conn.close()
            except Exception:
                pass

    def stop(self):
        self.stop_evt.set()




description = 'put the food in the box'




vla_flag = False
action_flag = True
reset_needed = False

ot = 0
i=0

import time
SEND_HZ = 60
send_period = 1.0 / SEND_HZ
next_send_time = time.perf_counter()

HOST = "0.0.0.0"
PORT = 5001
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"[SERVER] Waiting on {HOST}:{PORT}")
conn, addr = server.accept()
print("[SERVER] Connected:", addr)

sender = FrameSender(conn, jpeg_quality=70, input_format="auto", send_hz=SEND_HZ)
sender.start()




while simulation_app.is_running():
    my_world.step(render=True)
    sim_t = my_world.current_time

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
      
        
        now = time.perf_counter()

        # ⛔ 아직 보낼 시간이 아니면 skip
        if now < next_send_time:
            continue

        # 다음 전송 시간 예약 (누적 방식이 중요!)
        next_send_time += send_period

        state = Robot_inst.get_state()
        full_rgb = annotator_full.get_data()
        wrist_rgb = annotator_wrist.get_data()

        if full_rgb is not None:
            sender.push(full_rgb)
    
        # Robot_inst.set_action(np.array([]), action_type="joint")


        #### action
        if reset_needed:
            my_world.reset()
            Robot_inst.reset()
            reset_needed = False

        # Robot_inst.action_step()