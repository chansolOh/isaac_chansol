import socket
import time
from typing import Any, List, Dict

import numpy as np
import cv2
import threading

from .rpc_proto import send_message, recv_message, pack_float32_array, slice_blob, unpack_float32_array


SERVER_IP = "127.0.0.1"
PORT = 1823


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return enc.tobytes()


class VLAClient:
    def __init__(self, server_ip: str, port: int):
        self.server_ip = server_ip
        self.port = port
        self.sock: socket.socket | None = None
        self.req_id = 0
        self.action = None

        self._lock = threading.Lock()
        self._latest_images = None
        self._latest_obs = None
        self._latest_action_type = "joint"
        self._has_new = False

        self._stop_evt = threading.Event()
        self._th = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[CLIENT] connected {self.server_ip}:{self.port}")

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None


    def infer(self, images_bgr: Dict[str, np.ndarray], obs: Dict[str, Any], action_type: str = "joint") -> np.ndarray:
        if self.sock is None:
            raise RuntimeError("not connected")

        self.req_id += 1
        req_id = self.req_id

        # 1) images_bgr(dict) -> jpeg bytes dict
        #    (dict는 순서가 유지되지만, 서버/클라 호환 위해 order를 명시해두는게 안전)
        image_order: List[str] = list(images_bgr.keys())
        img_bytes_dict: Dict[str, bytes] = {
            name: encode_jpeg(images_bgr[name], quality=80) for name in image_order
        }

        # 2) blobs는 "이미지들만" 순서대로 (obs는 JSON header에 바로 넣음)
        blobs: List[bytes] = [img_bytes_dict[name] for name in image_order]

        # 3) header: obs는 dict 그대로, images는 dict(name -> length) + image_order
        header = {
            "type": "request",
            "req_id": req_id,
            "action_type": action_type,
            "obs": obs,
            "image_order": image_order,
            "images": {name: {"length": len(img_bytes_dict[name])} for name in image_order},
        }

        # 4) send / recv 는 기존 그대로
        send_message(self.sock, header, blobs)

        resp = recv_message(self.sock)
        if resp is None:
            raise ConnectionError("server closed")

        header_r, blob_payload = resp
        if header_r.get("type") != "response" or header_r.get("req_id") != req_id:
            raise RuntimeError("bad response")

        a_meta = header_r["action"]
        a_blob = slice_blob(blob_payload, a_meta["offset"], a_meta["length"])
        self.action = unpack_float32_array(a_blob, tuple(a_meta["shape"]))  # (A,)

        # return action
    def _infer_loop(self, hz: float = None):
        period = (1.0 / hz) if hz else None
        next_t = time.perf_counter()

        while not self._stop_evt.is_set():
            # 주기 제한
            if period is not None:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now)
                    continue
                next_t += period

            # 최신 입력 가져오기 (쌓임 방지)
            with self._lock:
                if not self._has_new:
                    images = None
                else:
                    images = self._latest_images
                    obs = self._latest_obs
                    action_type = self._latest_action_type
                    self._has_new = False

            if images is None:
                time.sleep(0.001)
                continue

            # infer 실행 (결과는 self.action에 저장 + return도 됨)
            try:
                self.infer(images, obs, action_type=action_type)
            except Exception:
                # 필요하면 여기서 로그/재연결 처리
                pass
    def push(self, images_bgr: Dict[str, np.ndarray], obs: Dict[str, Any], action_type: str = "joint"):
        with self._lock:
            self._latest_images = images_bgr
            self._latest_obs = obs
            self._latest_action_type = action_type
            self._has_new = True

    def start_infer_thread(self, hz: float = None):
        if self._th is not None and self._th.is_alive():
            return
        self._stop_evt.clear()
        self._th = threading.Thread(target=self._infer_loop, args=(hz,), daemon=True)
        self._th.start()


def main():
    client = VLAClient(SERVER_IP, PORT)
    client.connect()

    try:
        # 테스트용 이미지/obs
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = img1.copy()
        cv2.putText(img1, "hello", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        img1 = cv2.resize(img1, (320, 240))
        img2 = cv2.resize(img2, (320, 240))

        obs = {"joint_state": [0.0] * 7}
        client.start_infer_thread(hz=25)

        while True:
            t0 = time.perf_counter()

            client.push(
                images_bgr={
                    "full": img1,
                    "wrist": img2,
                },
                obs=obs,
                action_type="joint",
            )

            dt = (time.perf_counter() - t0) * 1000.0
            print("action:", client.action, f"roundtrip {dt:.1f} ms")

            time.sleep(0.01)  # 25Hz 정도로

    finally:
        client.close()


if __name__ == "__main__":
    main()
