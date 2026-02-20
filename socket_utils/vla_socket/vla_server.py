# vla_rpc_server.py
# - TCP RPC server (class-based) import-friendly
# - Protocol: [4B total_len][4B header_len][header_json][blob_payload...]
# - Request (NEW):
#     header["obs"]         : dict (JSON, user-customizable)
#     header["image_order"] : list[str]
#     header["images"]      : dict[name] -> {"length": int}
#     blob_payload          : concatenated JPEG bytes in image_order
# - Response (same as before):
#     header_resp["action"] : {"offset":0,"length":...,"shape":[A]}
#     blob_payload          : action float32 bytes

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import cv2

from .rpc_proto import recv_message, send_message


@dataclass
class VLAServerConfig:
    host: str = "0.0.0.0"
    port: int = 1823
    backlog: int = 1
    tcp_nodelay: bool = True
    decode_jpeg: bool = True  # True면 JPEG->BGR로 디코드해서 infer_fn에 넘김


# infer_fn signature (UPDATED for new request format):
#   infer_fn(images_dict, obs_dict, action_type) -> np.ndarray
# where:
#   - images_dict: Dict[str, np.ndarray] (BGR) if decode_jpeg=True else Dict[str, bytes] (jpeg bytes)
#   - obs_dict: Dict[str, Any]  (JSON dict 그대로)
#   - action_type: str
InferFn = Callable[[Dict[str, Any], Dict[str, Any], str], np.ndarray]


class VLARpcServer:
    """
    Import-friendly server.
    - start_forever(): blocking accept loop, reconnectable
    - serve_client(): handles one client until disconnect
    """

    def __init__(self, cfg: VLAServerConfig, infer_fn: InferFn):
        self.cfg = cfg
        self.infer_fn = infer_fn
        self._srv: Optional[socket.socket] = None

    def start_forever(self) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.cfg.host, self.cfg.port))
        self._srv.listen(self.cfg.backlog)

        print(f"[SERVER] listening on {self.cfg.host}:{self.cfg.port}")

        while True:
            print("[SERVER] waiting for client...")
            conn, addr = self._srv.accept()
            try:
                self._configure_conn(conn)
                self.serve_client(conn, addr)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    def close(self) -> None:
        try:
            if self._srv is not None:
                self._srv.close()
        finally:
            self._srv = None

    def _configure_conn(self, conn: socket.socket) -> None:
        if self.cfg.tcp_nodelay:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def serve_client(self, conn: socket.socket, addr) -> None:
        print(f"[SERVER] connected: {addr}")

        try:
            while True:
                msg = recv_message(conn)
                if msg is None:
                    print(f"[SERVER] client closed: {addr}")
                    return

                header, blob_payload = msg

                if header.get("type") != "request":
                    continue

                req_id = header.get("req_id", 0)
                action_type = header.get("action_type", "joint")

                # ---- obs: JSON dict 그대로 ----
                obs: Dict[str, Any] = header.get("obs", {})

                # ---- images: dict 방식 ----
                image_order: List[str] = header.get("image_order", [])
                images_meta: Dict[str, Any] = header.get("images", {})

                images_out: Dict[str, Any] = {}
                offset = 0

                for name in image_order:
                    meta = images_meta.get(name)
                    if not meta or "length" not in meta:
                        continue

                    length = int(meta["length"])
                    if length <= 0:
                        continue

                    jpg = blob_payload[offset : offset + length]
                    offset += length

                    if self.cfg.decode_jpeg:
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        images_out[name] = bgr
                    else:
                        images_out[name] = jpg

                # ---- inference ----
                t0 = time.perf_counter()
                action = self.infer_fn(images_out, obs, action_type)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                action = np.asarray(action, dtype=np.float32).reshape(-1)
                action_bytes = action.tobytes(order="C")

                header_resp: Dict[str, Any] = {
                    "type": "response",
                    "req_id": req_id,
                    "action_type": action_type,
                    "infer_ms": infer_ms,
                    "action": {
                        "offset": 0,
                        "length": len(action_bytes),
                        "shape": [int(action.size)],
                    },
                }
                send_message(conn, header_resp, [action_bytes])

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            print(f"[SERVER] disconnected: {addr} ({type(e).__name__})")
            return


# -------------------------
# Optional: runnable example
# -------------------------
def _dummy_infer(images: Dict[str, Any], obs: Dict[str, Any], action_type: str) -> np.ndarray:
    # 예: obs["joint"] 있으면 그대로 반환, 없으면 7 zeros
    if action_type == "joint":
        j = obs.get("joint", None)
        if j is None:
            return np.zeros((7,), dtype=np.float32)
        return np.asarray(j, dtype=np.float32).reshape(-1)

    # eepose 예시
    p = obs.get("ee_pose", None)
    if p is None:
        return np.zeros((7,), dtype=np.float32)
    out = np.asarray(p, dtype=np.float32).reshape(-1)
    if out.size > 0:
        out[0] += 0.01
    return out


if __name__ == "__main__":
    cfg = VLAServerConfig(host="0.0.0.0", port=1823, decode_jpeg=True)
    server = VLARpcServer(cfg, infer_fn=_dummy_infer)
    try:
        server.start_forever()
    finally:
        server.close()
