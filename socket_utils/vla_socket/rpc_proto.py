import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def _recv_exact(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def send_message(sock, header: Dict[str, Any], blobs: List[bytes]) -> None:
    """
    One message:
      [4B total_len][4B header_len][header_json][blob0][blob1]...
    """
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)
    payload = struct.pack(">I", header_len) + header_bytes + b"".join(blobs)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_message(sock) -> Tuple[Dict[str, Any], bytes] | None:
    """
    Returns (header_dict, blob_payload_bytes) or None if socket closed.
    """
    total_len_b = _recv_exact(sock, 4)
    if total_len_b is None:
        return None
    (total_len,) = struct.unpack(">I", total_len_b)

    payload = _recv_exact(sock, total_len)
    if payload is None:
        return None

    (header_len,) = struct.unpack(">I", payload[:4])
    header_bytes = payload[4:4 + header_len]
    blob_payload = payload[4 + header_len:]
    header = json.loads(header_bytes.decode("utf-8"))
    return header, blob_payload


def pack_float32_array(arr: np.ndarray) -> bytes:
    a = np.asarray(arr, dtype=np.float32)
    return a.tobytes(order="C")


def unpack_float32_array(blob: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    a = np.frombuffer(blob, dtype=np.float32)
    return a.reshape(shape)


def slice_blob(blob_payload: bytes, offset: int, length: int) -> bytes:
    return blob_payload[offset: offset + length]
