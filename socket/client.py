import socket
import json
import struct
import threading
from typing import Any, Dict, Optional

# -------------------------
# Framing: length-prefixed JSON
# -------------------------
def _send_msg(conn: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(data))
    conn.sendall(header + data)

def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving")
        buf += chunk
    return buf

def _recv_msg(conn: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack("!I", header)
    body = _recv_exact(conn, length)
    return json.loads(body.decode("utf-8"))

# -------------------------
# Client
# -------------------------
class TcpClient:
    """
    Usage:
      c = RosNameTcpClient("127.0.0.1", 9000, name="/chansol/robot")
      c.connect()
      res = c.send(op="echo", data={"x": 1})
      c.close()

    - send()는 요청 보내고 응답을 return으로 바로 받음 (동기식)
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        timeout_sec: float = 5.0,
        auto_reconnect: bool = False,
    ):
        self.host = host
        self.port = port
        self.name = name
        self.timeout_sec = timeout_sec
        self.auto_reconnect = auto_reconnect

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._next_id = 1

    def connect(self) -> None:
        if self._sock is not None:
            return

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout_sec)
        s.connect((self.host, self.port))

        # Handshake
        _send_msg(s, {"type": "hello", "name": self.name})
        resp = _recv_msg(s)

        if resp.get("type") == "denied":
            s.close()
            raise PermissionError(f"Denied by server: {resp.get('reason')}")

        if resp.get("type") != "ok":
            s.close()
            raise ConnectionError(f"Unexpected handshake response: {resp}")

        # switch to blocking with timeout (keep)
        s.settimeout(self.timeout_sec)
        self._sock = s

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def send(self, op: str, data: Any = None) -> Any:
        """
        Sends a request and returns response data.
        Raises RuntimeError on server-side error.
        """
        with self._lock:
            try:
                self._ensure_connected()
                assert self._sock is not None

                req_id = self._next_id
                self._next_id += 1

                _send_msg(self._sock, {"type": "req", "id": req_id, "op": op, "data": data})
                res = _recv_msg(self._sock)

                if res.get("type") != "res" or res.get("id") != req_id:
                    raise ConnectionError(f"Mismatched response: {res}")

                if not res.get("ok", False):
                    raise RuntimeError(res.get("error", "unknown error"))

                return res.get("data")

            except (ConnectionError, OSError) as e:
                # 연결 끊김/타임아웃 등
                self.close()
                if self.auto_reconnect:
                    self.connect()
                    return self.send(op=op, data=data)
                raise e


if __name__ == "__main__":
    c = RosNameTcpClient("192.168.0.137", 9111, name="/chansol/robot")
    c.connect()

    print(c.send("echo", 
                    {
                        "hello": "world"
                    }
            )
        )


    c.close()
