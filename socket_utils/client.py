import socket
import json
import struct
import threading
import time
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
        # For async usage: latest response data is stored here (single-slot, overwrite).
        self.res: Any = None
        # Internal raw response cache used by blocking path.
        self._raw_res: Dict[int, Dict[str, Any]] = {}
        self._res_cv = threading.Condition()
        self._recv_stop = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None

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
        self._start_recv_thread()

    def close(self) -> None:
        self._recv_stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=0.2)
        self._recv_thread = None

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def _start_recv_thread(self) -> None:
        self._recv_stop.clear()
        if self._recv_thread is not None and self._recv_thread.is_alive():
            return
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _recv_loop(self) -> None:
        while not self._recv_stop.is_set():
            if self._sock is None:
                break
            try:
                msg = _recv_msg(self._sock)
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                break
            except Exception:
                break

            if msg.get("type") != "res":
                continue
            req_id = msg.get("id")
            if not isinstance(req_id, int):
                continue

            with self._res_cv:
                self._raw_res[req_id] = msg
                if msg.get("ok", False):
                    self.res = msg.get("data")
                else:
                    self.res = {"error": msg.get("error", "unknown error"), "ok": False}
                self._res_cv.notify_all()

    def _wait_response_for(self, req_id: int, timeout_sec: float) -> Dict[str, Any]:
        end_t = time.time() + timeout_sec
        with self._res_cv:
            while req_id not in self._raw_res:
                remaining = end_t - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timeout waiting response request_id={req_id}, timeout={timeout_sec}s"
                    )
                self._res_cv.wait(timeout=remaining)
            return self._raw_res.pop(req_id)

    def get_response(self, pop: bool = False) -> Optional[Any]:
        with self._res_cv:
            if pop:
                out = self.res
                self.res = None
                return out
            return self.res

    def send(
        self,
        op: str,
        data: Any = None,
        blocking: bool = True,
        recv_timeout_sec: Optional[float] = None,
    ) -> Any:
        """
        Sends a request.
        - blocking=False (default): send only and return request_id for later correlation.
        - blocking=True: waits until matching response arrives and returns response data.
        - recv_timeout_sec: timeout for waiting response when blocking=True.
        Raises RuntimeError on server-side error when blocking=True.
        """
        with self._lock:
            try:
                self._ensure_connected()
                assert self._sock is not None

                req_id = self._next_id
                self._next_id += 1

                _send_msg(self._sock, {"type": "req", "id": req_id, "op": op, "data": data})
                if not blocking:
                    return req_id

                wait_timeout = self.timeout_sec if recv_timeout_sec is None else recv_timeout_sec
                res = self._wait_response_for(req_id, wait_timeout)

                if not res.get("ok", False):
                    raise RuntimeError(res.get("error", "unknown error"))

                return res.get("data")

            except (ConnectionError, OSError) as e:
                # 연결 끊김/타임아웃 등
                self.close()
                if self.auto_reconnect:
                    self.connect()
                    return self.send(op=op, data=data, blocking=blocking, recv_timeout_sec=recv_timeout_sec)
                raise e


if __name__ == "__main__":
    c = TcpClient("192.168.0.137", 9111, name="/chansol/robot")
    c.connect()

    print(c.send("echo", 
                    {
                        "hello": "world"
                    }
            )
        )


    c.close()
