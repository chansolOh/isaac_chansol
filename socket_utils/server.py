import socket
import threading
import json
import struct
from typing import Any, Dict, Callable, Optional, Set, Tuple

# -------------------------
# Framing: length-prefixed JSON
# -------------------------
def _send_msg(conn: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(data))  # 4-byte unsigned int, network byte order
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
# Server
# -------------------------
class TcpServer:
    """
    TCP server that only allows clients whose 'name' matches allowed_names.
    After handshake, it accepts request messages and returns responses.

    Request format:
      {"type":"req","id":123,"op":"echo","data":{...}}

    Response format:
      {"type":"res","id":123,"ok":true,"data":{...}} or {"ok":false,"error":"..."}
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        allowed_names: Optional[Set[str]] = None,
        handler: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.host = host
        self.port = port
        self.allowed_names = allowed_names or {"/chansol/robot"}
        self.handler = handler or self.default_handler
        self._sock: Optional[socket.socket] = None
        self._stop_event = threading.Event()

    def default_handler(self, request: Dict[str, Any]) -> Any:
        """
        기본 처리기 예시:
        - op=="echo": data 그대로 반환
        - op=="add":  data={"a":..,"b":..} -> a+b
        """
        op = request.get("op")
        data = request.get("data")

        if op == "echo":
            return {"echo": data}
        if op == "add":
            a = float(data["a"])
            b = float(data["b"])
            return {"sum": a + b}

        raise ValueError(f"Unknown op: {op}")

    def start(self) -> None:
        if self._sock is not None:
            raise RuntimeError("Server already started")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(50)

        print(f"[server] listening on {self.host}:{self.port}")
        try:
            while not self._stop_event.is_set():
                conn, addr = self._sock.accept()
                t = threading.Thread(target=self._client_thread, args=(conn, addr), daemon=True)
                t.start()
        finally:
            self.close()

    def close(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        print("[server] closed")

    def _client_thread(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        peer = f"{addr[0]}:{addr[1]}"
        try:
            # ---- Handshake ----
            hello = _recv_msg(conn)
            if hello.get("type") != "hello":
                _send_msg(conn, {"type": "denied", "reason": "missing hello"})
                return

            name = hello.get("name")
            if name not in self.allowed_names:
                _send_msg(conn, {"type": "denied", "reason": f"name not allowed: {name}"})
                return

            _send_msg(conn, {"type": "ok", "name": name})
            print(f"[server] accepted {peer} as {name}")

            # ---- Requests loop ----
            while True:
                req = _recv_msg(conn)  # raises on disconnect
                if req.get("type") != "req":
                    _send_msg(conn, {"type": "res", "id": req.get("id"), "ok": False, "error": "invalid type"})
                    continue

                req_id = req.get("id")
                try:
                    result = self.handler(req)
                    _send_msg(conn, {"type": "res", "id": req_id, "ok": True, "data": result})
                except Exception as e:
                    _send_msg(conn, {"type": "res", "id": req_id, "ok": False, "error": str(e)})

        except ConnectionError:
            print(f"[server] disconnected: {peer}")
        except Exception as e:
            print(f"[server] error with {peer}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    allowed = {
        "/chansol/robot",
        "chansol"
    }
    server = TcpServer(host="192.168.0.137", port=9111, allowed_names=allowed)
    server.start()
