import threading
import struct
import numpy as np
import cv2
import socket
import time


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
        fmt = input_format.lower()
        if fmt == "rgba":
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif fmt == "bgra":
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            # auto treat as RGBA
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    elif img.ndim == 3 and img.shape[-1] == 3:
        fmt = input_format.lower()
        if fmt == "rgb":
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
    """
    - 내부에서 TCP server listen/accept를 유지
    - 연결되면 최신 프레임만 send_hz로 전송
    - 연결 끊기면 자동으로 다시 accept 대기
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 5001,
        jpeg_quality: int = 80,
        input_format: str = "rgba",
        send_hz: float = 30.0,  # None이면 "가능한 빨리"(비추)
        backlog: int = 1,
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.backlog = backlog

        self.jpeg_quality = jpeg_quality
        self.input_format = input_format
        self.send_hz = send_hz

        self._latest = None
        self._lock = threading.Lock()

        self.stop_evt = threading.Event()
        self.last_err = None

        self._server_sock: socket.socket | None = None

    def push(self, frame: np.ndarray):
        with self._lock:
            self._latest = frame

    def _pop_latest(self):
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    def _make_server(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(self.backlog)
        self._server_sock = srv
        print(f"[SERVER] listening on {self.host}:{self.port}")

    def stop(self):
        self.stop_evt.set()
        # accept() 깨우기 위해 서버 소켓 닫기
        try:
            if self._server_sock is not None:
                self._server_sock.close()
        except Exception:
            pass

    def run(self):
        try:
            self._make_server()

            period = (1.0 / self.send_hz) if self.send_hz else None

            while not self.stop_evt.is_set():
                # 1) accept (끊기면 다시 여기로 돌아옴)
                try:
                    print("[SERVER] waiting for client...")
                    conn, addr = self._server_sock.accept()
                except OSError:
                    # stop()에서 server socket 닫으면 여기로 옴
                    break

                print("[SERVER] connected:", addr)

                try:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    next_t = time.perf_counter() if period else None

                    # 2) connected loop
                    while not self.stop_evt.is_set():
                        if period is not None:
                            now = time.perf_counter()
                            if now < next_t:
                                time.sleep(next_t - now)
                                continue
                            next_t += period

                        frame = self._pop_latest()
                        if frame is None:
                            time.sleep(0.001)
                            continue

                        jpg = encode_to_jpg_bytes(
                            frame,
                            jpeg_quality=self.jpeg_quality,
                            input_format=self.input_format,
                        )
                        if jpg is None:
                            continue

                        send_bytes(conn, jpg)

                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    print("[SERVER] client disconnected:", addr)

                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

        except Exception as e:
            self.last_err = e

        finally:
            try:
                if self._server_sock is not None:
                    self._server_sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    sender = FrameSender(host="0.0.0.0", port=1823, jpeg_quality=70, input_format="rgba", send_hz=60)
    sender.start()

    # 테스트: 4채널 RGBA 프레임 넣기
    test = np.zeros((480, 640, 4), dtype=np.uint8)
    test[..., 0] = 255  # R
    sender.push(test)

    # 메인 스레드가 종료되지 않게 대기
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sender.stop()
