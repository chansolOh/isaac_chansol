import socket
import struct
import numpy as np
import cv2

SERVER_IP = "127.0.0.1"
PORT = 5001


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP, PORT))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[CLIENT] connected {SERVER_IP}:{PORT}")

    frames = 0

    while True:
        header = recv_exact(s, 4)
        if header is None:
            print("[CLIENT] server closed")
            break

        (length,) = struct.unpack(">I", header)
        payload = recv_exact(s, length)
        if payload is None:
            print("[CLIENT] server closed (payload)")
            break

        arr = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        cv2.imshow("Stream", frame_bgr)

        # ESC 또는 q로 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

        frames += 1
        if frames % 60 == 0:
            print("[CLIENT] frames:", frames)

    s.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
