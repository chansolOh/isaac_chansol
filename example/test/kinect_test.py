import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from pyk4a import PyK4A, Config
from pyk4a import ColorResolution, DepthMode, FPS


DEPTH_PURPOSE = "CALIBRATION_CameraPurposeDepth"
COLOR_PURPOSE = "CALIBRATION_CameraPurposePhotoVideo"


def load_calibration_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def pick_camera(calib: Dict[str, Any], purpose: str) -> Dict[str, Any]:
    cams = calib.get("CalibrationInformation", {}).get("Cameras", [])
    for cam in cams:
        if cam.get("Purpose") == purpose:
            return cam
    raise KeyError(f"Camera with Purpose={purpose} not found.")


def parse_intrinsics(cam: Dict[str, Any]) -> Dict[str, Any]:
    intr = cam["Intrinsics"]
    mp = intr["ModelParameters"]  # len can be 14

    if len(mp) < 14:
        raise ValueError(f"ModelParameters length={len(mp)} (expected >= 14)")

    w = int(cam["SensorWidth"])
    h = int(cam["SensorHeight"])

    # normalized -> pixel
    cx_n, cy_n, fx_n, fy_n = mp[0:4]
    cx = cx_n * w
    cy = cy_n * h
    fx = fx_n * w
    fy = fy_n * h

    k1, k2, k3, k4, k5, k6 = mp[4:10]
    # In your JSON codx,cody are present but are 0,0
    codx, cody = mp[10:12]
    p2, p1 = mp[12:14]

    return {
        "w": w, "h": h,
        "cx": cx, "cy": cy, "fx": fx, "fy": fy,
        "k1": k1, "k2": k2, "k3": k3, "k4": k4, "k5": k5, "k6": k6,
        "p1": p1, "p2": p2,
        "codx": codx, "cody": cody,
        "metric_radius": cam.get("MetricRadius", None),
        "model_type": intr.get("ModelType", None),
        "skew": 0.0,  # Azure Kinect는 skew를 별도로 제공하지 않음 -> 0 취급
    }


def print_intrinsics(tag: str, p: Dict[str, Any]) -> None:
    print(f"\n=== {tag} ===")
    print(f"resolution: {p['w']} x {p['h']}")
    print(f"fx={p['fx']:.6f}, fy={p['fy']:.6f}, cx={p['cx']:.6f}, cy={p['cy']:.6f}, skew={p['skew']:.1f}")
    print("distortion:",
          f"k1={p['k1']:.6e}, k2={p['k2']:.6e}, k3={p['k3']:.6e}, "
          f"k4={p['k4']:.6e}, k5={p['k5']:.6e}, k6={p['k6']:.6e}, "
          f"p1={p['p1']:.6e}, p2={p['p2']:.6e}, codx={p['codx']:.6e}, cody={p['cody']:.6e}")
    print(f"model_type: {p['model_type']}")
    print(f"metric_radius: {p['metric_radius']}")


def try_set_color_controls(k4a: PyK4A,
                           wb_kelvin: Optional[int] = 4500,
                           exposure_us: Optional[int] = 8000,
                           auto_wb: bool = False,
                           auto_exposure: bool = False) -> None:
    """
    pyk4a 버전별로 color control API/enum이 다를 수 있어서:
    - 심볼이 있으면 적용
    - 없으면 '지원 안함' 출력
    """
    dev = getattr(k4a, "device", None)
    if dev is None:
        print("\n[controls] pyk4a: k4a.device not found -> cannot set color controls in this version.")
        return

    set_fn = getattr(dev, "set_color_control", None)
    if set_fn is None:
        print("\n[controls] pyk4a: device.set_color_control not found -> cannot set color controls in this version.")
        return

    # enum들을 동적으로 가져오기 (버전차 흡수)
    try:
        from pyk4a import ColorControlCommand as Cmd
        from pyk4a import ColorControlMode as Mode
    except Exception:
        # 다른 이름일 수 있음
        print("\n[controls] pyk4a: ColorControlCommand/Mode enums not found. "
              "Run a quick dir(pyk4a) check and tell me the names, I'll patch exactly.")
        return

    def _mode(auto: bool):
        return Mode.AUTO if auto else Mode.MANUAL

    # White Balance
    if wb_kelvin is not None:
        try:
            set_fn(Cmd.WHITEBALANCE, _mode(auto_wb), 0 if auto_wb else int(wb_kelvin))
            print(f"[controls] WHITEBALANCE set: {'AUTO' if auto_wb else str(wb_kelvin)+'K'}")
        except Exception as e:
            print(f"[controls] WHITEBALANCE set failed: {e}")

    # Exposure (microseconds)
    if exposure_us is not None:
        try:
            set_fn(Cmd.EXPOSURE_TIME_ABSOLUTE, _mode(auto_exposure), 0 if auto_exposure else int(exposure_us))
            print(f"[controls] EXPOSURE set: {'AUTO' if auto_exposure else str(exposure_us)+'us'}")
        except Exception as e:
            print(f"[controls] EXPOSURE set failed: {e}")


def main():
    # 네가 쓰고 싶은 모드로 바꾸면 됨
    cfg = Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30,
    )

    k4a = PyK4A(cfg)
    k4a.start()

    # (A) 컨트롤 먼저 적용하고 싶으면 여기서
    try_set_color_controls(
        k4a,
        wb_kelvin=4500,      # None이면 건드리지 않음
        exposure_us=8000,    # None이면 건드리지 않음
        auto_wb=False,
        auto_exposure=False,
    )

    # (B) 캘리브레이션 저장 + 파싱
    out = Path("k4a_calibration.json")
    k4a.save_calibration_json(str(out))

    k4a.stop()

    calib = load_calibration_json(out)

    depth_cam = pick_camera(calib, DEPTH_PURPOSE)
    color_cam = pick_camera(calib, COLOR_PURPOSE)

    depth = parse_intrinsics(depth_cam)
    color = parse_intrinsics(color_cam)

    print_intrinsics("DEPTH (D0)", depth)
    print_intrinsics("COLOR (PV0)", color)


if __name__ == "__main__":
    main()
