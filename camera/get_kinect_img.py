from typing import Optional, Tuple
import numpy as np

from pyk4a import PyK4A, Config
from pyk4a import ColorResolution, DepthMode, FPS
'''
    OFF = 0
    RES_720P = 1
    RES_1080P = 2
    RES_1440P = 3
    RES_1536P = 4
    RES_2160P = 5
    RES_3072P = 6
'''
class K4ACamera:
    def __init__(
        self,
        resolution:str = "1080P",
        wb_kelvin: Optional[int] = 4500,
        exposure_us: Optional[int] = 8000,
        auto_wb: bool = False,
        auto_exposure: bool = False,
    ):
        res = {
            "OFF": ColorResolution.OFF,
            "720P": ColorResolution.RES_720P,
            "1080P": ColorResolution.RES_1080P,
            "1440P": ColorResolution.RES_1440P,
            "1536P": ColorResolution.RES_1536P,
            "2160P": ColorResolution.RES_2160P,
            "3072P": ColorResolution.RES_3072P,
        }.get(resolution.upper(), ColorResolution.RES_1080P)
        # ✅ Config는 클래스 내부에서 고정

        self.cfg = Config(
            color_resolution=res,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
        )
        self.k4a = PyK4A(self.cfg)

        self._wb_kelvin = wb_kelvin
        self._exposure_us = exposure_us
        self._auto_wb = auto_wb
        self._auto_exposure = auto_exposure
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.k4a.start()
        self._started = True
        self._try_set_color_controls()

    def stop(self) -> None:
        if not self._started:
            return
        self.k4a.stop()
        self._started = False

    def get(self, timeout_ms: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        return: (color_bgra, depth_mm)
          - color_bgra: (H,W,4) uint8
          - depth_mm:   (H,W)   uint16
        """
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")
        try:
            cap = self.k4a.get_capture(timeout_ms=timeout_ms) if timeout_ms is not None else self.k4a.get_capture()
        except TypeError:
            cap = self.k4a.get_capture()
        return cap.color, cap.transformed_depth/1000
    
    def get_rgb(self, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")
        try:
            cap = self.k4a.get_capture(timeout_ms=timeout_ms) if timeout_ms is not None else self.k4a.get_capture()
        except TypeError:
            cap = self.k4a.get_capture()
        return cap.color.take([2, 1, 0], axis=-1)#cap.color[..., :3][...,::-1]  # BGR -> RGB
    
    def get_depth(self, timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")
        try:
            cap = self.k4a.get_capture(timeout_ms=timeout_ms) if timeout_ms is not None else self.k4a.get_capture()
        except TypeError:
            cap = self.k4a.get_capture()
        return cap.transformed_depth /1000
    
    def get_rgb_depth(self, timeout_ms: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")
        try:
            cap = self.k4a.get_capture(timeout_ms=timeout_ms) if timeout_ms is not None else self.k4a.get_capture()
        except TypeError:
            cap = self.k4a.get_capture()
        rgb = cap.color.take([2, 1, 0], axis=-1)  # BGR -> RGB
        depth = cap.transformed_depth/1000
        return rgb, depth
    

    def _try_set_color_controls(self) -> None:
        dev = getattr(self.k4a, "device", None)
        if dev is None:
            return
        set_fn = getattr(dev, "set_color_control", None)
        if set_fn is None:
            return

        try:
            from pyk4a import ColorControlCommand as Cmd
            from pyk4a import ColorControlMode as Mode
        except Exception:
            return

        def _mode(auto: bool):
            return Mode.AUTO if auto else Mode.MANUAL

        if self._wb_kelvin is not None:
            try:
                set_fn(Cmd.WHITEBALANCE, _mode(self._auto_wb), 0 if self._auto_wb else int(self._wb_kelvin))
            except Exception:
                pass

        if self._exposure_us is not None:
            try:
                set_fn(
                    Cmd.EXPOSURE_TIME_ABSOLUTE,
                    _mode(self._auto_exposure),
                    0 if self._auto_exposure else int(self._exposure_us),
                )
            except Exception:
                pass


def main():
    cam = K4ACamera(resolution="1080P", wb_kelvin=4500, exposure_us=8000)  # 여기만 바꾸면 됨
    cam.start()

    # color_bgra, depth_mm = cam.get()  # ✅ get으로 깔끔하게 받기
    rgb, depth_mm = cam.get_rgb_depth()
    cam.stop()

    print("color:", None if rgb is None else (rgb.shape, rgb.dtype))
    print("depth:", None if depth_mm is None else (depth_mm.shape, depth_mm.dtype))

    import matplotlib.pyplot as plt
    plt.imshow(rgb)
    plt.show()
    plt.imshow(depth_mm)
    plt.show()
if __name__ == "__main__":
    main()
