import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

class InteractiveGraspRect:
    def __init__(self, ax, fixed_height=20):
        self.ax = ax
        self.fixed_h = float(fixed_height)

        self.mode = 'idle'          # 'idle' -> 'drag' -> 'rotate' -> 'done'
        self.cx = self.cy = None
        self.width = None
        self.angle_deg = 0.0

        # 패치(사각형) 준비
        # Rectangle은 좌하단(left,bottom) 기준이라 중심기반으로 계산해서 넣어줄 것
        self.rect = Rectangle((0,0), 0, self.fixed_h, edgecolor='blue',fill=False)
        self.ax.add_patch(self.rect)

        # 상태 표시 텍스트
        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                va='top', ha='left')

        # 이벤트 연결
        self.cid_press  = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_move   = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_release= self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def set_rect(self, cx, cy, width, angle_deg):
        """중심(cx,cy), 폭(width), 고정높이(self.fixed_h), 회전(angle_deg)로 패치를 갱신"""
        # 좌하단 계산
        x0 = cx - width/2.0
        y0 = cy - self.fixed_h/2.0

        # 기본 위치/크기 갱신
        self.rect.set_xy((x0, y0))
        self.rect.set_width(width)
        self.rect.set_height(self.fixed_h)

        # 회전 변환 (중심 기준 회전)
        base = self.ax.transData
        rot = Affine2D().rotate_deg_around(cx, cy, angle_deg)
        self.rect.set_transform(rot + base)

        # 텍스트 업데이트
        self.txt.set_text(f"mode: {self.mode}\ncenter=({cx:.1f},{cy:.1f})\nwidth={width:.1f}\nangle={angle_deg:.1f}°")

        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if self.mode == 'idle':
            # 1) 첫 클릭: center 지정, 드래그로 width 정하는 단계로
            self.cx, self.cy = float(event.xdata), float(event.ydata)
            self.width = 1.0
            self.angle_deg = 0.0
            self.mode = 'drag'
            self.set_rect(self.cx, self.cy, self.width, self.angle_deg)

        elif self.mode == 'rotate':
            # 4) 다시 클릭하면 현재 각도 확정 → 끝
            # 클릭 순간 각도는 on_move에서 계속 업데이트되고 있음
            print(f"[final] center=({self.cx:.2f}, {self.cy:.2f}), width={self.width:.2f}, angle={self.angle_deg:.2f}°")
            self.mode = 'done'
            self.txt.set_text(self.txt.get_text() + "\n(finalized)")
            self.ax.figure.canvas.draw_idle()

    def on_move(self, event):
        if event.inaxes != self.ax:
            return

        if self.mode == 'drag' and self.cx is not None:
            # 2) 드래그한 만큼이 width
            # 풀폭(width) = center와 현재 위치 사이 거리의 2배로 정의
            dx = float(event.xdata) - self.cx
            dy = float(event.ydata) - self.cy
            half_w = np.hypot(dx, dy)
            self.width = max(1.0, 2.0 * half_w)  # 최소 1픽셀
            self.angle_deg = 0.0                 # 드래그 동안은 회전 0으로 미리보기
            self.set_rect(self.cx, self.cy, self.width, self.angle_deg)

        elif self.mode == 'rotate' and self.cx is not None:
            # 3) 마우스 움직임에 따라 중심에서의 벡터 방향을 각도로 사용
            vx = float(event.xdata) - self.cx
            vy = float(event.ydata) - self.cy
            if vx == 0 and vy == 0:
                return
            # 가로 방향(+)를 0도로 두고 반시계(수학 좌표) 기준
            ang = np.degrees(np.arctan2(vy, vx))
            self.angle_deg = ang
            self.set_rect(self.cx, self.cy, self.width, self.angle_deg)

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        if self.mode == 'drag':
            # 2) 드래그 종료 → 폭 확정, 회전 단계로 진입
            self.mode = 'rotate'
            self.set_rect(self.cx, self.cy, self.width, self.angle_deg)