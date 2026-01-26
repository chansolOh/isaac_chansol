import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.action import ActionClient

from dsr_msgs2.srv import MoveJoint, MoveLine, MoveStop
from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class Controller(Node):
    def __init__(self):
        super().__init__('move_test_client')

        # Services (기존)
        self.cli_movej = self.create_client(MoveJoint, '/motion/move_joint')
        self.cli_movel = self.create_client(MoveLine, '/motion/move_line')
        self.cli_stop  = self.create_client(MoveStop, '/motion/move_stop')

        # Action (추가): FollowJointTrajectory
        self.traj_ac = ActionClient(
            self,
            FollowJointTrajectory,
            '/dsr_moveit_controller/follow_joint_trajectory'
        )

        # joint state를 별도 executor로 받는 구조 (기존)
        self.state_node = rclpy.create_node('joint_state_node')
        self.state_executor = SingleThreadedExecutor()
        self.state_executor.add_node(self.state_node)
        self.state_thread = threading.Thread(target=self.state_executor.spin, daemon=True)
        self.state_thread.start()

        self.goal_handle = None
        self.effort_od = None
        self.effort = None
        self.emergency_stop_flag = False

        # 최신 joint_states 저장용
        self.joint_states = None   # np.ndarray shape (6,)
        self.joint_vel = None
        self._last_js_stamp = None

        self.joint_state_sub = self.state_node.create_subscription(
            JointState,
            "/joint_states",
            self.get_joint_state,
            10
        )

        # 서비스 대기 (기존)
        while not self.cli_movej.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for move_joint service...')
        while not self.cli_movel.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for move_line service...')
        while not self.cli_stop.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /motion/move_stop service...')

        # 액션 서버 대기 (추가)
        self.get_logger().info('Waiting for follow_joint_trajectory action server...')
        if not self.traj_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('follow_joint_trajectory action server not available (yet).')

    # ---------------------------
    # 기존 movej/movel/stop
    # ---------------------------
    def movej(self, pos, vel=50.0, acc=100.0, time=0.0, mode=0, sync_type=0):
        req = MoveJoint.Request()
        req.pos = pos
        req.vel = vel
        req.acc = acc
        req.time = time
        req.radius = 0.0
        req.mode = mode
        req.blend_type = 0
        req.sync_type = sync_type

        future = self.cli_movej.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'MoveJoint success: {future.result().success}')
        else:
            self.get_logger().error('MoveJoint service call failed.')

    def movel(self, pos, vel=[50.0, 50.0], acc=[50.0, 50.0], time=0.0, mode=0, sync_type=0):
        req = MoveLine.Request()
        req.pos = pos
        req.vel = vel
        req.acc = acc
        req.time = time
        req.radius = 0.0
        req.ref = 0
        req.mode = mode
        req.blend_type = 0
        req.sync_type = sync_type

        future = self.cli_movel.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'MoveLine success: {future.result().success}')
        else:
            self.get_logger().error('MoveLine service call failed.')

    # def stop(self, mode=1):
    #     req = MoveStop.Request()
    #     req.stop_mode = mode  # 0: Normal stop, 1: Quick stop, 2: Emergency stop
    #     future = self.cli_stop.call_async(req)
    #     rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

    #     if future.result() is not None:
    #         self.get_logger().info('Stop command executed.')
    #     else:
    #         self.get_logger().error('Failed to call /motion/move_stop.')

    def stop(self, wait: bool = False) -> bool:
        if self.goal_handle is None:
            self.get_logger().warn("No active trajectory goal.")
            return False

        future = self.goal_handle.cancel_goal_async()
        if wait:
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info("Requested trajectory cancel.")
        return True
    
    def get_joint_state(self, msg: JointState):
        pos_map = dict(zip(msg.name, msg.position))
        self.joint_states = np.array([pos_map[f'joint_{i}'] for i in range(1, 7)], dtype=np.float64)

        if msg.velocity:
            vel_map = dict(zip(msg.name, msg.velocity))
            self.joint_vel = np.array([vel_map.get(f'joint_{i}', 0.0) for i in range(1, 7)], dtype=np.float64)
        else:
            self.joint_vel = None

        if msg.effort:
            eff_map = dict(zip(msg.name, msg.effort))
            self.effort = np.array([eff_map.get(f'joint_{i}', 0.0) for i in range(1, 7)], dtype=np.float64)
        else:
            self.effort = None

        # ---------------------------
        # 추가: Trajectory Action
        # ---------------------------
        
    def send_joint_trajectory(
        self,
        points: list,                 # [[j1..j6], [j1..j6], ...]
        times_sec: list,              # 누적 시간: [5,10,15] 같은 형태
        include_start_from_current: bool = True,
        start_hold_sec: float = 0.2,  # 너무 길면 덜덜 가능 -> 0.1~0.3 추천
        wait_result: bool = True,
        stop_at_end: bool = True,     # 마지막에서 멈출지(velocity=0)
        degrees: bool = False         # 각도 단위(deg/rad)
    ):
        if not self.traj_ac.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("follow_joint_trajectory action server not available.")
            return None

        # times: float 누적
        times_sec = [float(t) for t in times_sec]
        if any(t2 <= t1 for t1, t2 in zip(times_sec[:-1], times_sec[1:])):
            self.get_logger().error("times_sec must be strictly increasing cumulative times (e.g., [5,10,15])")
            return None
        
        if degrees:
            points = [ [np.deg2rad(q) for q in pt] for pt in points ]
        # prepend current pose (권장: reject 방지 + 연속성)
        if include_start_from_current:
            t0 = time.time()
            while self.joint_states is None and (time.time() - t0) < 2.0:
                time.sleep(0.01)
            if self.joint_states is None:
                self.get_logger().error("No /joint_states received yet.")
                return None

            points = [self.joint_states.tolist()] + points
            times_sec = [float(start_hold_sec)] + [float(start_hold_sec) + t for t in times_sec]

        # velocity 자동 계산
        # v[i] = (q[i] - q[i-1]) / (t[i] - t[i-1]), v[0]=0, last=0(optional)
        q = np.array(points, dtype=np.float64)            # (N,6)
        t = np.array(times_sec, dtype=np.float64)         # (N,)
        N = q.shape[0]

        v = np.zeros_like(q)
        for i in range(1, N):
            dt = float(t[i] - t[i-1])
            if dt <= 1e-6:
                self.get_logger().error(f"Non-positive dt at segment {i-1}->{i}: {dt}")
                return None
            v[i] = (q[i] - q[i-1]) / dt

        v[0] = 0.0
        if stop_at_end:
            v[-1] = 0.0

        # goal 구성
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']

        for i in range(N):
            pt = JointTrajectoryPoint()
            pt.positions = q[i].tolist()
            pt.velocities = v[i].tolist()  # ✅ 여기서 자동 속도 적용

            sec = int(np.floor(t[i]))
            nanosec = int((t[i] - sec) * 1e9)
            pt.time_from_start.sec = sec
            pt.time_from_start.nanosec = nanosec
            goal.trajectory.points.append(pt)

        goal.goal_time_tolerance.sec = 2
        goal.goal_time_tolerance.nanosec = 0

        send_future = self.traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            return None

        self.get_logger().info("Trajectory goal accepted.")
        self.goal_handle = goal_handle

        if wait_result:
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            res = result_future.result()
            if res is not None:
                self.get_logger().info(f"Trajectory result status: {res.status}")
            return res

        return goal_handle

    def reset_robot(self, sync_type=0):
        self.movej(
            pos=[94.43, -34.29, 113.58, -4.01, 100.53, 1.71],
            vel=30.0,
            sync_type=sync_type
        )


def main(args=None):
    rclpy.init(args=args)
    robot = Controller()

    # 1) 천천히 0,0,0,0,0,0 보내기 (action trajectory)
    #    ※ 네 로봇이 deg 기반이면 아래 값도 deg로 일관되게 써야 함.
    #    지금은 20초로 아주 느리게.
    robot.send_joint_trajectory(
        points=[
            [0.0]*6,
            [0.2]*6,
            [0.1]*6,
            [0.3]*6,
        ],
        times_sec=[
            3.0,
            6.0,
            9.0,
            12.0,],
        include_start_from_current=True,   # 첫점 현재값 자동 추가
        start_hold_sec=1.0,
        wait_result=True,
        degrees=False,
    )

    # time.sleep(7)
    # 2) 필요하면 stop
    robot.stop(1)

    # rclpy.shutdown()  # 필요 시 정리


if __name__ == '__main__':
    main()
