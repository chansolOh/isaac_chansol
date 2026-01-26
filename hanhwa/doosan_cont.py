
import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint, MoveLine, MoveStop
from sensor_msgs.msg import JointState
import threading
from rclpy.executors import SingleThreadedExecutor

import time
import numpy as np


class Controller(Node):
    def __init__(self):
        super().__init__('move_test_client')

        self.cli_movej = self.create_client(MoveJoint, '/motion/move_joint')
        self.cli_movel = self.create_client(MoveLine, '/motion/move_line')
        self.cli_stop = self.create_client(MoveStop, '/motion/move_stop')

        self.state_node = rclpy.create_node('joint_state_node')
        self.state_executor = SingleThreadedExecutor()
        self.state_executor.add_node(self.state_node)
        self.state_thread = threading.Thread(target=self.state_executor.spin, daemon=True)
        self.state_thread.start()


        self.joint_inputs= [0,0,0,0,0,0]
        self.dt_sec = 10
        self.goal_handle = None
        self.effort_od = None
        self.effort = None
        self.emergency_stop_flag = False
        self.joint_state = []

        self.joint_state_node = self.state_node.create_subscription(
            JointState,
            "/joint_states",
            self.get_joint_state,
            10
        )


        while not self.cli_movej.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for move_joint service...')
        while not self.cli_movel.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for move_line service...')
        while not self.cli_stop.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /motion/move_stop service...')
    
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

    def movel(self, pos, vel=[50.0,50.0], acc=[50.0,50.0], time=0.0, mode=0, sync_type=0):
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

    def stop(self, mode=1):
        req = MoveStop.Request()
        req.stop_mode = mode  # 0: Normal stop, 1: Quick stop, 2: Emergency stop
        future = self.cli_stop.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None:
            self.get_logger().info('Stop command executed.')
        else:
            self.get_logger().error('Failed to call /motion/move_stop.')
    
    def get_joint_state(self, msg: JointState):
        self.joint_states = np.array(msg.position)[[0,1,4,2,3,5]]
        self.joint_vel = np.array(msg.velocity)
        self.effort = np.array(msg.effort)
        if np.any(abs(self.joint_vel)>2.2):
            print("vel : ", self.joint_vel)
            self.stop()
        if self.effort_od is not None:
            eff_diff = np.abs(self.effort_od - self.effort)
            if np.any(eff_diff >300):
                print("eff    :    ",eff_diff)
                self.stop()
        # print(self.joint_states)
        self.effort_od = self.effort.copy()
    
    def reset_robot(self, sync_type = 0):
        self.movej(pos=[  94.43 ,  -34.29 , 113.58  , -4.01 , 100.53, 1.71],
                   vel=30.0,sync_type=sync_type)

  

def main(args=None):
    rclpy.init(args=args)
    robot = Controller()
    # robot.movej(pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],vel=5.0, sync_type=0)
    robot.movej(pos=[  94.43 ,  -34.29 , 113.58  , -4.01 , 100.53, 1.71],vel=5.0, sync_type=0)
    # robot.movej(pos=[47.75, -56.93, -103.52,    0.,    160.54,  -47.78], vel=10.0, sync_type=1)
    # robot.movel(pos=[500.0, 500.0, 500.0, 10.0, 170.0, 0.0],vel=[50.0,50.0], sync_type=1)
    time.sleep(3)
    robot.stop(1)
    # move_client.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
