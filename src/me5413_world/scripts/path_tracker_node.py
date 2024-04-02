#!/usr/bin/env python3

import tf
import math
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.dt = 0.1

    def calculate(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        P_term = self.kp * error
        I_term = self.ki * self.integral
        D_term = self.kd * derivative

        output = max(min(P_term + I_term + D_term, 1.0), -1.0)

        self.prev_error = error

        return output

    def updateSettings(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


class PathTracker:
    def __init__(self):
        self.world_frame = "world"
        self.robot_frame = "base_link"
        self.odom_world_robot = Odometry()
        self.params_update = False
        self.kp = 0.5
        self.ki = 0.2
        self.kd = 0.2
        self.speed_target = 0.5
        self.stanley_K = 0.5
        self.pid = PID(self.kp, self.ki, self.kd)

        self.sub_robot_odom = rospy.Subscriber("gazebo/ground_truth/state", Odometry, self.odom_callback)
        self.sub_local_path = rospy.Subscriber("/me5413_world/planning/local_path", Path, self.local_path_callback)
        self.pub_cmd_vel = rospy.Publisher("/jackal_velocity_controller/cmd_vel", Twist, queue_size=1)

    def odom_callback(self, msg):
        self.world_frame = msg.header.frame_id
        self.robot_frame = msg.child_frame_id
        self.odom_world_robot = msg

    def local_path_callback(self, msg):
        self.pose_world_goal = msg.poses[5].pose
        self.compute_cmd_vel()
        self.pub_cmd_vel.publish(self.compute_cmd_vel())

    def compute_cmd_vel(self):
        tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_buffer)

        q_robot = [self.odom_world_robot.pose.pose.orientation.x, self.odom_world_robot.pose.pose.orientation.y,
                   self.odom_world_robot.pose.pose.orientation.z, self.odom_world_robot.pose.pose.orientation.w]
        q_goal = [self.pose_world_goal.orientation.x, self.pose_world_goal.orientation.y,
                  self.pose_world_goal.orientation.z, self.pose_world_goal.orientation.w]

        euler_robot = tf.transformations.euler_from_quaternion(q_robot)
        euler_goal = tf.transformations.euler_from_quaternion(q_goal)

        yaw_robot = euler_robot[2]
        yaw_goal = euler_goal[2]

        heading_error = yaw_robot - yaw_goal

        # 计算横向误差
        point_robot = [self.odom_world_robot.pose.pose.position.x, self.odom_world_robot.pose.pose.position.y]
        point_goal = [self.pose_world_goal.position.x, self.pose_world_goal.position.y]
        vector_goal_robot = [point_robot[0] - point_goal[0], point_robot[1] - point_goal[1]]

        angle_goal_robot = math.atan2(vector_goal_robot[1], vector_goal_robot[0])
        angle_diff = angle_goal_robot - yaw_goal
        lat_error = math.sqrt(vector_goal_robot[0] ** 2 + vector_goal_robot[1] ** 2) * math.sin(angle_diff)

        # 速度
        velocity = math.sqrt(self.odom_world_robot.twist.twist.linear.x ** 2 +
                             self.odom_world_robot.twist.twist.linear.y ** 2)

        cmd_vel = Twist()

        if self.params_update:  # 假设这是一个已定义的条件变量
            self.pid.updateSettings(self.kp, self.ki, self.kd)  # 假设这个PID类和方法已经在Python中实现
            self.params_update = False

        cmd_vel.linear.x = self.pid.calculate(self.speed_target, velocity)  # 假设SPEED_TARGET是一个预定义的目标速度
        cmd_vel.angular.z = self.computeStanleyControl(heading_error, lat_error, velocity)  # 假设这个函数已经定义

        return cmd_vel

    def computeStanleyControl(self, heading_error, lat_error, velocity):
        stanley_output = -1.0 * (heading_error + math.atan2(self.stanley_K * lat_error, max(velocity, 0.3)))
        return min(max(stanley_output, -2.2), 2.2)


if __name__ == '__main__':
    rospy.init_node('path_tracker_node')
    pt = PathTracker()
    rospy.spin()
