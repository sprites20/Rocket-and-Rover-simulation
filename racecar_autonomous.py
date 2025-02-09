import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
parentdir = os.path.join(currentdir, "../gym")

os.sys.path.insert(0, parentdir)

import pybullet as p
import pybullet_data

import time
import math
import numpy as np

cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0, 0, -10)

useRealTimeSim = 1

#for video recording (works best on Mac and Linux, not well on Windows)
#p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
p.setRealTimeSimulation(useRealTimeSim)  # either this
#p.loadURDF("plane.urdf")
p.loadSDF(os.path.join(pybullet_data.getDataPath(), "stadium.sdf"))

car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
for i in range(p.getNumJoints(car)):
  print(p.getJointInfo(car, i))

inactive_wheels = [3, 5, 7]
wheels = [2]

for wheel in inactive_wheels:
  p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

steering = [4, 6]

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -10, 10, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 10, 10)
steeringSlider = p.addUserDebugParameter("steering", -0.5, 0.5, 0)


ball = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere_small.urdf"), 
                  basePosition=[1, 0, 0.3], 
                  globalScaling=3.0)  # Scale the sphere 3x larger


# PID Controller Parameters
Kp_steering = 1.2  # Adjusted for smooth turning
Ki_steering = 0.001
Kd_steering = 0.05

Kp_speed = 1.0
Ki_speed = 0.002
Kd_speed = 0.05

# PID State Variables
integral_speed = 0
previous_error_speed = 0

integral_steering = 0
previous_error_steering = 0

while True:
    # Get car and ball positions
    car_pos, car_ori = p.getBasePositionAndOrientation(car)
    ball_pos, _ = p.getBasePositionAndOrientation(ball)

    # Convert quaternion to yaw angle (car's forward direction)
    car_euler = p.getEulerFromQuaternion(car_ori)
    car_yaw = car_euler[2]  # Yaw angle (rotation around Z)

    # Compute vector from car to ball
    dx = ball_pos[0] - car_pos[0]
    dy = ball_pos[1] - car_pos[1]

    # Compute the angle to the ball
    target_angle = math.atan2(dy, dx)  # Angle from car to ball
    angle_diff = target_angle - car_yaw  # Difference between desired and actual angle

    # Normalize angle between -pi and pi
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # PID for steering
    integral_steering += angle_diff
    derivative_steering = angle_diff - previous_error_steering
    steering_angle = (Kp_steering * angle_diff) + (Ki_steering * integral_steering) + (Kd_steering * derivative_steering)
    previous_error_steering = angle_diff

    # Limit steering angle
    max_steering_angle = 0.5
    steering_angle = max(-max_steering_angle, min(steering_angle, max_steering_angle))

    # Distance to ball
    distance_to_ball = math.sqrt(dx**2 + dy**2)

    # PID for speed (slow down when turning too much)
    integral_speed += distance_to_ball
    derivative_speed = distance_to_ball - previous_error_speed
    speed = (Kp_speed * distance_to_ball) + (Ki_speed * integral_speed) + (Kd_speed * derivative_speed)
    previous_error_speed = distance_to_ball

    # Reduce speed if turning too much
    speed *= max(0.5, 1 - abs(angle_diff))  # Slow down when angle is large

    # Limit speed
    max_speed = 10.0 if distance_to_ball > 2 else 5.0
    speed = max(-max_speed, min(speed, max_speed))

    # Apply motor and steering control
    for wheel in wheels:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=speed, force=5)

    for steer in steering:
        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steering_angle)

    if useRealTimeSim == 0:
        p.stepSimulation()

    time.sleep(0.01)