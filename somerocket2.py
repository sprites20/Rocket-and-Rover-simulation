import pybullet as p
import pybullet_data
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the ground
plane_id = p.loadURDF("plane.urdf")

# Create the rocket (rectangular prism)
prism_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 0.2, 0.2])
prism_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.0, 0.2, 0.2], rgbaColor=[1, 0, 0, 1])
initial_orientation = p.getQuaternionFromEuler([0, -math.pi / 2, 0])
prism_body = p.createMultiBody(
    baseMass=5,
    baseCollisionShapeIndex=prism_collision,
    baseVisualShapeIndex=prism_visual,
    basePosition=[0, 0, 2],
    baseOrientation=initial_orientation
)

# Create multiple target balls
ball_positions = [[3, 0, 50], [5, 0, 100], [7, 0, 120], [10, 0, 150], [12, 0, 200], [15, 0, 250], [17, 0, 300]]
balls = []

for pos in ball_positions:
    ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
    ball_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=ball_collision,
        baseVisualShapeIndex=ball_visual,
        basePosition=pos
    )
    balls.append(ball_body)

# Set gravity
p.setGravity(0, 0, -9.8)
p.changeDynamics(prism_body, -1, angularDamping=0.05, linearDamping=0.02)

# PID Control Variables
Kp, Ki, Kd = 0.8, 0.05, 0.3
previous_error, integral = 0, 0
dt = 1 / 240  # Simulation time step
current_target_index = 0  # Track the current target ball

# Matplotlib Figure Setup
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("X Position")
ax.set_ylabel("Z Position")
ax.set_xlim(-10, 20)
ax.set_ylim(0, 350)
ax.grid(True)

# Plot Target Positions
target_x = [pos[0] for pos in ball_positions]
target_z = [pos[2] for pos in ball_positions]
ax.scatter(target_x, target_z, color='green', label="Targets")

# Rocket Trajectory Data
rocket_x, rocket_z = [], []
trajectory_plot, = ax.plot([], [], 'r-', label="Rocket Path")
ax.legend()

# Simulation loop
for step in range(100000):
    rocket_pos, rocket_orientation = p.getBasePositionAndOrientation(prism_body)
    
    # Store trajectory for plotting
    rocket_x.append(rocket_pos[0])
    rocket_z.append(rocket_pos[2])
    
    # Update Matplotlib Plot
    trajectory_plot.set_xdata(rocket_x)
    trajectory_plot.set_ydata(rocket_z)
    ax.set_xlim(min(rocket_x) - 5, max(rocket_x) + 5)
    ax.set_ylim(min(rocket_z) - 5, max(rocket_z) + 50)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Get the current target ball position
    target_ball = balls[current_target_index]
    target_pos, _ = p.getBasePositionAndOrientation(target_ball)
    
    # Compute distance to current target
    distance_to_target = math.sqrt(sum((rocket_pos[i] - target_pos[i]) ** 2 for i in range(3)))
    
    # Switch to the next target if close to the current one
    if distance_to_target < 10 and current_target_index < len(balls) - 1:
        current_target_index += 1
        print(f"Switching to target {current_target_index + 1}")

    # Compute direction to ball
    direction_to_ball = [target_pos[i] - rocket_pos[i] for i in range(3)]
    desired_pitch_rad = math.atan2(direction_to_ball[2], direction_to_ball[0])
    
    # Get current pitch angle from quaternion
    euler_angles = p.getEulerFromQuaternion(rocket_orientation)
    current_pitch_rad = euler_angles[1]
    
    # PID Error computation
    error = desired_pitch_rad + current_pitch_rad
    integral += error * dt if abs(error) > 0.01 else 0  # Prevent integral windup
    derivative = (error - previous_error) / dt
    previous_error = error
    
    max_torque = 50
    dead_zone = 0.01
    if abs(error) < dead_zone and abs(derivative) < dead_zone:
        torque_y = 0  
    else:
        torque_y = -max(-max_torque, min(max_torque, Kp * error + Ki * integral + Kd * derivative))

    # Apply torque in the LOCAL frame
    p.applyExternalTorque(prism_body, -1, [0, torque_y, 0], p.LINK_FRAME)
    
    # Compute thrust force
    thrust_magnitude = 200
    
    # Use the Z-axis column from the rotation matrix
    rotation_matrix = p.getMatrixFromQuaternion(rocket_orientation)
    forward_direction = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
    
    # Get rocket velocity
    linear_velocity, _ = p.getBaseVelocity(prism_body)
    speed = math.sqrt(sum(v**2 for v in linear_velocity))

    # Target speed based on distance
    min_speed, max_speed = 5, 50
    distance_factor = max(0.2, min(1.0, distance_to_target / 20))
    target_speed = min_speed + (max_speed - min_speed) * distance_factor

    # Thrust control to avoid excessive speed
    if speed > target_speed:
        thrust_magnitude *= 0.9
    elif speed < target_speed * 0.9:
        thrust_magnitude *= 1.05

    # Apply thrust in the direction of the rocket
    thrust_vector = [thrust_magnitude * d for d in forward_direction]
    p.applyExternalForce(prism_body, -1, thrust_vector, rocket_pos, p.WORLD_FRAME)

    # Camera follows the rocket
    p.resetDebugVisualizerCamera(10, 0, -30, rocket_pos)
    
    p.stepSimulation()
    time.sleep(dt)

# Disconnect from PyBullet
p.disconnect()
