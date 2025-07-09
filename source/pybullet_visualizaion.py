import pybullet as p
import pybullet_data
import numpy as np
import os
import time

# Set this to your simulation output folder
DATA_DIR = "path/to/your/env_folder/simulation_data"  # <-- change this

# Load robot trajectory data
robot_pos = np.load(os.path.join(DATA_DIR, "robot_pos.npy"))  # (N, 2)
robot_pos_estimated = np.load(os.path.join(DATA_DIR, "robot_pos_estimated.npy"))  # optional
sensor_positions = np.load(os.path.join(DATA_DIR, "sensor_positions.npy"))  # optional

# Setup PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to load URDFs
p.setGravity(0, 0, -9.81)

# Load plane
plane_id = p.loadURDF("plane.urdf")

# Load a simple robot (R2D2)
robot_start = [robot_pos[0, 0], robot_pos[0, 1], 0.1]
robot_id = p.loadURDF("r2d2.urdf", basePosition=robot_start)

# Load sensors as spheres (optional visualization)
for sensor in sensor_positions:
    p.loadURDF("sphere_small.urdf", basePosition=[sensor[0], sensor[1], 0.1], globalScaling=0.3)

# You can manually add obstacles here for now (visual only)
# Example: add a cube at (5, 5)
# p.loadURDF("cube_small.urdf", basePosition=[5, 5, 0.1], globalScaling=2.0)

# Visualization loop
dt = 0.02  # 50 Hz
for pos in robot_pos:
    p.resetBasePositionAndOrientation(robot_id, [pos[0], pos[1], 0.1], [0, 0, 0, 1])
    p.stepSimulation()
    time.sleep(dt)  # match original control rate

print("Playback complete. Press Enter to exit.")
input()
p.disconnect()