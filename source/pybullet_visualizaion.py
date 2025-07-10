import pybullet as p
import pybullet_data
import numpy as np
import json
import os
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class PyBulletPlayback:
    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        self.robot_pos = np.load(f"{env_dir}/simulation_data/robot_pos.npy")
        self.sensors = self._load_sensors()
        self.obstacles = self._load_obstacles()

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self._init_world()

    def _load_sensors(self):
        path = f"{self.env_dir}/env_data.json"
        with open(path, "r") as f:
            return json.load(f)["sensors"]

    def _load_obstacles(self):
        path = f"{self.env_dir}/env_data.json"
        with open(path, "r") as f:
            return json.load(f)["obstacles"]

    def _init_world(self):
        p.loadURDF("plane.urdf")

        # Scale robot to 1m wide
        scale_factor = 1.0 / 0.344  # ≈ 2.91
        robot_height = 0.604 * scale_factor
        self.z_base = 0.604 * scale_factor / 2

        # Rotate so arm faces +X
        self.rotation = R.from_euler('z', -90, degrees=True).as_quat()
        start = [self.robot_pos[0, 0], self.robot_pos[0, 1], self.z_base]

        self.robot_id = p.loadURDF(
            "r2d2.urdf",
            basePosition=start,
            baseOrientation=self.rotation,
            globalScaling=scale_factor
        )
        aabb = p.getAABB(self.robot_id)
        robot_size = np.array(aabb[1]) - np.array(aabb[0])
        print(f"Robot size (x, y, z): {robot_size}")

        self._spawn_sensors()
        self._spawn_obstacles()

    def _spawn_sensors(self):
        sensor_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 0, 0, 1]
        )
        for sensor in self.sensors:
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sensor_visual,
                basePosition=[sensor["center"][0], sensor["center"][1], 3.0],
            )

    def _spawn_obstacles(self):
        for obs in self.obstacles:
            pos = obs["center"]
            radius = obs["radius"]
            vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=2.0, rgbaColor=[1, 0, 0, 1]
            )
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=2.0)
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[pos[0], pos[1], 0.1],
            )

    def playback(self, dt: float = 0.02):
        for pos in self.robot_pos:
            p.resetBasePositionAndOrientation(
                self.robot_id, [pos[0], pos[1], self.z_base], self.rotation
            )
            p.stepSimulation()
            time.sleep(dt)
        print("Playback finished.")
        input("Press Enter to exit...")
        p.disconnect()


if __name__ == "__main__":
    VISUALIZER = PyBulletPlayback("./runs/experiment_success/simulation_results/loaded_env_0")  # <--- update path if needed
    VISUALIZER.playback()