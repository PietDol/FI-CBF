import pybullet as p
import pybullet_data
import numpy as np
import json
import os
import time
from scipy.spatial.transform import Rotation as R
import tkinter as tk


class PyBulletPlayback:
    def __init__(self, env_dir: str):
        self.env_dir = env_dir
        self.robot_pos = np.load(f"{env_dir}/simulation_data/robot_pos.npy")
        self.sensors = self._load_sensors()
        self.obstacles = self._load_obstacles()

        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.paused = False

        self._init_world()

    def _load_sensors(self):
        path = f"{self.env_dir}/env_data.json"
        with open(path, "r") as f:
            return json.load(f)["sensors"]

    def _load_obstacles(self):
        path = f"{self.env_dir}/env_data.json"
        with open(path, "r") as f:
            return json.load(f)["obstacles"]
    
    def toggle_pause(self, event=None):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")

    def _init_world(self):
        p.loadURDF("plane.urdf")

        # Scale robot to 1m wide
        scale_factor = 1.0 / 0.344  # ≈ 2.91
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

    def playback(self, dt=0.02):
        # === Load all simulation data ===
        h_true = np.load(f"{self.env_dir}/simulation_data/h_true.npy")           # (T, N)
        h_est = np.load(f"{self.env_dir}/simulation_data/h_estimated.npy")      # (T, N)
        noise = np.load(f"{self.env_dir}/simulation_data/noise.npy")            # (T,)
        confidence = np.load(f"{self.env_dir}/simulation_data/conf_level.npy")  # (T,)
        margin = np.load(f"{self.env_dir}/simulation_data/safety_margin.npy")   # (T, N)

        num_cbfs = h_true.shape[1]

        # === Setup GUI ===
        root = tk.Tk()
        root.bind("<space>", self.toggle_pause)
        root.title("Simulation Monitor")
        root.geometry("500x450")
        root.resizable(False, False)

        # Header: time + step
        time_var = tk.StringVar()
        tk.Label(root, textvariable=time_var, font=("Courier", 11, "bold")).pack(anchor="w", padx=10, pady=5)

        # CBF lines (true, estimated, margin)
        cbf_vars = []
        for i in range(num_cbfs):
            var = tk.StringVar()
            cbf_vars.append(var)
            tk.Label(root, textvariable=var, font=("Courier", 10)).pack(anchor="w", padx=10)

        # Extra values
        noise_var = tk.StringVar()
        conf_var = tk.StringVar()
        tk.Label(root, textvariable=noise_var, font=("Courier", 10)).pack(anchor="w", padx=10, pady=(10, 0))
        tk.Label(root, textvariable=conf_var, font=("Courier", 10)).pack(anchor="w", padx=10)

        # === Run simulation ===
        for t, pos in enumerate(self.robot_pos):
            while self.paused:
                root.update()
                time.sleep(0.05)
            
            p.resetBasePositionAndOrientation(
                self.robot_id, [pos[0], pos[1], self.z_base], self.rotation
            )
            p.stepSimulation()

            time_var.set(f"Time: {t * dt:.2f}s  (Step {t})")

            for i in range(num_cbfs):
                ht = h_true[t, i]
                he = h_est[t, i]
                sm = margin[t, i]
                cbf_vars[i].set(f"h[{i}]  true: {ht:6.3f}   est: {he:6.3f}   margin: {sm:6.3f}")

            noise_var.set(f"Perception noise:     {noise[t]:.3f}")
            conf_var.set(f"Confidence level:      {confidence[t]:.3f}")

            root.update_idletasks()
            root.update()

            time.sleep(dt)

        print("Playback finished.")
        input("Press Enter to exit...")
        p.disconnect()
        root.destroy()


if __name__ == "__main__":
    VISUALIZER = PyBulletPlayback("./runs/experiment_success/simulation_results/loaded_env_0")  # <--- update path if needed
    VISUALIZER.playback()