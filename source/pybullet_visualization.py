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
        self.step = 0  # current frame index
        self.robot_height = 0.0       # height of the robot (obstacles have same height as robot)

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

    def step_forward(self, event=None):
        if self.paused and self.step < len(self.robot_pos) - 1:
            self.step += 1
            self.update_frame()

    def step_back(self, event=None):
        if self.paused and self.step > 0:
            self.step -= 1
            self.update_frame()
    
    def update_frame(self):
        t = self.step
        pos = self.robot_pos[t]
        p.resetBasePositionAndOrientation(
            self.robot_id, [pos[0], pos[1], self.robot_height], self.rotation
        )
        p.stepSimulation()

        self.time_var.set(f"Time: {t * self.dt:.2f}s  (Step {t})")

        for i in range(self.num_cbfs):
            ht = self.h_true[t, i]
            he = self.h_est[t, i]
            sm = self.margin[t, i]
            self.cbf_vars[i].set(f"h[{i}]  true: {ht:6.3f}   est: {he:6.3f}   margin: {sm:6.3f}")

        self.noise_var.set(f"Perception noise:     {self.noise[t]:.3f}")
        self.conf_var.set(f"Confidence level:      {self.confidence[t]:.3f}")
        self.root.update_idletasks()
        self.root.update()

    def _init_world(self):
        p.loadURDF("plane.urdf")
        self.rotation = [0, 0, 0, 1]  # no rotation needed
        start = [self.robot_pos[0, 0], self.robot_pos[0, 1], self.robot_height]
        self.robot_id = self._spawn_robot(start)
        self._spawn_sensors()
        self._spawn_obstacles()
    
    def _spawn_robot(self, position):
        half_extents = [0.5, 0.5, self.robot_height]  # → 1m x 1m robot
        visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.0, 0.6, 0.9, 1.0]
        )
        collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )
        # return p.createMultiBody(
        #     baseMass=1,
        #     baseCollisionShapeIndex=collision,
        #     baseVisualShapeIndex=visual,
        #     basePosition=position
        # )
        return p.loadURDF(
            "husky/husky.urdf",
            basePosition=position,
            baseOrientation=self.rotation,
            useFixedBase=True  # disable dynamics unless you want it moving
        )

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
        obstacle_height = 0.5   # height at which the obstacle can be placed
        for obs in self.obstacles:
            obs_type = obs.get("type", "circle")  # Default to circle if type is missing
            pos = obs["center"]

            if obs_type == "rectangle":
                width = obs["width"]
                height = obs["height"]
                z_half = obstacle_height    # same height as robot
                half_extents = [width / 2, height / 2, z_half / 2]

                vis = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=[0.2, 0.6, 1.0, 1.0]  # light blue
                )
                col = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=half_extents
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    baseCollisionShapeIndex=col,
                    basePosition=[pos[0], pos[1], half_extents[2]]  # center at top surface
                )

            else:  # assume circular obstacle
                radius = obs["radius"]
                z_half = obstacle_height    # same height as robot
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=radius, length=2 * z_half, rgbaColor=[1, 0, 0, 1]
                )
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=radius, height=2 * z_half
                )
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    baseCollisionShapeIndex=col,
                    basePosition=[pos[0], pos[1], z_half]  # center at top surface
                )

    def playback(self, dt=0.02):
        # === Load simulation data ===
        self.h_true = np.load(f"{self.env_dir}/simulation_data/h_true.npy")
        self.h_est = np.load(f"{self.env_dir}/simulation_data/h_estimated.npy")
        self.noise = np.load(f"{self.env_dir}/simulation_data/noise.npy")
        self.confidence = np.load(f"{self.env_dir}/simulation_data/conf_level.npy")
        self.margin = np.load(f"{self.env_dir}/simulation_data/safety_margin.npy")

        self.num_cbfs = self.h_true.shape[1]
        self.dt = dt

        # === Setup Tkinter GUI ===
        self.root = tk.Tk()
        self.root.title("Simulation Monitor")
        self.root.geometry("500x450")
        self.root.resizable(False, False)

        # Bind keys
        self.root.bind("<space>", self.toggle_pause)
        self.root.bind("<Right>", self.step_forward)
        self.root.bind("<Left>", self.step_back)

        # Time and Step Display
        self.time_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.time_var, font=("Courier", 11, "bold")).pack(anchor="w", padx=10, pady=5)

        # CBF displays
        self.cbf_vars = []
        for i in range(self.num_cbfs):
            var = tk.StringVar()
            self.cbf_vars.append(var)
            tk.Label(self.root, textvariable=var, font=("Courier", 10)).pack(anchor="w", padx=10)

        # Other data
        self.noise_var = tk.StringVar()
        self.conf_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.noise_var, font=("Courier", 10)).pack(anchor="w", padx=10, pady=(10, 0))
        tk.Label(self.root, textvariable=self.conf_var, font=("Courier", 10)).pack(anchor="w", padx=10)

        # === Run Simulation ===
        while self.step < len(self.robot_pos):
            if not self.paused:
                self.update_frame()
                self.step += 1
                time.sleep(dt)
            else:
                self.root.update()
                time.sleep(0.05)

        print("Playback finished.")
        input("Press Enter to exit...")
        p.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    # pybullet_visualizer = PyBulletPlayback("./runs/experiment_fake_success/simulation_results/fake_experiment_3")  
    # pybullet_visualizer = PyBulletPlayback("./runs/experiment_fabric_success/simulation_results/fabric_experiment_3")
    pybullet_visualizer = PyBulletPlayback("./runs/experiment_cluttered_success/simulation_results/cluttered_experiment_3")    
    pybullet_visualizer.playback()