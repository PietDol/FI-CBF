# this file contains all the experiments
from loguru import logger
import numpy as np
from env_generator import EnvGenerator
from env_generator_config import EnvGeneratorConfig
import json


# level 1 experiment: through two circles
class FakeEnvironment:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.experiment_name = "fake_experiment"
        self.env_json_path = f"{self.env_dir}/{self.experiment_name}.json"

        # save the env json
        self.save_env_json()

    def save_env_json(self):
        # function to create json for expirement
        env_dict = {
            "start_pos": [-5.0, 0.0],
            "start_vel": [0.0, 0.0],
            "goal_pos": [5.0, 0.0],
            "obstacles": [
                {
                    "type": "circle",
                    "center": [0.0, 3.5],
                    "radius": 2.5,
                    "robot_radius": 0.7071,
                },
                {
                    "type": "circle",
                    "center": [0.0, -3.5],
                    "radius": 2.5,
                    "robot_radius": 0.7071,
                },
            ],
            "sensors": [{"center": [0.0, 0.0], "max_distance": 10}],
        }

        # save the information in the env_dir such that it can be loaded in via the
        # load_env_information function
        with open(self.env_json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Environment data saved: {self.env_json_path}")


# level 2 experiment: fabric setting
class FabricEnvironment:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.experiment_name = "fabric_experiment"
        self.env_json_path = f"{self.env_dir}/{self.experiment_name}.json"

        # save the env json
        self.save_env_json()

    def save_env_json(self):
        # function to create json for expirement
        robot_radius = 0.4
        env_dict = {
            "start_pos": [-4.0, 9.0],
            "start_vel": [0.0, 0.0],
            "goal_pos": [3.0, -4.5],
            "obstacles": [
                {
                    "type": "rectangle",
                    "center": [-6, 0],
                    "height": 16.0,
                    "width": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [-2.5, 4.0],
                    "height": 2.0,
                    "width": 5.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [-4.0, -4.5],
                    "height": 3.0,
                    "width": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [1.0, 7.5],
                    "height": 1.0,
                    "width": 8.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [5.5, 0.0],
                    "height": 16.0,
                    "width": 1.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [2.5, 2.0],
                    "height": 6.0,
                    "width": 1.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [2.0, -2.0],
                    "height": 2.0,
                    "width": 6.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "rectangle",
                    "center": [2.0, -7.0],
                    "height": 2.0,
                    "width": 6.0,
                    "robot_radius": robot_radius,
                },
            ],
            "sensors": [
                {"center": [-7.5, -7.5], "max_distance": 10},
                {"center": [7.5, 7.5], "max_distance": 10},
                # {"center": [1.0, 1.0], "max_distance": 10},
            ],
        }

        # save the information in the env_dir such that it can be loaded in via the
        # load_env_information function
        with open(self.env_json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Environment data saved: {self.env_json_path}")


# level 3 experiment: cluttered environment
class ClutterdEnvironment:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.experiment_name = "cluttered_experiment"
        self.env_json_path = f"{self.env_dir}/{self.experiment_name}.json"


if __name__ == "__main__":
    # directory where the experiments are saved
    directory = "./runs/experiments"

    # set parameters for the environment config
    config = EnvGeneratorConfig(
        number_of_simulations=1,
        work_dir=directory,
        max_duration_of_simulation=60,
        min_goal_distance=15,
        min_number_of_obstacles=5,
        max_number_of_obstacles=10,
        max_obstacle_size={"circle": 3.0, "rectangle": [3.0, 3.0]},
        min_number_of_sensors=1,
        max_number_of_sensors=5,
        costmap_size=np.array([20, 20]),
        grid_size=0.1,
        planner_mode="CBF infused A*",
        noise_cost_gain=0.0,  # change for the cost to go through uncertain regions (5.0)
        robot_width=1.0,
        robot_height=1.0,
        min_values_state=np.array([-10, -10, -1.5, -1.5]),
        max_values_state=np.array([10, 10, 1.5, 1.5]),
        min_sensor_noise=0.0,
        max_sensor_noise=0.1,
        magnitude_threshold=2.0,
        cbf_state_uncertainty_mode="robust",  # probabilistic or robust
        cbf_switch_velocity_thres=0.2,  # 0.2
        cbf_switch_control_diff_thres=0.01,  # 0.01
        cbf_switch_nominal_control_mag=0.1,  # 0.1
        cbf_confidence_config={
            "levels": [1, 2, 3],
            "vmax": [1.5, 1.0, 0.5],
            "k": [4.0, 3.0, 2.0],
            "sigma_thresholds": [0.03, 0.07],
            "deltas": [
                0.01,
                0.01,
            ],  # with of the sigmoid belonging to the corresponding sigma
            "percentiles": [80.0, 100.0]
        },
        control_fps=50,
        state_estimation_fps=50,
        goal_tolerance=0.1,
        Kp=0.5,  # 0.5
        Kd=0.2,  # 0.1
        u_min_max=np.array([-1000, 1000]),
    )

    # create the environment
    env = EnvGenerator(config=config)

    # create experiment environments for the experiments
    fake_experiment = FakeEnvironment(env_dir=directory)
    fabric_experiment = FabricEnvironment(env_dir=directory)
    cluttered_experiment = ClutterdEnvironment(env_dir=directory)

    # for now only use fake experiment and experiment_mode 3 to set everything up
    # experiments = [fake_experiment, fabric_experiment, cluttered_experiment]
    experiments = [fake_experiment]
    experiment_modes = [0, 1, 2, 3]
    # experiment_modes = [3]

    # iterate over the experiments
    # experiment modes:
    # 0: Baseline
    # 1: Global max based on confidence level
    # 2: Gloabl risk-aware approach based on percentiles
    # 3: Local risk-aware horizon approach
    for experiment in experiments:
        for i in experiment_modes:
            env.run_env_from_file(
                env_file=experiment.env_json_path,
                env_folder=f"{experiment.experiment_name}_{i}",
                experiment_mode=i,
            )
    
