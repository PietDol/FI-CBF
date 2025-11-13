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
        robot_radius = 0.7071
        env_dict = {
            "start_pos": [-8.0, 8.0],
            "start_vel": [0.0, 0.0],
            "goal_pos": [3.0, -4.0],
            "obstacles": [
                {
                    "type": "circle",
                    "center": [-2.0, 7.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [3.0, 5.0],
                    "radius": 3.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [3.0, 0.0],
                    "radius": 3.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [3.0, -7.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [-5.0, 1.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [-5.0, -4.0],
                    "radius": 3.0,
                    "robot_radius": robot_radius,
                },
            ],
            "sensors": [
                {"center": [-7.5, -7.5], "max_distance": 10},
                {"center": [7.5, 7.5], "max_distance": 10},
                {"center": [0.0, 0.0], "max_distance": 10},
            ],
        }

        # save the information in the env_dir such that it can be loaded in via the
        # load_env_information function
        with open(self.env_json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Environment data saved: {self.env_json_path}")


# level 3 experiment: cluttered environment
class ClutteredEnvironment:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.experiment_name = "cluttered_experiment"
        self.env_json_path = f"{self.env_dir}/{self.experiment_name}.json"
        self.save_env_json()

    def save_env_json(self):
        robot_radius = 0.7071

        env_dict = {
            "start_pos": [-9.0, -9.0],
            "start_vel": [0.0, 0.0],
            "goal_pos": [-4.0, 8.0],
            "obstacles": [
                {
                    "type": "circle",
                    "center": [-6.0, 4.0],
                    "radius": 3.5,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [-1.0, -1.0],
                    "radius": 4.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [7.0, -7.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [5.0, -1.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [6.0, 6.0],
                    "radius": 3.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [-0.75, 6.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [-6.0, -8.0],
                    "radius": 2.0,
                    "robot_radius": robot_radius,
                },
            ],
            "sensors": [
                {"center": [-7.0, 3.0], "max_distance": 10},
                {"center": [0.0, -7.0], "max_distance": 10},
                {"center": [5.0, 5.0], "max_distance": 10},
                {"center": [-2.0, 8.0], "max_distance": 10},
            ],
        }

        with open(self.env_json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Cluttered environment saved: {self.env_json_path}")


# control experiment: different gaps so see limitation
class GapEnvironment:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.experiment_name = "gap_experiment"
        self.env_json_path = f"{self.env_dir}/{self.experiment_name}.json"
        self.save_env_json()

    def save_env_json(self):
        robot_radius = 0.7071
        obstacle_radius = 2.5
        sensor_range = 3.5
        width_openings = [2.5, 2.35, 2.2, 2.05, 1.9, 1.75]
        x_locations = [-7, -4, -1, 2, 5, 8]
        obstacles = []
        sensors = []

        for i in range(len(width_openings)):
            # obstacles
            y = width_openings[i] / 2 + obstacle_radius
            obstacles.append(
                {
                    "type": "circle",
                    "center": [x_locations[i], -y],
                    "radius": obstacle_radius,
                    "robot_radius": robot_radius,
                }
            )
            obstacles.append(
                {
                    "type": "circle",
                    "center": [x_locations[i], y],
                    "radius": obstacle_radius,
                    "robot_radius": robot_radius,
                }
            )
            # sensor at each opening
            sensors.append(
                {"center": [x_locations[i], 0.0], "max_distance": sensor_range}
            )

        env_dict = {
            "start_pos": [-12.0, 0.0],
            "start_vel": [0.0, 0.0],
            "goal_pos": [11.0, 0.0],
            "obstacles": obstacles,
            # sensor at each opening
            "sensors": sensors,
        }

        with open(self.env_json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Cluttered environment saved: {self.env_json_path}")


# debug experiment: an environment to play around with different obstacles
class DebugEnvironment:
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.experiment_name = "debug_experiment"
        self.env_json_path = f"{self.env_dir}/{self.experiment_name}.json"
        self.save_env_json()

    def save_env_json(self):
        robot_radius = 0.7071

        env_dict = {
            "start_pos": [-12.0, 0.0],
            "start_vel": [0.0, 0.0],
            "goal_pos": [11.0, 0.0],
            "obstacles": [
                # opening 1: 1.8 m
                {
                    "type": "circle",
                    "center": [-7.0, -3.4],
                    "radius": 2.5,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [-7.0, 3.4],
                    "radius": 2.5,
                    "robot_radius": robot_radius,
                },
                # opening 2: 1.7 m
                {
                    "type": "circle",
                    "center": [0.5, -3.35],
                    "radius": 2.5,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [0.5, 3.35],
                    "radius": 2.5,
                    "robot_radius": robot_radius,
                },
                # opening 3: 1.6 m
                {
                    "type": "circle",
                    "center": [8.0, -3.3],
                    "radius": 2.5,
                    "robot_radius": robot_radius,
                },
                {
                    "type": "circle",
                    "center": [8.0, 3.3],
                    "radius": 2.5,
                    "robot_radius": robot_radius,
                },
            ],
            # sensor at each opening
            "sensors": [
                {"center": [-7.0, 0.0], "max_distance": 3.5},
                {"center": [-4.0, 0.0], "max_distance": 3.5},
                {"center": [-1.0, 0.0], "max_distance": 3.5},
                {"center": [2.0, 0.0], "max_distance": 3.5},
                {"center": [5.0, 0.0], "max_distance": 3.5},
                {"center": [8.0, 0.0], "max_distance": 3.5},
            ],
        }

        with open(self.env_json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Cluttered environment saved: {self.env_json_path}")


if __name__ == "__main__":
    # directory where the experiments are saved
    # directory = "./runs/experiments_debug"
    # value = 0.15
    directory = f"./runs/E1_overall_performance/cluttered_env"

    # set parameters for the environment config
    config = EnvGeneratorConfig(
        number_of_simulations=1,
        work_dir=directory,
        max_duration_of_simulation=70,
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
            "calculate_grid_per_level": False,  # in our experiment the lipschitz does not depend on velocity
            "vmax": [3.0, 2.0, 1.0],
            "k": 3.0,
            "sigma_thresholds": [0.03, 0.07],
            "deltas": [
                0.01,
                0.01,
            ],  # with of the sigmoid belonging to the corresponding sigma
            "percentiles": [60.0, 70.0, 80.0, 90.0, 100.0],
            "percentile_velocity": [0.5, 1.0, 2.0, 2.5, 3.0],
            "T": 25,
            "T_hold": 25,
            "eta_rel": 0.02, 
            "eta_up": 0.05,
            "h_rel": 0.02,
            "h_up": 0.06,
        },
        cbf_percentile=100.0,
        control_fps=50,
        state_estimation_fps=50,
        goal_tolerance=0.1,
        Kp=0.5,  # 0.5
        Kd=0.0,  # 0.1 0.2
        u_min_max=np.array([-3, 3]),
    )

    # create the environment
    env = EnvGenerator(config=config)

    # create experiment environments for the experiments
    fake_experiment = FakeEnvironment(env_dir=directory)
    fabric_experiment = FabricEnvironment(env_dir=directory)
    cluttered_experiment = ClutteredEnvironment(env_dir=directory)
    gaps_experiment = GapEnvironment(env_dir=directory)
    debug_experiment = DebugEnvironment(env_dir=directory)

    # for now only use fake experiment and experiment_mode 3 to set everything up
    # experiments = [fake_experiment, fabric_experiment, cluttered_experiment]
    experiments = [cluttered_experiment]
    # safety_modes = [0, 1, 2, 3]
    safety_modes = [0, 1, 3]
    # seeds = [7, 15, 22, 28, 33, 43]
    seeds = [7, 15, 22]

    # iterate over the experiments, basically there are 4 robots with a different
    # safety strategy
    # safety strategy modes:
    # 0: Baseline (robot_0)
    # 1: Global max based on confidence level (robot_1)
    # 2: Gloabl risk-aware approach based on percentiles (robot_2)
    # 3: Local risk-aware horizon approach (robot_3)
    for experiment in experiments:
        for i in safety_modes:
            for seed in seeds:
                np.random.seed(seed)
                env.run_env_from_file(
                    env_file=experiment.env_json_path,
                    env_folder=f"{experiment.experiment_name}_{i}_seed_{seed}",
                    experiment_mode=i,
            )
