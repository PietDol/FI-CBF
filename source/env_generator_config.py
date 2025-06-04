import numpy as np
from loguru import logger
import json


class EnvGeneratorConfig:
    def __init__(
        self,
        # parameters for the environment
        number_of_simulations: int = 1,
        work_dir: str = None,
        max_duration_of_simulation: float = None,
        min_goal_distance: float = None,
        min_number_of_obstacles: int = 1,
        max_number_of_obstacles: int = None,
        max_obstacle_size: float = None,
        min_number_of_sensors: int = 1,
        max_number_of_sensors: int = None,
        # parameters for the robot
        costmap_size: np.ndarray = None,
        grid_size: float = 1,
        planner_mode: str = "CBF infused A*",
        noise_cost_gain: float = 0.0,
        robot_width: float = 1.0,
        robot_height: float = 1.0,
        min_values_state: np.ndarray = None,
        max_values_state: np.ndarray = None,
        min_sensor_noise: float = 0.0,
        max_sensor_noise: float = 0.1,
        magnitude_threshold: float = 2.0,
        cbf_state_uncertainty_mode: str = "robust",
        cbf_switch_velocity_thres: float = None,
        cbf_switch_control_diff_thres: float = None,
        cbf_switch_nominal_control_mag: float = None,
        control_fps: float = 50,
        state_estimation_fps: float = 50,
        goal_tolerance: float = 0.1,
        Kp: float = 0.5,
        Kd: float = 0.1,
        u_min_max: np.ndarray = np.array([-1000, 1000]),
    ):
        # general parameters
        self.number_of_simulations = number_of_simulations
        self.max_duration_of_simulation = max_duration_of_simulation  # in seconds
        self.min_goal_distance = min_goal_distance  # in m
        self.work_dir = work_dir

        # environment parameters (robot and obstacles)
        self.costmap_size = costmap_size  # in m
        self.robot_width = robot_width  # in m
        self.robot_height = robot_height  # in m
        self.min_number_of_obstacles = min_number_of_obstacles
        self.max_number_of_obstacles = max_number_of_obstacles
        self.max_obstacle_size = max_obstacle_size  # in m

        # other robot parameters
        self.planner_mode = planner_mode
        self.planner_alpha = noise_cost_gain
        self.min_values_state = min_values_state
        self.max_values_state = max_values_state
        self.control_fps = control_fps
        self.state_estimation_fps = state_estimation_fps
        self.goal_tolerance = goal_tolerance
        self.Kp = Kp
        self.Kd = Kd
        self.u_min_max = u_min_max

        # costmap parameters
        self.grid_size = grid_size  # in m

        # cbf parameters
        self.cbf_state_uncertainty_mode = (
            cbf_state_uncertainty_mode  # choose between robust or probabilistic
        )
        self.cbf_switch_velocity_thres = cbf_switch_velocity_thres
        self.cbf_switch_control_diff_thres = cbf_switch_control_diff_thres
        self.cbf_switch_nominal_control_mag = cbf_switch_nominal_control_mag

        # perception parameters
        self.min_number_of_sensors = min_number_of_sensors
        self.max_number_of_sensors = max_number_of_sensors
        self.min_sensor_noise = min_sensor_noise    # in m
        self.max_sensor_noise = max_sensor_noise    # in m
        self.magnitude_threshold = magnitude_threshold

    def log_information(self):
        # log all the inforation
        config_dict = self.__dict__.copy()
        for key, val in config_dict.items():
            logger.info(f"{key}: {val}")

    def save_to_file(self, path: str):
        # Convert all class attributes to a serializable dict
        config_dict = self.__dict__.copy()

        # convert np.arrays to lists
        for key, val in config_dict.items():
            if isinstance(val, np.ndarray):
                config_dict[key] = val.tolist()
        
        # save json
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logger.success(f"Configuration saved to {path}")

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        for key, val in config_dict.items():
            if isinstance(val, list):
                config_dict[key] = np.array(val)

        logger.success(f"Configuration loaded from {path}")
        return cls(**config_dict)
