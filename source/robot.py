from perception import Perception
from a_star import AStarPlanner
from cbf_infused_a_star import CBFInfusedAStar, CBFCostmap
from robot_base_config import RobotBaseCBFConfig
from loguru import logger
import json


class RobotConfig:
    def __init__(self):
        # structured class needed to create the robot
        pass


class Robot:
    def __init__(self):
        # this robot represents the robot
        pass

    @classmethod
    def from_config(self):
        # function to create the robot object from the config file
        pass

    @classmethod
    def from_file(self):
        # function to create the robot from a file (containing the dict)
        pass

    def save_to_file(self, path: str):
        # Convert all class attributes to a serializable dict
        config_dict = self.__dict__.copy()
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logger.success(f"Configuration saved to {path}")
