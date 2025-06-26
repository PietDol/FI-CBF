# this file will be used for validating the simulation
from visualization import VisualizationData
import json
import numpy as np
from loguru import logger


class ValidateSimulation:
    def __init__(self, sim_dir: str):
        self.sim_dir = sim_dir  # path to the simulation directory

        # load the data
        self.data = VisualizationData.from_directory(dir_path=self.sim_dir)

        # load the env_dict and the env_config
        self.env_dict, self.env_config_dict = self._load_env_files()

    def _load_env_files(self):
        # load the env_data.json dict
        with open(f"{self.sim_dir}/env_data.json", "r") as file:
            env_dict = json.load(file)

        # create path to env_config.json
        split = self.sim_dir.split("/")[:-2]
        config_path = ""
        for i in split:
            config_path += f"{i}/"
        config_path += "env_config.json"

        # load the env_config.json dict
        with open(config_path, "r") as file:
            env_config_dict = json.load(file)

        return env_dict, env_config_dict

    def _distance_from_planner(self):
        # function to calculate the distance from the planner
        planned_path = self.data.path
        robot_path = self.data.robot_pos

        def point_to_segment_distance(p, a, b):
            ap = p - a
            ab = b - a
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
            closest = a + t * ab
            return np.linalg.norm(p - closest)

        distances = []
        for robot_pos in robot_path:
            min_dist = np.inf
            for i in range(len(planned_path) - 1):
                a = planned_path[i]
                b = planned_path[i + 1]
                dist = point_to_segment_distance(robot_pos, a, b)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)

        distances = np.array(distances)
        print(f"Mean distance to path: {np.mean(distances):.4f} m")
        print(f"Std deviation: {np.std(distances):.4f} m")
        print(f"Max distance: {np.max(distances):.4f} m")

        return {
            "mean": np.mean(distances),
            "std": np.std(distances),
            "max": np.max(distances),
            "all_distances": distances,
        }

    def _goal_reached(self):
        # function to check whether the goal is reached
        # load the goal location, goal tolerance and last position
        goal_pos = np.array(self.env_dict["goal_pos"])
        goal_tolerance = self.env_config_dict["goal_tolerance"]
        final_pos = self.data.robot_pos[-1]

        # calculate the distance and check whether the goal is reached
        distance = np.linalg.norm(final_pos - goal_pos)
        reached = distance <= goal_tolerance
        logger.info(f"Goal reached: {reached}")
        return reached

    def _used_information(self):
        # used information during the simulation
        # TODO: How is used information defined?
        pass

    def _cbf_interventions(self):
        # analyse the cbf interventions during the simulation and how heavy they are
        u_cbf = self.data.u_cbf
        u_nominal = self.data.u_nominal
        
        # Compute intervention differences
        diffs = u_cbf - u_nominal  # shape (N, 2)
        magnitudes = np.linalg.norm(diffs, axis=1)  # shape (N,)

        # Consider intervention if magnitude > 1e-6 (to avoid float issues)
        intervention_mask = magnitudes > 1e-6
        num_interventions = np.sum(intervention_mask)

        # Stats on intervention strength
        if num_interventions > 0:
            intervention_magnitudes = magnitudes[intervention_mask]
            mean_intervention = np.mean(intervention_magnitudes)
            std_intervention = np.std(intervention_magnitudes)
            max_intervention = np.max(intervention_magnitudes)
        else:
            mean_intervention = std_intervention = max_intervention = 0.0
        
        logger.info(f"Number of interventions: {num_interventions}")
        logger.info(f"Intervention rate: {num_interventions / len(u_cbf):.2%}")
        logger.info(f"Mean intervention magnitude: {mean_intervention:.4f}")
        logger.info(f"Std intervention magnitude: {std_intervention:.4f}")
        logger.info(f"Max intervention magnitude: {max_intervention:.4f}")

        return {
            "num_interventions": num_interventions,
            "intervention_rate": num_interventions / len(u_cbf),
            "mean": mean_intervention,
            "std": std_intervention,
            "max": max_intervention,
        }
        

    def validate(self):
        # main function to validate the simulation
        goal_reached = self._goal_reached()
        cbf_intervention_info = self._cbf_interventions()

    def __call__(self):
        # validate the simulation
        self.validate()


if __name__ == "__main__":
    validater = ValidateSimulation(
        sim_dir="./runs/baseline_small_gap_succes_switch_mechanism_extra_sensor/simulation_results/loaded_env"
    )
    validater()
