# this file will be used for validating the simulation
from visualization import VisualizationData
import json
import numpy as np
from loguru import logger
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt


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
        logger.info(f"Mean distance to path: {np.mean(distances):.4f} m")
        logger.info(f"Std deviation: {np.std(distances):.4f} m")
        logger.info(f"Max distance: {np.max(distances):.4f} m")

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
        final_pos = self.data.robot_pos_estimated[-1]

        # calculate the distance and check whether the goal is reached
        distance = np.linalg.norm(final_pos - goal_pos)
        reached = distance <= goal_tolerance
        time_to_goal = self.data.control_time[-1]
        if reached:
            logger.info(
                f"Goal reached: {reached} in {time_to_goal} s (distance: {distance:.4f} m)"
            )
        else:
            logger.info(
                f"Goal reached: {reached} after {time_to_goal} s (distance: {distance:.4f} m)"
            )
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

    def _safety_margin(self):
        min_safety_margin = np.min(self.data.safety_margin, axis=0)
        max_safety_margin = np.max(self.data.safety_margin, axis=0)
        mean_safety_margin = np.mean(self.data.safety_margin, axis=0)
        std_safety_margin = np.std(self.data.safety_margin, axis=0)

        # log everything
        logger.info(f"Min safety margin: {np.round(min_safety_margin, 4)}")
        logger.info(f"Max safety margin: {np.round(max_safety_margin, 4)}")
        logger.info(f"Mean safety margin: {np.round(mean_safety_margin, 4)}")
        logger.info(f"Std safety margin: {np.round(std_safety_margin, 4)}")

        return {
            "min": min_safety_margin,
            "max": max_safety_margin,
            "mean": mean_safety_margin,
            "std": std_safety_margin,
        }

    def _h_diffs(self):
        # difference between true and estimated h -> h_true > h_est
        h_diffs = self.data.h_true - self.data.h_estimated  # > 0
        min_diffs = np.min(h_diffs, axis=0)
        max_diffs = np.max(h_diffs, axis=0)
        mean_diffs = np.mean(h_diffs, axis=0)
        std_diffs = np.std(h_diffs, axis=0)

        # log
        logger.info(f"Min h_true - h_est: {np.round(min_diffs, 4)}")
        logger.info(f"Max h_true - h_est: {np.round(max_diffs, 4)}")
        logger.info(f"Mean h_true - h_est: {np.round(mean_diffs, 4)}")
        logger.info(f"Std h_true - h_est: {np.round(std_diffs, 4)}")

        return {
            "min": min_diffs,
            "max": max_diffs,
            "mean": mean_diffs,
            "std": std_diffs,
        }

    def _control_effort(self):
        u_cbf = self.data.u_cbf
        u_cbf_mean = np.linalg.norm(u_cbf, axis=1).mean()
        u_cbf_sum = np.linalg.norm(u_cbf, axis=1).sum()
        logger.info(f"Mean control effort (||u_cbf||): {np.round(u_cbf_mean, 4)}")
        logger.info(f"Sum control effort (||u_cbf||): {np.round(u_cbf_sum, 4)}")

    def cbf_data(self):
        # function to return all the important cbf data
        h_diffs = self.data.h_true - self.data.h_estimated  # > 0

        # Compute intervention differences
        diffs = self.data.u_cbf - self.data.u_nominal  # shape (N, 2)
        magnitudes = np.linalg.norm(diffs, axis=1)  # shape (N,)

        # Consider intervention if magnitude > 1e-6 (to avoid float issues)
        intervention_mask = magnitudes > 1e-6
        intervention_magnitudes = magnitudes[intervention_mask]

        return {
            "h_true": self.data.h_true,
            "h_est": self.data.h_estimated,
            "safety_margin": self.data.safety_margin,
            "h_diffs": h_diffs,
            "intervention_mag": intervention_magnitudes,
        }

    def analyze_cbf(self):
        # function to analyze the cbf performance
        # minimum h value
        h_true_min = np.min(self.data.h_true, axis=0)
        if np.any(h_true_min < 0):
            logger.error(f"Collision during simulation!")
        logger.info(f"Minimum h values: {np.round(h_true_min, 4)}")

        # analyze cbf interventions
        cbf_intervention = self._cbf_interventions()

        # safety margin
        safety_margin = self._safety_margin()

        # difference between true and estimated h -> h_true > h_est
        h_diffs = self._h_diffs()

        return {
            "h_min": h_true_min,
            "cbf_intervention": cbf_intervention,
            "safety_margin": safety_margin,
            "h_diffs": h_diffs,
        }

    def validate(self):
        # main function to validate the simulation
        logger.info(f"Validation for {self.sim_dir}")

        logger.info(f"- - - - - - - - - - Goal metrics - - - - - - - - - -")
        self._goal_reached()

        logger.info(f"- - - - - - - - - - Control metrics - - - - - - - - - -")
        self._control_effort()

        logger.info(f"- - - - - - - - - - CBF metrics - - - - - - - - - -")
        self.analyze_cbf()

        logger.info(f"- - - - - - - - - - Trajectory metrics - - - - - - - - - -")
        self._distance_from_planner()

    def __call__(self):
        # validate the simulation
        self.validate()


# class to analyze the experiments
class ValidateExperiments:
    def __init__(self, validate_cfg: dict):
        self.validate_cfg = validate_cfg
        self.validation_objects = self._create_validation_objects()

    def _create_validation_objects(self):
        # function to create the validation objects
        validation_objects = {}

        # iterate over all the experiment modes
        for exp_folder in self.validate_cfg["exp_folders"]:
            sim_dirs = os.listdir(f"{exp_folder}/simulation_results")
            exp_dict = {}

            # iterate over simulation dirs
            for sim_dir in sim_dirs:
                # skip the all_envs folder
                if sim_dir == "all_envs":
                    continue

                exp_mode = "_".join(sim_dir.split("_")[:-1])
                cbf_mode = sim_dir.split("_")[-1]

                validater = ValidateSimulation(
                    sim_dir=f"{exp_folder}/simulation_results/{sim_dir}"
                )
                exp_dict[cbf_mode] = validater

            # add all the different cbf mode to experiment
            validation_objects[exp_mode] = exp_dict

        return validation_objects

    def _plot_boxplot_on_ax(
        self,
        ax,
        data_dict,
        ylabel,
        xlabel="Robot ID",
        title=None,
        group_order=None,
        color_palette="Set2",
        showfliers=True,
        rotation=0,
    ):
        """
        Plot a boxplot on a given matplotlib axis.

        Parameters:
        - ax: Matplotlib axis object to plot on.
        - data_dict: dict of {group_name: list or np.array of values}
        - ylabel: Label for the y-axis
        - xlabel: Label for the x-axis
        - title: Plot title (optional)
        - group_order: Optional list to control group order on the x-axis
        - color_palette: Seaborn color palette
        - showfliers: Whether to show outlier points
        - rotation: X-axis label rotation in degrees

        Returns:
        - Modified matplotlib axis
        """
        # Convert to DataFrame
        plot_data = []
        for group, values in data_dict.items():
            for v in values:
                plot_data.append({"Group": group, "Value": v})
        df = pd.DataFrame(plot_data)

        # Plot boxplot and swarm overlay
        sns.boxplot(
            data=df,
            x="Group",
            y="Value",
            palette=color_palette,
            order=group_order,
            showfliers=showfliers,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="Group",
            y="Value",
            color="black",
            size=3,
            alpha=0.5,
            order=group_order,
            ax=ax,
        )

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        return ax

    def creat_cbf_plots(self):
        for exp_mode, exp_dict in self.validation_objects.items():
            # Step 1: Collect data: cbf_mode -> robot -> values
            cbf_data_by_mode = {}

            for robot_id, validator in exp_dict.items():
                robot_data = validator.cbf_data()  # Dict: cbf_mode -> values

                for cbf_mode, values in robot_data.items():
                    if cbf_mode not in cbf_data_by_mode:
                        cbf_data_by_mode[cbf_mode] = {}
                    cbf_data_by_mode[cbf_mode][robot_id] = values

            # Step 2: Create plots
            num_cbf_modes = len(cbf_data_by_mode)
            fig, axes = plt.subplots(num_cbf_modes, 1, figsize=(20, 5 * num_cbf_modes))

            if num_cbf_modes == 1:
                axes = [axes]

            for i, (cbf_mode, data_dict) in enumerate(cbf_data_by_mode.items()):
                self._plot_boxplot_on_ax(
                    ax=axes[i],
                    data_dict=data_dict,
                    ylabel=cbf_mode,
                    title=f"CBF mode: {cbf_mode}",
                    xlabel="Robot",
                )

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    experiment_modes = [0, 1, 2, 3]
    for i in experiment_modes:
        validater = ValidateSimulation(
            sim_dir=f"./runs/experiment_fake_success/simulation_results/fake_experiment_{i}"
        )
        validater()
    # for now debug on fake experiment
    # validate_cfg = {
    #     "exp_folders": [
    #         "./runs/experiment_fake_success",
    #         # "./runs/experiment_fabric_success",
    #         # "./runs/experiment_cluttered_success",
    #     ],
    #     "exp_colors": ["k", "g", "r", "b"],
    # }
    # validate_experiments = ValidateExperiments(validate_cfg=validate_cfg)
    # validate_experiments.creat_cbf_plots()
