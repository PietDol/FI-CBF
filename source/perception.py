# this file is created to simulate a sensor signal
import numpy as np
from cbfpy import CBF
from loguru import logger
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import copy
from obstacles import CircleObstacle, RectangleObstacle


class Sensor:
    # this class is created to simulate the sensor signal
    # basically this class generates the magnitude of the sensor signal
    # based on this magnitude we can calculate the noise (and thus epsilon for the safety margin)
    def __init__(self, sensor_position: np.ndarray, max_distance: float = 10):
        self.sensor_position = sensor_position  # in m
        self.max_distance = max_distance  # in m

    def get_sensor_magnitude(self, x_true: np.ndarray):
        # based on the true state, calculate the sensor magnitude (value between [0, 1])
        dist = jnp.linalg.norm(x_true - jnp.array(self.sensor_position))
        return jnp.clip(1 - dist / self.max_distance, 0.0, 1.0)

    def info(self, sensor_index: str = None):
        # print the info in the terminal
        if sensor_index is None:
            logger.info(
                f"Sensor: location [m]: {self.sensor_position}, max sensor distance [m]: {self.max_distance}"
            )
        else:
            logger.info(
                f"Sensor {sensor_index}: location [m]: {self.sensor_position}, max sensor distance [m]: {self.max_distance}"
            )


class Perception:
    # this class simulates the perception module of the robot
    # you can define the sensor location, if it is not define the will be random sampled over the costmap
    def __init__(
        self,
        costmap_size: np.ndarray,
        grid_size: np.ndarray,
        cbf: CBF,
        obstacles: list,
        env_dir: str,
        confidence_config: dict,
        min_values_state: np.ndarray,
        max_values_state: np.ndarray,
        max_sensor_noise: float,
        min_sensor_noise: float = 0.0,
        magnitude_threshold: float = 2.0,
        num_samples_per_dim: int = 4,
        sensors: list = None,
        load_lipschitz_grid_path: str = None,
    ):
        self.costmap_size = costmap_size
        self.cbf = cbf
        self.env_dir = env_dir
        self.min_values_state = min_values_state
        self.max_values_state = max_values_state
        self.max_sensor_noise = max_sensor_noise
        self.min_sensor_noise = min_sensor_noise
        self.magnitude_threshold = magnitude_threshold
        self.obstacles = obstacles
        self.conf_levels = confidence_config["levels"]
        self.percentiles = confidence_config["percentiles"]

        # estimate the lipschitz constants for the grid
        try:
            self.L_Lfh_grids, self.L_Lgh_grids = self.load_lipschitz_grids(
                load_lipschitz_grid_path
            )
        except:
            self.L_Lfh_grids, self.L_Lgh_grids = {}, {}
            for i, level in enumerate(self.conf_levels):
                v_max = confidence_config["vmax"][i]
                _min_values_state = np.array(
                    [min_values_state[0], min_values_state[1], -v_max, -v_max]
                )
                _max_values_state = np.array(
                    [max_values_state[0], max_values_state[1], v_max, v_max]
                )
                L_Lfh_grids, L_Lgh_grids = self.create_lipschitz_grid_3(
                    min_values_state=_min_values_state,
                    max_values_state=_max_values_state,
                    percentiles=self.percentiles,
                    num_points_per_dim_per_cell=num_samples_per_dim,
                    save_histogram=False,
                )
                self.L_Lfh_grids[f"{level}"] = L_Lfh_grids
                self.L_Lgh_grids[f"{level}"] = L_Lgh_grids

        # plot the grid
        # self.plot_lipschitz_grids(
        #     x_domain=np.linspace(
        #         min_values_state[0], max_values_state[0], costmap_size[0] + 1
        #     ),
        #     y_domain=np.linspace(
        #         min_values_state[1], max_values_state[1], costmap_size[1] + 1
        #     ),
        # )

        # save the lipschitz grids
        self.save_lipschitz_grids()

        # calculate the maximum difference in the grid
        self.max_L_Lfh_diff, self.max_L_Lgh_diff = self.calculate_max_lipschitz_grid_diff()

        # create lipschitz consants for different experiment modes
        self.L_Lfhs, self.L_Lghs = self.calculate_lipschitz_constants()

        # create the sensors if not given
        self.sensors = sensors
        self.num_sensors = len(self.sensors)

        # create list with sensor positions
        self.sensor_positions = [sensor.sensor_position for sensor in self.sensors]

        # log info to the terminal
        self.info()

        # create the costmap
        self.grid_size = grid_size
        self.origin_offset = np.array(costmap_size) / (2 * self.grid_size)
        self.perception_magnitude_costmap = self.create_costmap(
            costmap_type="perception"
        )
        logger.success("Perception magnitude costmap created")
        self.noise_costmap = self.create_costmap(costmap_type="noise")
        logger.success("Noise costmap created")

    #######################################################################
    # MAIN FUNCTIONS
    #######################################################################
    def add_sensor(self, sensor: Sensor):
        # add sensor to perception module
        self.sensors.append(sensor)
        self.sensor_positions.append(sensor.sensor_position)
        self.num_sensors += 1

        # update the costmaps
        self.perception_magnitude_costmap = self.create_costmap(
            costmap_type="perception"
        )
        self.noise_costmap = self.create_costmap(costmap_type="noise")
        logger.success(
            f"Sensor added (pos={sensor.sensor_position}) and perception magnitude and noise costmaps updated"
        )

    def info(self):
        # function to plot all the information
        [sensor.info(i) for i, sensor in enumerate(self.sensors)]

    def calculate_safety_margin(
        self,
        experiment_mode: int,
        noise: float,
        u_nominal: np.ndarray,
        k: float,
        reachable_set: np.ndarray,
        confidence_level: int,
        percentile: float,
        G: jnp.ndarray,
        h: jnp.ndarray,
    ):
        # wrapper function to calculate the safety margin, L_Lfh, L_Lgh
        # and the constraint matrices, G and h
        # structure of G and h:
        # index 0-3: constraints to stay in working domain
        # index 4-7: constraints to bound maximum velocity 
        # index 8-11: constraints to stay within the reachable set
        # round percentile to 1 decimal -> stored in self.L_Lfhs and self.L_Lghs
        percentile = np.round(percentile, 1)

        # calculate the safety margins based on the experiment mode
        if experiment_mode == 0:
            safety_margin, L_Lfh, L_Lgh = self.safety_margin_0(u_nominal)

            # constraint is that the system stays within the working domain
            # -> lipschitz constants only calculated for working domain
            G = G[:4]
            h = h[:4]
        elif experiment_mode == 1:
            safety_margin, L_Lfh, L_Lgh = self.safety_margin_1(
                noise=noise,
                u_nominal=u_nominal,
                k=k,
                confidence_level=confidence_level,
            )

            # constraints are to stay in working domain and bound v_max
            G = G[:8]
            h = h[:8]
        elif experiment_mode == 2:
            safety_margin, L_Lfh, L_Lgh = self.safety_margin_2(
                noise=noise,
                u_nominal=u_nominal,
                k=k,
                confidence_level=confidence_level,
                percentile=percentile,
            )

            # constraints are to stay in working domain and bound v_max
            G = G[:8]
            h = h[:8]
        elif experiment_mode == 3:
            safety_margin, L_Lfh, L_Lgh = self.safety_margin_3(
                noise=noise,
                u_nominal=u_nominal,
                k=k,
                reachable_set=reachable_set,
                confidence_level=confidence_level,
                percentile=percentile,
            )

            # constraints are to stay in working domain and reachable set and bound v_max
            # so just return the full G and h
        else:
            logger.error(f"Current experiment mode is not supported: {experiment_mode}")
            raise NotImplementedError

        return safety_margin, L_Lfh, L_Lgh, G, h

    #######################################################################
    # HELPER FUNCTIONS
    #######################################################################
    def get_estimated_state(self, true_state: np.array):
        # function to do the state estimation
        # it adds the given noise to the true state. if the shapes are not the same, the true state is returned
        # currently the noise is only added to the position
        true_pos = true_state[:2]
        std = self.get_perception_noise(x_true=true_pos)
        noise = np.zeros(true_state.shape)
        noise[:2] = np.random.normal(loc=0.0, scale=std, size=true_pos.shape)
        estimated_state = true_state + noise
        return estimated_state

    def get_perception_magnitude(self, x_true: np.ndarray):
        # based on the true state calculate the perception magnitude: mean of the sensor magnitudes
        magnitudes = np.array(
            [sensor.get_sensor_magnitude(x_true) for sensor in self.sensors]
        )
        return np.sum(magnitudes)

    def get_perception_noise(self, x_true: np.ndarray):
        # returns the standard deviation for the noise
        mag = self.get_perception_magnitude(x_true)
        if mag > self.magnitude_threshold:
            return self.min_sensor_noise
        else:
            return (
                self.max_sensor_noise
                + mag
                * (self.min_sensor_noise - self.max_sensor_noise)
                / self.magnitude_threshold
            )

    def get_perception_magnitude_batched(self, x_true: jnp.ndarray) -> jnp.ndarray:
        # batched version: compute perception magnitude for each position in (N, 2).
        def single_mag(x):
            mags = jnp.array(
                [sensor.get_sensor_magnitude(x) for sensor in self.sensors]
            )
            return jnp.sum(mags)

        return jax.vmap(single_mag)(x_true)  # (N,)

    def get_perception_noise_batched(self, x_true: jnp.ndarray) -> jnp.ndarray:
        # batched version: compute noise std for each position in (N, 2).
        mags = self.get_perception_magnitude_batched(x_true)  # (N,)
        return jnp.where(
            mags > self.magnitude_threshold,
            self.min_sensor_noise,
            self.max_sensor_noise
            + mags
            * (self.min_sensor_noise - self.max_sensor_noise)
            / self.magnitude_threshold,
        )

    def create_grid_samples(self, min_vals, max_vals, num_points_per_dim):
        # creates a grid for the min and max values with each num_points_per_dim
        axes = [
            jnp.linspace(lo, hi, num_points_per_dim)
            for lo, hi in zip(min_vals, max_vals)
        ]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, len(min_vals)
        )
        return grid  # (num_points_per_dim^d, state_space_dimensions)

    def get_epsilon(self, noise: float, k: float):
        # calculate the value of epsilon based on the values of the noise and k
        epsilon = k * noise
        return epsilon

    def _estimate_cbf_lipschitz_constants(
        self, num_points_per_dim: int = None, Z=None, analyze=False
    ):
        if Z is None:
            Z = self.create_grid_samples(
                min_vals=self.min_values_state,
                max_vals=self.max_values_state,
                num_points_per_dim=num_points_per_dim,
            )

        # K is the number of barrier functions
        # m is the size of the controller
        init_safety_margin = np.zeros(self.cbf.num_cbf)
        Lfhs = jax.vmap(lambda z: self.cbf.h_and_Lfh(z, init_safety_margin)[1])(
            Z
        )  # (N, K)
        Lghs = jax.vmap(lambda z: self.cbf.Lgh(z, init_safety_margin))(Z)  # (N, K, m)

        def estimate_lipschitz_scalar(values, inputs):
            """Estimate Lipschitz constant for each scalar output"""
            N, K = values.shape
            lipschitz_per_output = []

            for k in range(K):
                y = values[:, k]  # (N,)
                diffs_x = inputs[:, None, :] - inputs[None, :, :]
                diffs_y = y[:, None] - y[None, :]

                dx = jnp.linalg.norm(diffs_x, axis=-1)
                dx = jnp.where(dx < 1e-6, 1e-6, dx)

                dy = jnp.abs(diffs_y)
                lipschitz_matrix = dy / dx
                lipschitz_matrix = jnp.nan_to_num(
                    lipschitz_matrix, nan=0.0, posinf=0.0, neginf=0.0
                )

                if analyze:
                    # add flat values
                    lipschitz_per_output.append(
                        jnp.triu(lipschitz_matrix, k=1).flatten()
                    )
                else:
                    # add max value
                    lipschitz_per_output.append(
                        jnp.max(jnp.triu(lipschitz_matrix, k=1))
                    )

            return jnp.array(lipschitz_per_output)

        def estimate_lipschitz_vector(values, inputs):
            """Estimate Lipschitz constant per vector-valued output (max over control dim)"""
            N, K, m = values.shape
            lipschitz_per_barrier = []

            for k in range(K):
                y = values[:, k, :]  # (N, m)
                diffs_x = inputs[:, None, :] - inputs[None, :, :]  # (N, N, D)
                diffs_y = y[:, None, :] - y[None, :, :]  # (N, N, m)

                dx = jnp.linalg.norm(diffs_x, axis=-1)
                dx = jnp.where(dx < 1e-6, 1e-6, dx)

                dy = jnp.linalg.norm(diffs_y, axis=-1)  # vector norm over control dim
                lipschitz_matrix = dy / dx
                lipschitz_matrix = jnp.nan_to_num(
                    lipschitz_matrix, nan=0.0, posinf=0.0, neginf=0.0
                )

                # return output based on mode
                if analyze:
                    # add flat values
                    lipschitz_per_barrier.append(
                        jnp.triu(lipschitz_matrix, k=1).flatten()
                    )
                else:
                    # add max value
                    lipschitz_per_barrier.append(
                        jnp.max(jnp.triu(lipschitz_matrix, k=1))
                    )

            return jnp.array(lipschitz_per_barrier)

        L_Lfh = estimate_lipschitz_scalar(Lfhs, Z)  # * 0.3  # (K,)
        L_Lgh = estimate_lipschitz_vector(Lghs, Z)  # * 0.3  # (K,)

        # only print if we are not analyzing
        if not analyze:
            logger.info(f"L_Lfh per barrier: {L_Lfh}")
            logger.info(f"L_Lgh per barrier: {L_Lgh}")

        return np.array(L_Lfh), np.array(L_Lgh)

    @staticmethod
    def is_square_fully_inside_circle(
        square_center, square_size, circle_center, circle_radius
    ):
        half_size = square_size / 2

        # Compute the coordinates of the square corners
        corners = np.array(
            [
                square_center + [-half_size, -half_size],
                square_center + [-half_size, half_size],
                square_center + [half_size, -half_size],
                square_center + [half_size, half_size],
            ]
        )

        # Check if all corners are within the circle
        distances = np.linalg.norm(corners - circle_center, axis=1)
        return np.all(distances <= circle_radius)

    def _lipschitz_constant_helper(self, confidence_level: int, percentile: float):
        # helper function to calculate the lipschitz constants for given confidence level and percentile
        L_Lfhs, L_Lghs = [], []
        percentile = np.round(percentile, 1)
        obstacle_masks = self.create_obstacle_masks()

        # make sure that the grids inside the obstacles are not taken into account
        for i in range(len(self.obstacles)):
            # only take max of values which are not inside obstacles
            L_Lfhs.append(
                np.amax(
                    self.L_Lfh_grids[f"{confidence_level}"][f"{percentile}"][:, :, i][
                        obstacle_masks[i]
                    ]
                )
            )
            L_Lghs.append(
                np.amax(
                    self.L_Lgh_grids[f"{confidence_level}"][f"{percentile}"][:, :, i][
                        obstacle_masks[i]
                    ]
                )
            )

        # convert to numpy and log values
        L_Lfhs = np.array(L_Lfhs)
        L_Lghs = np.array(L_Lghs)

        return L_Lfhs, L_Lghs

    def create_obstacle_masks(self):
        # function to create the masks of obstacles
        masks = []
        # make sure that the grids inside the obstacles are not taken into account
        for obstacle in self.obstacles:
            # get max values in world coordinates
            x_min = obstacle.pos_center[0] - (obstacle.radius + obstacle.robot_radius)
            x_max = obstacle.pos_center[0] + (obstacle.radius + obstacle.robot_radius)
            y_min = obstacle.pos_center[1] - (obstacle.radius + obstacle.robot_radius)
            y_max = obstacle.pos_center[1] + (obstacle.radius + obstacle.robot_radius)

            origin_offset = np.array(self.costmap_size) / 2
            # convert to indices of the grid
            col_min_ind = int(np.floor(x_min + origin_offset[1]))
            col_max_ind = int(np.floor(x_max + origin_offset[1]))
            row_min_ind = int(np.floor(y_min + origin_offset[0]))
            row_max_ind = int(np.floor(y_max + origin_offset[0]))

            # create mask
            mask = np.ones(
                self.L_Lfh_grids[f"1"][f"100.0"].shape[:2],
                dtype=bool,
            )
            if isinstance(obstacle, RectangleObstacle):
                raise NotImplementedError
            elif isinstance(obstacle, CircleObstacle):
                for col in range(col_min_ind, col_max_ind + 1):
                    for row in range(row_min_ind, row_max_ind + 1):
                        square_center = np.array([row, col])[::-1] + 0.5 - origin_offset
                        mask[row, col] = not (
                            self.is_square_fully_inside_circle(
                                square_center=square_center,
                                square_size=1.0,
                                circle_center=obstacle.pos_center,
                                circle_radius=obstacle.radius,
                            )
                        )

            # add the mask to the masks
            masks.append(mask)
        
        # convert to numpy and return it
        masks = np.array(masks)
        return masks

    @staticmethod
    def calculate_max_diff_grid(grid, mask):
        # Compute the maximum difference between adjacent grid cells,
        # considering only those where both involved cells are marked True in the mask.
        diffs = []

        # Horizontal (left-right)
        valid_h = mask[:, :-1] & mask[:, 1:]
        diff_h = np.abs(grid[:, :-1] - grid[:, 1:])
        diffs.append(diff_h[valid_h])

        # Vertical (top-bottom)
        valid_v = mask[:-1, :] & mask[1:, :]
        diff_v = np.abs(grid[:-1, :] - grid[1:, :])
        diffs.append(diff_v[valid_v])

        # Diagonal ↘
        valid_d1 = mask[:-1, :-1] & mask[1:, 1:]
        diff_d1 = np.abs(grid[:-1, :-1] - grid[1:, 1:])
        diffs.append(diff_d1[valid_d1])

        # Diagonal ↙
        valid_d2 = mask[:-1, 1:] & mask[1:, :-1]
        diff_d2 = np.abs(grid[:-1, 1:] - grid[1:, :-1])
        diffs.append(diff_d2[valid_d2])

        # Concatenate all valid diffs and compute max
        if any(d.size > 0 for d in diffs):
            max_diff = np.max(np.concatenate([d for d in diffs if d.size > 0]))
        else:
            logger.error("Not able to calculate the maximum difference in the Lipschitz grid!")

        return max_diff

    #######################################################################
    # PRECALCULATIONS FOR THE SAFETY MARGINS
    #######################################################################
    def calculate_lipschitz_constants(self):
        # function to calculate the lipschitz constants for experiment mode 0, 1 and 2
        # create the dict to store values for each experiment mode -> also dict with list for saving
        L_Lfhs = {
            0: None,
            1: {},
            2: {},
        }
        L_Lghs = {
            0: None,
            1: {},
            2: {},
        }
        L_Lfhs_save = {
            0: None,
            1: {},
            2: {},
        }
        L_Lghs_save = {
            0: None,
            1: {},
            2: {},
        }

        # experiment mode 0:
        # lipschitz constants are absolute maximum value -> level 1, percentile 100
        L_Lfhs_0, L_Lghs_0 = self._lipschitz_constant_helper(1, 100.0)
        L_Lfhs[0] = L_Lfhs_0
        L_Lghs[0] = L_Lghs_0
        L_Lfhs_save[0] = L_Lfhs_0.tolist()
        L_Lghs_save[0] = L_Lghs_0.tolist()

        # experiment mode 1:
        # iterate over the confidence level and take max value -> percentile 100
        for conf_level in self.conf_levels:
            L_Lfhs_1, L_Lghs_1 = self._lipschitz_constant_helper(conf_level, 100.0)
            L_Lfhs[1][conf_level] = L_Lfhs_1
            L_Lghs[1][conf_level] = L_Lghs_1
            L_Lfhs_save[1][conf_level] = L_Lfhs_1.tolist()
            L_Lghs_save[1][conf_level] = L_Lghs_1.tolist()

        # experiment mode 2:
        # iterate over the confidence level and percentile
        for conf_level in self.conf_levels:
            conf_dict_L_Lfh, conf_dict_L_Lgh = {}, {}
            conf_dict_L_Lfh_save, conf_dict_L_Lgh_save = {}, {}
            for percentile in self.percentiles:
                percentile = np.round(percentile, 1)
                L_Lfhs_2, L_Lghs_2 = (
                    self._lipschitz_constant_helper(conf_level, percentile)
                )
                conf_dict_L_Lfh[f"{percentile}"] = L_Lfhs_2
                conf_dict_L_Lgh[f"{percentile}"] = L_Lghs_2
                conf_dict_L_Lfh_save[f"{percentile}"] = L_Lfhs_2.tolist()
                conf_dict_L_Lgh_save[f"{percentile}"] = L_Lghs_2.tolist()

            # add dict for confidence to L_Lfhs and L_Lghs
            L_Lfhs[2][conf_level] = conf_dict_L_Lfh
            L_Lghs[2][conf_level] = conf_dict_L_Lgh
            L_Lfhs_save[2][conf_level] = conf_dict_L_Lfh_save
            L_Lghs_save[2][conf_level] = conf_dict_L_Lgh_save

        # save the dicts in the env folder
        with open(f"{self.env_dir}/L_Lfh_constants.json", "w") as L_Lfh_file:
            json.dump(L_Lfhs_save, L_Lfh_file, indent=4)
        logger.success(
            f"L_Lfh for experiment 0, 1, and 2 saved: {self.env_dir}/L_Lfh_constants.json"
        )
        with open(f"{self.env_dir}/L_Lgh_constants.json", "w") as L_Lgh_file:
            json.dump(L_Lghs_save, L_Lgh_file, indent=4)
        logger.success(
            f"L_Lgh for experiment 0, 1, and 2 saved: {self.env_dir}/L_Lgh_constants.json"
        )

        return L_Lfhs, L_Lghs

    def create_lipschitz_grid_3(
        self,
        min_values_state: np.ndarray,
        max_values_state: np.ndarray,
        percentiles: list | np.ndarray | tuple,
        num_points_per_dim_per_cell: int,
        save_histogram: bool = False,
    ):
        # set some important parameters
        num_barriers = self.cbf.num_cbf
        cell_grid = self.costmap_size
        lipschitz_dir = f"{self.env_dir}/lipschitz_constants_grid"

        # create dirs to save the values
        os.makedirs(lipschitz_dir, exist_ok=True)
        os.makedirs(f"{lipschitz_dir}/visuals", exist_ok=True)
        os.makedirs(f"{lipschitz_dir}/data", exist_ok=True)

        # create linspaces
        x_domain = np.linspace(
            min_values_state[0], max_values_state[0], cell_grid[0] + 1
        )
        y_domain = np.linspace(
            min_values_state[1], max_values_state[1], cell_grid[1] + 1
        )

        # create grids
        L_Lfh_grid, L_Lgh_grid = {}, {}
        for percentile in percentiles:
            percentile = np.round(percentile, 1)
            L_Lfh_grid[f"{percentile}"] = np.zeros(
                (cell_grid[1], cell_grid[0], num_barriers)
            )
            L_Lgh_grid[f"{percentile}"] = np.zeros(
                (cell_grid[1], cell_grid[0], num_barriers)
            )

        # iterate over all the cells
        for i in tqdm(range(len(x_domain) - 1), desc="Calculate Lipschitz for grid"):
            for j in range(len(y_domain) - 1):
                min_values = np.array(
                    [
                        x_domain[i],
                        y_domain[j],
                        min_values_state[2],
                        min_values_state[3],
                    ]
                )
                max_values = np.array(
                    [
                        x_domain[i + 1],
                        y_domain[j + 1],
                        max_values_state[2],
                        max_values_state[3],
                    ]
                )

                # generate grid for this cell
                Z = self.create_grid_samples(
                    min_values, max_values, num_points_per_dim_per_cell
                )

                # estimate Lipschitz values for this batch
                L_Lfhs, L_Lghs = self._estimate_cbf_lipschitz_constants(
                    Z=Z, analyze=True
                )
                L_Lfhs = np.array(L_Lfhs)
                L_Lghs = np.array(L_Lghs)

                # fill histograms
                if save_histogram:
                    fig, axes = plt.subplots(2, num_barriers, figsize=(12, 6))
                    for k in range(num_barriers):  # for each barrier function
                        # create the histograms
                        axes[0, k].hist(
                            L_Lfhs[k, :], bins=40, color="steelblue", edgecolor="black"
                        )
                        axes[0, k].set_title(
                            f"Lipschitz value distribution for Lfh[{k}] (num_points={L_Lfhs.shape[1]})"
                        )
                        axes[0, k].set_xlabel("Lfh value")
                        axes[0, k].set_ylabel("Frequency")
                        axes[0, k].grid(True)

                        axes[1, k].hist(
                            L_Lghs[k, :], bins=40, color="darkorange", edgecolor="black"
                        )
                        axes[1, k].set_title(
                            f"Lipschitz value distribution for Lgh[{k}] (num_points={L_Lghs.shape[1]})"
                        )
                        axes[1, k].set_xlabel("Lgh value")
                        axes[1, k].set_ylabel("Frequency")
                        axes[1, k].grid(True)

                    plt.tight_layout()
                    plt.savefig(
                        f"{lipschitz_dir}/visuals/lipschitz_constants_{i}_{j}.png"
                    )
                    plt.close()

                # update the grids
                for key in L_Lfh_grid.keys():
                    if key == "100":
                        L_Lfh_grid[key][j, i] = np.amax(L_Lfhs, axis=1)
                        L_Lgh_grid[key][j, i] = np.amax(L_Lghs, axis=1)
                    else:
                        L_Lfh_grid[key][j, i] = np.percentile(
                            L_Lfhs, float(key), axis=1
                        )
                        L_Lgh_grid[key][j, i] = np.percentile(
                            L_Lghs, float(key), axis=1
                        )

        # iterate over the grids to save them
        for key in L_Lfh_grid.keys():
            # save the grids
            np.save(
                f"{lipschitz_dir}/data/L_Lfh_grid_{key}_{max_values_state[2]}.npy",
                L_Lfh_grid[key],
            )
            np.save(
                f"{lipschitz_dir}/data/L_Lgh_grid_{key}_{max_values_state[2]}.npy",
                L_Lgh_grid[key],
            )

        return L_Lfh_grid, L_Lgh_grid

    def calculate_max_lipschitz_grid_diff(self):
        # function to calculate the maximum difference in the grids
        num_obstacles = len(self.obstacles)
        max_L_Lfh_diff, max_L_Lgh_diff = {}, {}
        obstacle_masks = self.create_obstacle_masks()

        # iterate over confidence levels and percentiles to get all the differences
        for conf_level in self.conf_levels:
            max_L_Lfh_diff_conf, max_L_Lgh_conf = {}, {}
            for percentile in self.percentiles:
                # set some parameters
                percentile = np.round(percentile, 1)
                _max_L_Lfh_diff, _max_L_Lgh_diff = np.zeros(num_obstacles), np.zeros(
                    num_obstacles
                )

                # iterate over the obstacles
                for i in range(num_obstacles):
                    # extract the grids
                    L_Lfh_grid = self.L_Lfh_grids[f"{conf_level}"][f"{percentile}"][:, :, i]
                    L_Lgh_grid = self.L_Lgh_grids[f"{conf_level}"][f"{percentile}"][:, :, i]

                    # calculate the maximum difference
                    _max_L_Lfh_diff[i] = self.calculate_max_diff_grid(L_Lfh_grid, obstacle_masks[i])
                    _max_L_Lgh_diff[i] = self.calculate_max_diff_grid(L_Lgh_grid, obstacle_masks[i])

                # add to dicts
                max_L_Lfh_diff_conf[f"{percentile}"] = _max_L_Lfh_diff
                max_L_Lgh_conf[f"{percentile}"] = _max_L_Lgh_diff

            # add dict to overall dict
            max_L_Lfh_diff[conf_level] = max_L_Lfh_diff_conf
            max_L_Lgh_diff[conf_level] = max_L_Lgh_conf

        # save the dictionaries
        max_L_Lfh_diff_save = copy.deepcopy(max_L_Lfh_diff)
        max_L_Lgh_diff_save = copy.deepcopy(max_L_Lgh_diff)

        # convert to list
        for conf_level in max_L_Lfh_diff_save.keys():
            for percentile in max_L_Lfh_diff_save[conf_level].keys():
                max_L_Lfh_diff_save[conf_level][percentile] = max_L_Lfh_diff_save[
                    conf_level
                ][percentile].tolist()
                max_L_Lgh_diff_save[conf_level][percentile] = max_L_Lgh_diff_save[
                    conf_level
                ][percentile].tolist()
        
        # and save it
        with open(f"{self.env_dir}/L_Lfh_grid_diffs.json", "w") as L_Lfh_file:
            json.dump(max_L_Lfh_diff_save, L_Lfh_file, indent=4)
        logger.success(
            f"Max Lipschitz grid diff for L_Lfh saved: {self.env_dir}/L_Lfh_grid_diffs.json"
        )
        with open(f"{self.env_dir}/L_Lgh_grid_diffs.json", "w") as L_Lgh_file:
            json.dump(max_L_Lgh_diff_save, L_Lgh_file, indent=4)
        logger.success(
            f"Max Lipschitz grid diff for L_Lgh saved: {self.env_dir}/L_Lgh_grid_diffs.json"
        )

        # return the differences
        return max_L_Lfh_diff, max_L_Lgh_diff

    def save_lipschitz_grids(self):
        L_Lfh_grids_to_save = copy.deepcopy(self.L_Lfh_grids)
        L_Lgh_grids_to_save = copy.deepcopy(self.L_Lgh_grids)

        # convert the np.arrays to list
        for level in L_Lfh_grids_to_save.keys():
            for percentile in L_Lfh_grids_to_save[level].keys():
                L_Lfh_grids_to_save[level][percentile] = L_Lfh_grids_to_save[level][
                    percentile
                ].tolist()
                L_Lgh_grids_to_save[level][percentile] = L_Lgh_grids_to_save[level][
                    percentile
                ].tolist()

        # save all the grids
        with open(f"{self.env_dir}/L_Lfh_grids.json", "w") as L_Lfh_file:
            json.dump(L_Lfh_grids_to_save, L_Lfh_file, indent=4)
        logger.success(
            f"Lipschitz grid for L_Lfh saved: {self.env_dir}/L_Lfh_grids.json"
        )
        with open(f"{self.env_dir}/L_Lgh_grids.json", "w") as L_Lgh_file:
            json.dump(L_Lgh_grids_to_save, L_Lgh_file, indent=4)
        logger.success(
            f"Lipschitz grid for L_Lgh saved: {self.env_dir}/L_Lgh_grids.json"
        )

    def load_lipschitz_grids(self, load_env_dir):
        # load saved grids back into the object
        with open(f"{load_env_dir}/L_Lfh_grids.json", "r") as L_Lfh_file:
            L_Lfh_grids = json.load(L_Lfh_file)
        with open(f"{load_env_dir}/L_Lgh_grids.json", "r") as L_Lgh_file:
            L_Lgh_grids = json.load(L_Lgh_file)

        # convert the lists to numpy array
        for level in L_Lfh_grids.keys():
            for percentile in L_Lfh_grids[level].keys():
                L_Lfh_grids[level][percentile] = np.array(
                    L_Lfh_grids[level][percentile]
                )
                L_Lgh_grids[level][percentile] = np.array(
                    L_Lgh_grids[level][percentile]
                )

        # log that loading grids is successfull
        logger.success(f"Loading Lipschitz grids done")

        # return the numpy grids
        return L_Lfh_grids, L_Lgh_grids

    def plot_lipschitz_grids(
        self,
        x_domain: np.ndarray,
        y_domain: np.ndarray,
    ):
        # plot the grids
        # some parameters
        num_barriers = self.cbf.num_cbf
        cell_grid = self.costmap_size
        lipschitz_dir = f"{self.env_dir}/lipschitz_constants_grid"
        extent = [x_domain[0], x_domain[-1], y_domain[0], y_domain[-1]]

        # create dirs for visuals
        os.makedirs(f"{lipschitz_dir}/visuals", exist_ok=True)

        # iterate over the grids
        for confidence_key, percentiles_dict in self.L_Lfh_grids.items():
            for percentile_key in percentiles_dict.keys():
                for i in range(num_barriers):
                    # create the figure
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                    # L_Lfh
                    im1 = axes[0].imshow(
                        self.L_Lfh_grids[confidence_key][percentile_key][:, :, i],
                        origin="lower",
                        extent=extent,
                        cmap="Blues",
                    )
                    axes[0].set_title(
                        f"L_Lfh {percentile_key}% percentile grid [Barrier {i}]"
                    )
                    axes[0].grid(True)
                    axes[0].axis("equal")
                    fig.colorbar(im1, ax=axes[0])

                    # annotate each cell with the max value
                    for xi in range(cell_grid[1]):
                        for yi in range(cell_grid[0]):
                            # Get center of cell
                            x_mid = 0.5 * (x_domain[xi] + x_domain[xi + 1])
                            y_mid = 0.5 * (y_domain[yi] + y_domain[yi + 1])
                            val = self.L_Lfh_grids[confidence_key][percentile_key][
                                yi, xi, i
                            ]
                            axes[0].text(
                                x_mid,
                                y_mid,
                                f"{val:.2f}",
                                color="black",
                                ha="center",
                                va="center",
                                fontsize=5,
                            )

                    # L_Lgh
                    im2 = axes[1].imshow(
                        self.L_Lgh_grids[confidence_key][percentile_key][:, :, i],
                        origin="lower",
                        extent=extent,
                        cmap="Oranges",
                    )
                    axes[1].set_title(
                        f"L_Lgh {percentile_key}% percentile grid [Barrier {i}]"
                    )
                    axes[1].grid(True)
                    axes[1].axis("equal")
                    fig.colorbar(im2, ax=axes[1])

                    # annotate each cell with the max value
                    for xi in range(cell_grid[1]):
                        for yi in range(cell_grid[0]):
                            x_mid = 0.5 * (x_domain[xi] + x_domain[xi + 1])
                            y_mid = 0.5 * (y_domain[yi] + y_domain[yi + 1])
                            val = self.L_Lgh_grids[confidence_key][percentile_key][
                                yi, xi, i
                            ]
                            axes[1].text(
                                x_mid,
                                y_mid,
                                f"{val:.2f}",
                                color="black",
                                ha="center",
                                va="center",
                                fontsize=5,
                            )

                    plt.tight_layout()
                    plt.savefig(
                        f"{lipschitz_dir}/visuals/grid_{confidence_key}_{percentile_key}_barrier_{i}.png"
                    )
                    plt.close()
                    logger.success(
                        f"Grid for confidence {confidence_key}, {percentile_key}% and barrier {i} saved: {lipschitz_dir}/visuals/grid_{confidence_key}_{percentile_key}_barrier_{i}.png"
                    )

    #######################################################################
    # DIFFERENT SAFETY MARGIN MODES
    #######################################################################
    def safety_margin_0(self, u_nominal: np.ndarray):
        # mode 0: baseline mrcbf paper
        # safety margin calculation based on the mrcbf paper
        L_alpha_h = 1.0

        # 3 * noise is 99,7% confidence interval so 4 is closer to robust
        epsilon = 4 * self.max_sensor_noise  # in the paper they use 0.4 for max noise
        a = (self.L_Lfhs[0] + L_alpha_h) * epsilon
        b = self.L_Lghs[0] * epsilon
        safety_margin = a + b * jnp.linalg.norm(u_nominal)
        return safety_margin, self.L_Lfhs[0], self.L_Lghs[0]

    def safety_margin_1(
        self,
        noise: float,
        u_nominal: np.ndarray,
        k: float,
        confidence_level: int,
    ):
        # mode 1: global maximum based on the confidence level
        # Assume alpha(h) = h, so L_alpha_h = 1
        L_alpha_h = 1.0

        # calculate epsilon
        epsilon = self.get_epsilon(noise, k)

        # get the values of L_Lfh and L_Lgh
        L_Lfh = self.L_Lfhs[1][confidence_level]
        L_Lgh = self.L_Lghs[1][confidence_level]

        # calculate the safety margin
        a = (L_Lfh + L_alpha_h) * epsilon
        b = L_Lgh * epsilon
        safety_margin = a + b * jnp.linalg.norm(u_nominal)
        return safety_margin, L_Lfh, L_Lgh

    def safety_margin_2(
        self,
        noise: float,
        u_nominal: np.ndarray,
        k: float,
        confidence_level: int,
        percentile: float,
    ):
        # mode 2: risk aware approach with global maximum on the percentiles
        # Assume alpha(h) = h, so L_alpha_h = 1
        L_alpha_h = 1.0

        # calculate epsilon
        epsilon = self.get_epsilon(noise, k)

        # get the values of L_Lfh and L_Lgh
        L_Lfh = self.L_Lfhs[2][confidence_level][f"{percentile}"]
        L_Lgh = self.L_Lghs[2][confidence_level][f"{percentile}"]

        # calculate the safety margin
        a = (L_Lfh + L_alpha_h) * epsilon
        b = L_Lgh * epsilon
        safety_margin = a + b * jnp.linalg.norm(u_nominal)
        return safety_margin, L_Lfh, L_Lgh

    def safety_margin_3(
        self,
        noise: float,
        u_nominal: np.ndarray,
        k: float,
        reachable_set: np.ndarray,
        confidence_level: int,
        percentile: float,
    ):
        # mode 3: risk aware horizon approach
        # Converts the uncertainty to the safety margin that needs to be used by the CBFs to
        # account for estimation uncertainty. Epsilon is upper bound on estimation error
        # Assume alpha(h) = h, so L_alpha_h = 1
        L_alpha_h = 1.0

        # calculate epsilon
        epsilon = self.get_epsilon(noise, k)

        # calculate the lipschitz constants based on the grid
        # get the indices of the grid
        indices = []
        origin_offset = np.array(self.costmap_size) / 2
        for x in reachable_set[0]:
            for y in reachable_set[1]:
                grid = np.floor(np.array([x, y]) + origin_offset).astype(int)
                indices.append(grid[::-1])
        indices = np.array(indices)  # (4, 2)

        # calculate the range of the indices
        row_min, col_min = np.amin(indices, axis=0)
        row_max, col_max = np.amax(indices, axis=0)
        rows = np.arange(
            max(row_min, 0), min(row_max + 1, self.costmap_size[0] - 1)
        )  # +1 because the stop must be included
        cols = np.arange(
            max(col_min, 0), min(col_max + 1, self.costmap_size[1] - 1)
        )  # +1 because the stop must be included

        # get the lipschitz values from the grid
        L_Lfhs, L_Lghs = [], []
        for i in rows:
            for j in cols:
                L_Lfhs.append(
                    self.L_Lfh_grids[f"{confidence_level}"][f"{percentile}"][i, j]
                )
                L_Lghs.append(
                    self.L_Lgh_grids[f"{confidence_level}"][f"{percentile}"][i, j]
                )

        # also add the maximum difference to the value of L_Lfh and L_Lgh
        max_L_Lfh_diff = self.max_L_Lfh_diff[confidence_level][f"{percentile}"]
        max_L_Lgh_diff = self.max_L_Lgh_diff[confidence_level][f"{percentile}"]
        L_Lfh = np.amax(np.array(L_Lfhs), axis=0) + max_L_Lfh_diff
        L_Lgh = np.amax(np.array(L_Lghs), axis=0) + max_L_Lgh_diff

        # calculate the new safety margin
        a = (L_Lfh + L_alpha_h) * epsilon
        b = L_Lgh * epsilon
        safety_margin = a + b * jnp.linalg.norm(u_nominal)
        return safety_margin, L_Lfh, L_Lgh

    #######################################################################
    # COSTMAP PART
    #######################################################################
    def grid_to_world(self, idx):
        # Convert grid index (row, col) to world coordinate (x, y) in meters. It returns the center of the grid.
        ij = np.array(idx[::-1])
        pos = (
            (ij * self.grid_size)
            + (0.5 * self.grid_size)
            - (np.array(self.origin_offset) * self.grid_size)
        )
        return pos

    def create_costmap(self, costmap_type: str):
        rows, cols = int(self.costmap_size[0] / self.grid_size), int(
            self.costmap_size[1] / self.grid_size
        )
        row_idx, col_idx = np.indices((rows, cols))
        ij = np.stack((col_idx, row_idx), axis=-1).reshape(-1, 2)  # convert to (N, 2)

        # convert grid to world pos (N, 2)
        pos = (
            (ij * self.grid_size)
            + (0.5 * self.grid_size)
            - (np.array(self.origin_offset) * self.grid_size)
        )

        # calculate the uncertainty/noise
        if costmap_type == "noise":
            costmap = self.get_perception_noise_batched(jnp.array(pos))  # shape (N,)
        elif costmap_type == "perception":
            costmap = self.get_perception_magnitude_batched(
                jnp.array(pos)
            )  # shape (N,)
        else:
            logger.error(
                f"Wrong costmap_type: {costmap_type}. Choose 'noise' or 'perception'"
            )
            costmap = np.zeros(ij.shape[0])
        return np.array(costmap).reshape(rows, cols)
