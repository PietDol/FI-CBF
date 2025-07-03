# this file is created to simulate a sensor signal
import numpy as np
from cbfpy import CBF
from loguru import logger
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
import sys


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
        env_dir: str,
        confidence_config: dict,
        min_values_state: np.ndarray,
        max_values_state: np.ndarray,
        max_sensor_noise: float,
        min_sensor_noise: float = 0.0,
        magnitude_threshold: float = 2.0,
        num_samples_per_dim: int = 4,
        sensors: list = None,
    ):
        self.costmap_size = costmap_size
        self.cbf = cbf
        self.min_values_state = min_values_state
        self.max_values_state = max_values_state
        self.max_sensor_noise = max_sensor_noise
        self.min_sensor_noise = min_sensor_noise
        self.magnitude_threshold = magnitude_threshold

        # estimate the lipschitz constants for the grid
        self.L_Lfh_grids, self.L_Lgh_grids = {}, {}
        for i in range(len(confidence_config["levels"])):
            v_max = confidence_config["vmax"][i]
            _min_values_state = np.array(
                [min_values_state[0], min_values_state[1], -v_max, -v_max]
            )
            _max_values_state = np.array(
                [max_values_state[0], max_values_state[1], v_max, v_max]
            )
            L_Lfh_grids, L_Lgh_grids = self.analyze_lipschitz_grid(
                min_values_state=_min_values_state,
                max_values_state=_max_values_state,
                num_points_per_dim_per_cell=num_samples_per_dim,
                env_dir=env_dir,
                save_histogram=False,
            )
            self.L_Lfh_grids[f"{i}"] = L_Lfh_grids
            self.L_Lgh_grids[f"{i}"] = L_Lgh_grids

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

    def calculate_safety_margin(
        self, noise: float, u_nominal: np.ndarray, k: float, reachable_set: np.ndarray, confidence_level: int,
    ):
        # Converts the uncertainty to the safety margin that needs to be used by the CBFs to
        # account for estimation uncertainty. Epsilon is upper bound on estimation error
        # Assume alpha(h) = h, so L_alpha_h = 1
        L_alpha_h = 1.0

        # calculate epsilon in the paper they use eps=0.4
        epsilon = self.get_epsilon(noise, k)

        # calculate the lipschitz constants based on the grid
        # get the indices of the grid
        indices = []
        origin_offset = np.array(self.costmap_size) / 2
        for x in reachable_set[0]:
            for y in reachable_set[1]:
                grid = np.floor((np.array([x, y]) + origin_offset)).astype(int)
                indices.append(grid[::-1])
        indices = np.array(indices)  # (4, 2)

        # calculate the range of the indices
        row_min, col_min = np.amin(indices, axis=0)
        row_max, col_max = np.amax(indices, axis=0)
        rows = np.arange(row_min, row_max + 1)  # +1 because the stop must be included
        cols = np.arange(col_min, col_max + 1)  # +1 because the stop must be included

        # get the lipschitz values from the grid
        # for now tak 90% percentile
        L_Lfhs, L_Lghs = [], []
        for i in rows:
            for j in cols:
                L_Lfhs.append(self.L_Lfh_grids[f"{confidence_level}"]["80"][i, j])
                L_Lghs.append(self.L_Lgh_grids[f"{confidence_level}"]["80"][i, j])
        L_Lfh = np.amax(np.array(L_Lfhs), axis=0)
        L_Lgh = np.amax(np.array(L_Lghs), axis=0)

        # calculate the new safety margin
        a = (L_Lfh + L_alpha_h) * epsilon
        b = L_Lgh * epsilon
        safety_margin = a + b * jnp.linalg.norm(u_nominal)
        return safety_margin

    def analyze_lipschitz_grid(
        self,
        min_values_state: np.ndarray,
        max_values_state: np.ndarray,
        num_points_per_dim_per_cell: int,
        env_dir: str,
        save_histogram: bool = False,
    ):
        # set some important parameters
        num_barriers = self.cbf.num_cbf
        cell_grid = self.costmap_size

        # create dir to save the values
        os.makedirs(f"{env_dir}/lipschitz_constants_grid", exist_ok=True)

        # create linspaces
        x_domain = np.linspace(
            min_values_state[0], max_values_state[0], cell_grid[0] + 1
        )
        y_domain = np.linspace(
            min_values_state[1], max_values_state[1], cell_grid[1] + 1
        )

        # create grids
        percentiles = ["80", "90", "95", "100"]
        L_Lfh_grid, L_Lgh_grid = {}, {}
        for percentile in percentiles:
            L_Lfh_grid[percentile] = np.zeros(
                (cell_grid[1], cell_grid[0], num_barriers)
            )
            L_Lgh_grid[percentile] = np.zeros(
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
                        f"{env_dir}/lipschitz_constants_grid/lipschitz_constants_{i}_{j}.png"
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

        # iterate over the grids
        for key in L_Lfh_grid.keys():
            # save the grids
            np.save(
                f"{env_dir}/lipschitz_constants_grid/L_Lfh_grid_{key}_{max_values_state[2]}.npy",
                L_Lfh_grid[key],
            )
            np.save(
                f"{env_dir}/lipschitz_constants_grid/L_Lgh_grid_{key}_{max_values_state[2]}.npy",
                L_Lgh_grid[key],
            )

            # plot the grids
            fig, axes = plt.subplots(2, num_barriers, figsize=(12, 10))
            extent = [x_domain[0], x_domain[-1], y_domain[0], y_domain[-1]]
            for i in range(num_barriers):
                # L_Lfh
                im1 = axes[0, i].imshow(
                    L_Lfh_grid[key][:, :, i],
                    origin="lower",
                    extent=extent,
                    cmap="Blues",
                )
                axes[0, i].set_title(f"L_Lfh {key}% percentile grid [Barrier {i}]")
                axes[0, i].grid(True)
                axes[0, i].axis("equal")
                fig.colorbar(im1, ax=axes[0, i])

                # annotate each cell with the max value
                for xi in range(cell_grid[1]):
                    for yi in range(cell_grid[0]):
                        # Get center of cell
                        x_mid = 0.5 * (x_domain[xi] + x_domain[xi + 1])
                        y_mid = 0.5 * (y_domain[yi] + y_domain[yi + 1])
                        val = L_Lfh_grid[key][yi, xi, i]
                        axes[0, i].text(
                            x_mid,
                            y_mid,
                            f"{val:.2f}",
                            color="black",
                            ha="center",
                            va="center",
                            fontsize=5,
                        )

                # L_Lgh
                im2 = axes[1, i].imshow(
                    L_Lgh_grid[key][:, :, i],
                    origin="lower",
                    extent=extent,
                    cmap="Oranges",
                )
                axes[1, i].set_title(f"L_Lgh {key}% percentile grid [Barrier {i}]")
                axes[1, i].grid(True)
                axes[1, i].axis("equal")
                fig.colorbar(im2, ax=axes[1, i])

                # annotate each cell with the max value
                for xi in range(cell_grid[1]):
                    for yi in range(cell_grid[0]):
                        x_mid = 0.5 * (x_domain[xi] + x_domain[xi + 1])
                        y_mid = 0.5 * (y_domain[yi] + y_domain[yi + 1])
                        val = L_Lgh_grid[key][yi, xi, i]
                        axes[1, i].text(
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
                f"{env_dir}/lipschitz_constants_grid/grid_{key}_{max_values_state[2]}.png"
            )
            logger.success(
                f"Grid for {key}% percentile saved: {env_dir}/lipschitz_constants_grid/grid_{key}_{max_values_state[2]}.png"
            )
        return L_Lfh_grid, L_Lgh_grid

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

    def info(self):
        # function to plot all the information
        [sensor.info(i) for i, sensor in enumerate(self.sensors)]

    # costmap part
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
