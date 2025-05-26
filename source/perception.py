# this file is created to simulate a sensor signal
import numpy as np
from cbfpy import CBF
from loguru import logger
import jax
import jax.numpy as jnp
import random


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
        min_values_state: np.ndarray,
        max_values_state: np.ndarray,
        max_sensor_noise: float,
        num_sensors: int = None,
        num_samples_per_dim: int = 4,
        sensor_location: np.ndarray = None,
        sensors: list = None,
    ):
        self.costmap_size = costmap_size
        self.cbf = cbf
        self.num_sensors = num_sensors
        self.min_values_state = min_values_state
        self.max_values_state = max_values_state
        self.max_sensor_noise = max_sensor_noise

        # parameters for the sigmoid function to convert magnitude to noise
        self.alpha = 0.8  # value for hybrid approach between max and mean

        # estimate the lipschitz constants
        self.L_Lfh, self.L_Lgh = self._estimate_cbf_lipschitz_constants(
            num_samples_per_dim
        )

        # create the sensors if not given
        if sensors is None:
            if sensor_location is not None:
                if sensor_location.ndim < 2:
                    sensor_location = np.expand_dims(sensor_location, 0)

                if num_sensors != sensor_location.shape[0]:
                    logger.warning(
                        f"Number of sensor != number of sensor locations {num_sensors} != {sensor_location.shape[0]}"
                    )
                    logger.info(
                        "Instead of given sensor location, generate random locations"
                    )
                    self.sensors = self.create_random_sensor_locations()
                else:
                    self.sensors = [
                        Sensor(sensor_location[i]) for i in range(self.num_sensors)
                    ]
            else:
                self.sensors = self.create_random_sensor_locations()
        else:
            self.sensors = sensors

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
            "Sensor added and perception magnitude and noise costmaps updated"
        )

    def create_random_sensor_locations(self):
        sensors = []
        for i in range(self.num_sensors):
            sx = np.round(
                random.uniform(-self.costmap_size[0] / 2, self.costmap_size[0] / 2), 2
            )
            sy = np.round(
                random.uniform(-self.costmap_size[1] / 2, self.costmap_size[1] / 2), 2
            )
            sensors.append(Sensor(np.array([sx, sy])))
        return sensors

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
        return self.alpha * np.max(magnitudes) + (1 - self.alpha) * np.mean(magnitudes)

    def get_perception_noise(self, x_true: np.ndarray):
        # returns the standard deviation for the noise
        mag = self.get_perception_magnitude(x_true)
        return (1 - mag) * self.max_sensor_noise

    def get_perception_magnitude_batched(self, x_true: jnp.ndarray) -> jnp.ndarray:
        # batched version: compute perception magnitude for each position in (N, 2).
        def single_mag(x):
            mags = jnp.array(
                [sensor.get_sensor_magnitude(x) for sensor in self.sensors]
            )
            return self.alpha * jnp.max(mags) + (1 - self.alpha) * jnp.mean(mags)

        return jax.vmap(single_mag)(x_true)  # (N,)

    def get_perception_noise_batched(self, x_true: jnp.ndarray) -> jnp.ndarray:
        # batched version: compute noise std for each position in (N, 2).
        mags = self.get_perception_magnitude_batched(x_true)  # (N,)
        return (1 - mags) * self.max_sensor_noise

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

    def get_epsilon(self, noise: float, mode: str):
        # calculate the value of epsilon based on the value of the noise and the mode
        if mode == "robust":
            epsilon = 4 * noise
        elif mode == "probabilistic":
            epsilon = 1.96 * noise
        else:
            logger.error(
                f"Unknown mode '{mode}' Supported modes: 'robust', 'probabilistic'"
            )
            logger.info("Use robust mode.")
            epsilon = 4 * noise
        return epsilon

    def calculate_safety_margin(
        self, noise: float, u_nominal: np.ndarray, mode: str = "robust"
    ):
        # Converts the uncertainty to the safety margin that needs to be used by the CBFs to
        # account for estimation uncertainty. Epsilon is upper bound on estimation error
        # Assume alpha(h) = h, so L_alpha_h = 1
        L_alpha_h = 1.0

        # calculate epsilon in the paper they use eps=0.4
        epsilon = self.get_epsilon(noise, mode)

        # calculate the safety margin
        a = (self.L_Lfh + L_alpha_h) * epsilon
        b = self.L_Lgh * epsilon
        safety_margin = a + b * jnp.linalg.norm(u_nominal) ** 2
        return safety_margin

    def _estimate_cbf_lipschitz_constants(self, num_points_per_dim: int):
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

                lipschitz_per_output.append(jnp.max(jnp.triu(lipschitz_matrix, k=1)))

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

                lipschitz_per_barrier.append(jnp.max(jnp.triu(lipschitz_matrix, k=1)))

            return jnp.array(lipschitz_per_barrier)

        L_Lfh = estimate_lipschitz_scalar(Lfhs, Z)  # (K,)
        L_Lgh = estimate_lipschitz_vector(Lghs, Z)  # (K,)

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
