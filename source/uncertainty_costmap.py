# this file is used to create the uncertainty costmap
from cbfpy import CBF
from loguru import logger
import jax
import jax.numpy as jnp
import numpy as np

class UncertaintyCostmap:
    def __init__(self, 
                cbf: CBF,
                min_values_state: np.ndarray,
                max_values_state: np.ndarray,
                num_samples: int = 100):
        self.cbf = cbf
        self.min_values_state = min_values_state
        self.max_values_state = max_values_state
        self.L_Lfh, self.L_Lgh = self._estimate_cbf_lipschitz_constants(num_samples)

    def calculate_safety_margin(self, epsilon: float, u_nominal: np.ndarray, mode: str = "robust"):
        """
        Converts the uncertainty to the safety margin that needs to be
        used by the CBFs to account for estimation uncertainty.

        Parameters:
            epsilon (float): upper bound on the state estimation error (e.g., std of Gaussian noise)
            mode (str): 'robust' or 'probabilistic'

        Returns:
            float: safety margin to be added to the robot's radius
        """
        # Assume alpha(h) = h, so L_alpha_h = 1
        L_alpha_h = 1.0

        if mode == "robust":
            a = (self.L_Lfh + L_alpha_h) * epsilon
            b = self.L_Lgh * epsilon
            safety_margin = a + b * jnp.linalg.norm(u_nominal)**2
            # logger.info(f"Nominal control: {u_nominal}")
            # logger.info(f"Calculated safety margin: {safety_margin}")
            return safety_margin

        elif mode == "probabilistic":
            raise NotImplementedError("Probabilistic margin not implemented yet.")

        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'robust', 'probabilistic'.")

    def _estimate_cbf_lipschitz_constants(self, num_samples: int):
        key = jax.random.PRNGKey(0)
        Z = jax.random.uniform(
            key, (num_samples, self.cbf.n),
            minval=self.min_values_state,
            maxval=self.max_values_state
        )

        # K is the number of barrier functions
        # m is the size of the controller
        init_safety_margin = np.zeros(self.cbf.num_cbf)
        Lfhs = jax.vmap(lambda z: self.cbf.h_and_Lfh(z, init_safety_margin)[1])(Z)    # (N, K)
        Lghs = jax.vmap(lambda z: self.cbf.Lgh(z, init_safety_margin))(Z)             # (N, K, m)

        def estimate_lipschitz_scalar(values, inputs):
            """Estimate Lipschitz constant for each scalar output"""
            N, K = values.shape
            lipschitz_per_output = []

            for k in range(K):
                y = values[:, k]                                 # (N,)
                diffs_x = inputs[:, None, :] - inputs[None, :, :]
                diffs_y = y[:, None] - y[None, :]

                dx = jnp.linalg.norm(diffs_x, axis=-1)
                dx = jnp.where(dx < 1e-6, 1e-6, dx)

                dy = jnp.abs(diffs_y)
                lipschitz_matrix = dy / dx
                lipschitz_matrix = jnp.nan_to_num(lipschitz_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                lipschitz_per_output.append(jnp.max(jnp.triu(lipschitz_matrix, k=1)))

            return jnp.array(lipschitz_per_output)

        def estimate_lipschitz_vector(values, inputs):
            """Estimate Lipschitz constant per vector-valued output (max over control dim)"""
            N, K, m = values.shape
            lipschitz_per_barrier = []

            for k in range(K):
                y = values[:, k, :]                               # (N, m)
                diffs_x = inputs[:, None, :] - inputs[None, :, :] # (N, N, D)
                diffs_y = y[:, None, :] - y[None, :, :]           # (N, N, m)

                dx = jnp.linalg.norm(diffs_x, axis=-1)
                dx = jnp.where(dx < 1e-6, 1e-6, dx)

                dy = jnp.linalg.norm(diffs_y, axis=-1)            # vector norm over control dim
                lipschitz_matrix = dy / dx
                lipschitz_matrix = jnp.nan_to_num(lipschitz_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                lipschitz_per_barrier.append(jnp.max(jnp.triu(lipschitz_matrix, k=1)))

            return jnp.array(lipschitz_per_barrier)

        L_Lfh = estimate_lipschitz_scalar(Lfhs, Z)    # (K,)
        L_Lgh = estimate_lipschitz_vector(Lghs, Z)    # (K,)

        logger.info(f"L_Lfh per barrier: {L_Lfh}")
        logger.info(f"L_Lgh per barrier: {L_Lgh}")

        return np.array(L_Lfh), np.array(L_Lgh)