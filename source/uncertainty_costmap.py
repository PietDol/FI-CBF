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
        self.L_Lfh, self.L_Lgh = self.estimate_cbf_lipschitz_constants(num_samples)

    def calculate_safety_margin(self, epsilon: float, u_nominal: np.ndarray, mode: str = "robust") -> float:
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
            return float(safety_margin)

        elif mode == "probabilistic":
            raise NotImplementedError("Probabilistic margin not implemented yet.")

        else:
            raise ValueError(f"Unknown mode '{mode}'. Supported modes: 'robust', 'probabilistic'.")


    def estimate_cbf_lipschitz_constants(self, num_samples: int):
        """
        Estimate Lipschitz constants L_{L_f h} and L_{L_g h} over a sampled state space.
        It basically calculates the max value over the sampled states.

        Returns:
            Tuple[float, float]: Estimated Lipschitz constants (L_Lfh, L_Lgh)
        """
        key = jax.random.PRNGKey(0)
        Z = jax.random.uniform(key, (num_samples, self.cbf.n), minval=self.min_values_state, maxval=self.max_values_state)

        Lfhs = jax.vmap(lambda z: self.cbf.h_and_Lfh(z)[1])(Z)
        Lghs = jax.vmap(lambda z: self.cbf.Lgh(z))(Z)
        if not jnp.isnan(Lfhs).any():
            logger.warning("NaNs detected in Lfhs")
        if not jnp.isnan(Lghs).any():
            logger.warning("NaNs detected in Lghs")

        def estimate_lipschitz(values, inputs):
            # values: (N, D), inputs: (N, D)
            diffs_x = inputs[:, None, :] - inputs[None, :, :]       # (N, N, D)
            diffs_y = values[:, None, :] - values[None, :, :]       # (N, N, D)

            dx = jnp.linalg.norm(diffs_x, axis=-1)                  # (N, N)
            dx = jnp.where(dx < 1e-6, 1e-6, dx)                     # Clamp to avoid near-zero

            dy = jnp.linalg.norm(diffs_y, axis=-1)                  # (N, N)
            lipschitz_matrix = dy / dx

            # Optional: Clean up numerical edge cases
            lipschitz_matrix = jnp.nan_to_num(lipschitz_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            return jnp.max(jnp.triu(lipschitz_matrix, k=1))

        L_Lfh = estimate_lipschitz(Lfhs, Z)
        L_Lgh = estimate_lipschitz(Lghs.reshape(num_samples, -1), Z)

        logger.info(f"Lipschitz constant L_Lfh: {L_Lfh}")
        logger.info(f"Lipschitz constant L_Lgh: {L_Lgh}")
        return L_Lfh, L_Lgh
