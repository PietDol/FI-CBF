import numpy as np
from loguru import logger


class ConfidenceManager:
    def __init__(self, confidence_config: dict):
        self.vmax_levels = confidence_config["vmax"]
        self.k_levels = confidence_config["k"]
        self.sigma_thresholds = confidence_config["sigma_thresholds"]
        self.delta = confidence_config["deltas"]
        self.num_levels = len(self.vmax_levels)

    def transition_function(self, sigma, s, delta):
        """Smooth sigmoid transition function φ(σ; s, δ)."""
        return 1 / (1 + np.exp((sigma - s) / delta))

    def get_confidence_info(self, sigma: float):
        """
        Returns the current confidence level index, v_max, and k based on the current sigma.
        """
        for i, s in enumerate(self.sigma_thresholds):
            if sigma < s:
                # played around with wolfram alpha. /8 param gives that the change take place in the delta range
                delta = self.delta[i] / 8
                transition_value = self.transition_function(sigma, s, delta)
                vmax = (1 - transition_value) * self.vmax_levels[
                    i + 1
                ] + transition_value * self.vmax_levels[i]
                k = (1 - transition_value) * self.k_levels[
                    i + 1
                ] + transition_value * self.k_levels[i]
                # logger.debug(f"σ, T, v, k: {sigma}, {transition_value}, {vmax}, {k}")
                return i, vmax, k
        # beyond last threshold → return last level
        return self.num_levels - 1, self.vmax_levels[-1], self.k_levels[-1]
