# this file is used to create the uncertainty costmap
from loguru import logger
from perception import Perception
import numpy as np
import jax.numpy as jnp

class UncertaintyCostmap:
    def __init__(self, 
                costmap_size, 
                grid_size: np.ndarray,
                perception: Perception
                ):
        self.grid_size = grid_size  
        self.costmap_size = costmap_size
        self.perception = perception    # perception module
        self.origin_offset = np.array(costmap_size) / (2 * self.grid_size)
        self.costmap = self.create_costmap()
        logger.success("Uncertainty costmap created")
    
    def grid_to_world(self, idx):
        # Convert grid index (row, col) to world coordinate (x, y) in meters. It returns the center of the grid.
        ij = np.array(idx[::-1])
        pos = (ij * self.grid_size) + (0.5 * self.grid_size) - (np.array(self.origin_offset) * self.grid_size)
        return pos
    
    def create_costmap(self):
        rows, cols = int(self.costmap_size[0] / self.grid_size), int(self.costmap_size[1] / self.grid_size)
        row_idx, col_idx = np.indices((rows, cols))
        ij = np.stack((col_idx, row_idx), axis=-1).reshape(-1, 2)   # convert to (N, 2)
        
        # convert grid to world pos (N, 2)
        pos = (ij * self.grid_size) + (0.5 * self.grid_size) - (np.array(self.origin_offset) * self.grid_size)

        # calculate the uncertainty/noise
        # uncertainty = self.perception.get_perception_noise_batched(jnp.array(pos))  # shape (N,)
        uncertainty = self.perception.get_perception_magnitude_batched(jnp.array(pos))  # shape (N,)
        return np.array(uncertainty).reshape(rows, cols)
        