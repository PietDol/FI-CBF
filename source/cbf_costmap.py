from cbfpy import CBF
from robot_base_config import RobotBaseCBFConfig, RobotBaseCLFCBFConfig
import numpy as np
import jax.numpy as jnp
from loguru import logger

class CBFCostmap:
    def __init__(self, costmap_size, grid_size, cbf: RobotBaseCBFConfig | RobotBaseCLFCBFConfig, cbf_reduction='min'):
        self.grid_size = grid_size  
        self.costmap_size = costmap_size
        self.origin_offset = np.array(costmap_size) / (2 * self.grid_size)
        self.cbf = cbf
        self.cbf_reduction = cbf_reduction
        self.costmap = self.create_costmap()
        logger.success("CBF costmap created")
    
    def grid_to_world(self, idx):
        """Convert grid index (row, col) to world coordinate (x, y) in meters. It returns the center of the grid."""
        ij = np.array(idx[::-1])
        pos = (ij * self.grid_size) + (0.5 * self.grid_size) - (np.array(self.origin_offset) * self.grid_size)
        return pos
    
    def create_costmap(self):
        # Creates the costmap representation of the CBF.
        rows, cols = int(self.costmap_size[0] / self.grid_size), int(self.costmap_size[1] / self.grid_size)

        # Generate grid indices
        row_idx, col_idx = np.indices((rows, cols))

        # Stack into (N, 2)
        ij = np.stack((col_idx, row_idx), axis=-1).reshape(-1, 2)

        # Convert grid indices to world coordinates
        pos = (ij * self.grid_size) + (0.5 * self.grid_size) - (np.array(self.origin_offset) * self.grid_size)

        # Build full states (N, 4)
        states = np.zeros((pos.shape[0], 4))
        states[:, :2] = pos

        # Get h values
        h_values = self.cbf.h_1(states, batched=True)  # (N, num_obstacles)

        # Apply alpha function to each
        h_alpha = self.cbf.alpha_batch(h_values)  # (N, num_obstacles)

        # Choose reduction method
        # default is min, because otherwise if you do the sum of mean it could be positive
        # after this step the size is (N,)
        if self.cbf_reduction == "min":
            h_final = jnp.min(h_alpha, axis=1)  # (N,)
        elif self.cbf_reduction == "mean":
            h_final = jnp.mean(h_alpha, axis=1) # (N,)
        elif self.cbf_reduction == "sum":    
            h_final = jnp.sum(h_alpha, axis=1)  # (N,)
        else:
            logger.error(f"Unknown reduction method '{self.cbf_reduction}'. Choose from 'min', 'mean', or 'sum'.")
            logger.info(f"Min cbf reduction is used!")
            h_final = jnp.min(h_alpha, axis=1)  # (N,)

        # Reshape into costmap
        cbf_costmap = h_final.reshape(rows, cols)

        return cbf_costmap
    
