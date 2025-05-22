# this file combines both costmaps
from a_star import AStarPlanner
from cbf_costmap import CBFCostmap
from loguru import logger
import numpy as np


class CBFInfusedAStar(AStarPlanner):
    def __init__(
        self,
        costmap_size,
        grid_size,
        obstacles,
        cbf_costmap: CBFCostmap,
        diagonal_movement=True,
    ):
        super().__init__(costmap_size, grid_size, obstacles, diagonal_movement)
        self.cbf_costmap = cbf_costmap
        self.combine_costmaps(True)  # update the costmap

    def combine_costmaps(self, update=True):
        # Ensure the costmaps have the same shape
        if self.costmap.shape != self.cbf_costmap.costmap.shape:
            logger.error(
                f"Costmap shapes do not match! Shapes are {self.costmap.shape} and {self.cbf_costmap.costmap.shape}"
            )
            return self.costmap  # optionally return unmodified map

        # get unsafe regions
        unsafe_mask = self.cbf_costmap.costmap < 0
        modified_costmap = self.costmap.copy()
        modified_costmap[unsafe_mask] = np.inf

        if update:
            self.costmap = modified_costmap

        return modified_costmap
