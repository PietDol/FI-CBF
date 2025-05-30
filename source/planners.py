import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger
from cbf_costmap import CBFCostmap


class Node:
    def __init__(self, x, y, grid_size, origin_offset):
        self.x = x  # grid index (row)
        self.y = y  # grid index (column)
        self.grid_size = grid_size
        self.origin_offset = origin_offset
        self.pos = self.grid_to_world((x, y))  # Real-world position in meters
        self.g = float("inf")
        self.h = float("inf")
        self.f = float("inf")
        self.parent = None

    def grid_to_world(self, idx):
        ij = np.array(idx[::-1])
        pos = (
            (ij * self.grid_size)
            + (0.5 * self.grid_size)
            - (np.array(self.origin_offset) * self.grid_size)
        )
        return tuple(pos)

    def __lt__(self, other):  # Needed for heapq
        return self.f < other.f

    def coords(self):
        return (self.x, self.y)


class AStarPlanner:
    def __init__(
        self,
        costmap_size,
        grid_size=1,
        obstacles=[],
        noise_costmap: np.ndarray = None,
        noise_cost_gain: float = 1.0,
    ):
        # A* planner with euclidean heuristic
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.origin_offset = np.array(costmap_size) / (2 * self.grid_size)
        self.costmap_size = costmap_size
        self.noise_cost_gain = noise_cost_gain  # fraction of the g cost that is for the moving cost the rest (1-alpha) is for the noise
        
        # some check on alpha
        # if there is no costmap, all the cost is made by the distance
        if noise_costmap is None:
            self.noise_cost_gain = 0.0
        
        # set the movements
        self.directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        # attributes for the paths and distance map
        self.noise_costmap = noise_costmap
        self.costmap = None
        self.path_grid = None
        self.path_world = None

    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.costmap[x, y] < np.inf

    def heuristic_cost(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def world_to_grid(self, pos):
        """Convert world coordinate (x, y) in meters to grid index (i, j)."""
        grid = np.floor(
            (np.array(pos) + np.array(self.origin_offset) * self.grid_size)
            / self.grid_size
        ).astype(int)
        return tuple(grid[::-1])  # (i, j) as (row, col)

    def grid_to_world(self, idx):
        """Convert grid index (i, j) to world coordinate (x, y) in meters."""
        ij = np.array(idx[::-1])
        pos = (
            (ij * self.grid_size)
            + (0.5 * self.grid_size)
            - (np.array(self.origin_offset) * self.grid_size)
        )
        return tuple(pos)

    def plan(self, start_coords, goal_coords):
        # create the costmap
        self.create_costmap(start_coords)

        # normalize costmap
        noise_costmap = (self.noise_costmap - np.nanmin(self.noise_costmap)) / (np.nanmax(self.noise_costmap) - np.nanmin(self.noise_costmap) + 1e-6)

        # convert to grid
        start_coords = self.world_to_grid(start_coords)
        goal_coords = self.world_to_grid(goal_coords)

        # initialize the nodes
        start_node = Node(*start_coords, self.grid_size, self.origin_offset)
        goal_node = Node(*goal_coords, self.grid_size, self.origin_offset)

        open_set = []
        heapq.heappush(open_set, (0, start_node))

        start_node.g = 0
        start_node.h = self.heuristic_cost(start_node.coords(), goal_node.coords())
        start_node.f = start_node.g + start_node.h

        visited = {(start_node.x, start_node.y): start_node}

        while open_set:
            _, current = heapq.heappop(open_set)

            if (current.x, current.y) == (goal_node.x, goal_node.y):
                return {
                    "path_grid": self.reconstruct_path_grid(current),
                    "path_world": self.reconstruct_path_world(current),
                }

            for dx, dy in self.directions:
                nx, ny = current.x + dx, current.y + dy
                if not self.is_valid(nx, ny):
                    continue
                    
                # calculate the costs
                move_cost = self.grid_size * np.linalg.norm([dx, dy])
                if self.noise_costmap is not None:
                    noise_cost = noise_costmap[nx, ny]
                else:
                    noise_cost = 0.0
                tentative_g = current.g + move_cost + self.noise_cost_gain * noise_cost

                if (nx, ny) not in visited:
                    neighbor = Node(nx, ny, self.grid_size, self.origin_offset)
                    visited[(nx, ny)] = neighbor
                else:
                    neighbor = visited[(nx, ny)]

                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic_cost(
                        neighbor.coords(), goal_node.coords()
                    )
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current
                    heapq.heappush(open_set, (neighbor.f, neighbor))
        return None  # No path found

    def reconstruct_path_grid(self, current):
        path_grid = []
        while current:
            path_grid.append(current.coords())
            current = current.parent
        path_grid.reverse()
        self.path_grid = np.array(path_grid)

        return path_grid

    def reconstruct_path_world(self, current):
        path_world = []
        while current:
            path_world.append(self.grid_to_world(current.coords()))
            current = current.parent
        path_world.reverse()
        self.path_world = np.array(path_world)

        return path_world

    def create_costmap(self, start):
        """Compute a distance-from-start costmap for visualization."""
        # generate the costmap
        if self.costmap is None:
            costmap = (
                np.ones(
                    (
                        int(self.costmap_size[0] / self.grid_size),
                        int(self.costmap_size[1] / self.grid_size),
                    )
                )
                * self.grid_size
            )
            self.rows, self.cols = costmap.shape
        else:
            costmap = self.costmap.copy()

        # add the obstacles to it
        for obstacle in self.obstacles:
            costmap = obstacle.add_obstacle_to_costmap(
                costmap, self.origin_offset, self.grid_size
            )

        # add the distances
        for x in range(self.rows):
            for y in range(self.cols):
                if self.is_valid(x, y):
                    world_cor = self.grid_to_world((x, y))
                    distance = np.linalg.norm(np.array(world_cor) - np.array(start))
                    costmap[x, y] = distance
        self.costmap = costmap
        return costmap

    def plot_costmap(self, path=None, start=None, goal=None, use_distance_map=True):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Compute distance map
        if use_distance_map and start:
            display_map = self.create_costmap(start)
        else:
            display_map = self.costmap.copy()

        # Mask obstacles
        obstacle_mask = np.isinf(self.costmap)
        display_map = display_map.copy()
        if np.any(obstacle_mask):
            max_val = np.max(display_map[~obstacle_mask])
            display_map[obstacle_mask] = max_val + 1
            vmax = max_val + 1
            vmin = np.min(display_map[~obstacle_mask])
        else:
            vmax = np.max(display_map)
            vmin = np.min(display_map)

        # Show distance/cost map
        extent = [
            *self.grid_to_world((0, 0)),  # lower left (x_min, y_min)
            *self.grid_to_world((self.rows, self.cols)),  # upper right (x_max, y_max)
        ]
        extent = [
            extent[0],
            extent[2],
            extent[1],
            extent[3],
        ]  # reorder for imshow: [x_min, x_max, y_min, y_max]

        cmap = plt.cm.viridis
        img = ax.imshow(
            display_map, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, extent=extent
        )

        # Draw obstacles as black rectangles
        for obstacle in self.obstacles:
            drawing = obstacle.pyplot_drawing(0.7)
            ax.add_patch(drawing)

        # Convert path to world and plot
        if path:
            path_world = [self.grid_to_world(p) for p in path]
            x_coords, y_coords = zip(*path_world)
            ax.plot(x_coords, y_coords, color="cyan", linewidth=2, label="Path")

        # Start and goal markers (converted to world)
        if start:
            sx, sy = start
            ax.plot(sx, sy, "go", markersize=8, label="Start")
        if goal:
            gx, gy = goal
            ax.plot(gx, gy, "ro", markersize=8, label="Goal")

        # Axis labels
        ax.set_title("Costmap and A* Path")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.grid(True)
        ax.axis("equal")

        # Legend
        obstacle_patch = mpatches.Patch(color="black", label="Obstacle")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(obstacle_patch)
        ax.legend(handles=handles, loc="upper left")

        # Colorbar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Distance from start", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()


class CBFInfusedAStar(AStarPlanner):
    def __init__(
        self,
        costmap_size,
        grid_size,
        obstacles,
        cbf_costmap: CBFCostmap,
        noise_costmap: np.ndarray = None,
        noise_cost_gain: float = 1.0,
    ):
        super().__init__(costmap_size, grid_size, obstacles, noise_costmap, noise_cost_gain)
        self.cbf_costmap = cbf_costmap
        self.combine_costmaps(True)  # update the costmap

    def combine_costmaps(self, update=True):
        if self.costmap is None:
            modified_costmap = (
                np.ones(
                    (
                        int(self.costmap_size[0] / self.grid_size),
                        int(self.costmap_size[1] / self.grid_size),
                    )
                )
                * self.grid_size
            )
            self.rows, self.cols = modified_costmap.shape
        # Ensure the costmaps have the same shape
        elif self.costmap.shape != self.cbf_costmap.costmap.shape:
            logger.error(
                f"Costmap shapes do not match! Shapes are {self.costmap.shape} and {self.cbf_costmap.costmap.shape}"
            )
            return self.costmap  # optionally return unmodified map
        else:
            modified_costmap = self.costmap.copy()

        # get unsafe regions
        unsafe_mask = self.cbf_costmap.costmap < 0
        modified_costmap[unsafe_mask] = np.inf

        if update:
            self.costmap = modified_costmap

        return modified_costmap
