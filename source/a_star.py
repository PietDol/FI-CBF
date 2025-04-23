import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from obstacles import RectangleObstacle, CircleObstacle
from robot_base import RobotBase
from env_config import EnvConfig
from matplotlib.collections import PatchCollection

class Node:
    def __init__(self, x, y, grid_size, origin_offset):
        self.x = x  # grid index (row)
        self.y = y  # grid index (column)
        self.grid_size = grid_size
        self.origin_offset = origin_offset
        self.pos = self.grid_to_world((x, y))  # Real-world position in meters
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        self.parent = None
    
    def grid_to_world(self, idx):
        ij = np.array(idx[::-1])
        pos = (ij * self.grid_size) - (np.array(self.origin_offset) * self.grid_size)
        return tuple(pos)

    def __lt__(self, other):  # Needed for heapq
        return self.f < other.f

    def coords(self):
        return (self.x, self.y)

class AStarPlanner:
    def __init__(self, costmap_size, grid_size=1, obstacles=[], diagonal_movement=True, heuristic='euclidean'):
        self.obstacles = obstacles
        self.grid_size = grid_size  
        self.origin_offset = np.array(costmap_size) / (2 * self.grid_size)

        # create costmap costmap size is given in m
        self.costmap = self.create_costmap(costmap_size)
        self.rows, self.cols = self.costmap.shape

        # set the movements
        self.diagonal_movement = diagonal_movement
        self.heuristic = heuristic

        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1)
        ]
        if diagonal_movement:
            self.directions += [
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
    
    def create_costmap(self, costmap_size):
        # generate the costmap
        costmap = np.ones((int(costmap_size[0] / self.grid_size), int(costmap_size[1] / self.grid_size))) * self.grid_size

        # add the obstacles to it
        for obstacle in self.obstacles:
            costmap = obstacle.add_obstacle_to_costmap(costmap, self.origin_offset, self.grid_size)
        return costmap

    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.costmap[x, y] < np.inf

    def heuristic_cost(self, a, b):
        if self.heuristic == 'manhattan':
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:  # Euclidean by default
            return np.linalg.norm(np.array(a) - np.array(b))
    
    def world_to_grid(self, pos):
        """Convert world coordinate (x, y) in meters to grid index (i, j)."""
        grid = np.floor((np.array(pos) + np.array(self.origin_offset) * self.grid_size) / self.grid_size).astype(int)
        return tuple(grid[::-1])  # (i, j) as (row, col)

    def grid_to_world(self, idx):
        """Convert grid index (i, j) to world coordinate (x, y) in meters."""
        ij = np.array(idx[::-1])
        pos = (ij * self.grid_size) - (np.array(self.origin_offset) * self.grid_size)
        return tuple(pos)

    def plan(self, start_coords, goal_coords):
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
                    "path_world":self.reconstruct_path_world(current)
                }

            for dx, dy in self.directions:
                nx, ny = current.x + dx, current.y + dy
                if not self.is_valid(nx, ny):
                    continue

                move_cost = self.grid_size * np.linalg.norm([dx, dy])
                tentative_g = current.g + move_cost

                if (nx, ny) not in visited:
                    neighbor = Node(nx, ny, self.grid_size, self.origin_offset)
                    visited[(nx, ny)] = neighbor
                else:
                    neighbor = visited[(nx, ny)]

                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic_cost(neighbor.coords(), goal_node.coords())
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

        return path_grid

    def reconstruct_path_world(self, current):
        path_world = []
        while current:
            path_world.append(self.grid_to_world(current.coords()))
            current = current.parent
        path_world.reverse()

        return path_world
    
    def compute_distance_map(self, start, metric='euclidean'):
        """Compute a distance-from-start costmap for visualization."""
        distance_map = np.full_like(self.costmap, np.inf, dtype=float)
        for x in range(self.rows):
            for y in range(self.cols):
                if self.is_valid(x, y):
                    if metric == 'manhattan':
                        distance = abs(x - start[0]) + abs(y - start[1])
                    else:  # Euclidean by default
                        distance = np.linalg.norm(np.array([x, y]) - np.array(start))
                    distance_map[x, y] = distance
        return distance_map

    def plot_costmap(self, path=None, start=None, goal=None, use_distance_map=True):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Compute distance map
        if use_distance_map and start:
            display_map = self.compute_distance_map(start)
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
            *self.grid_to_world((0, 0)),                     # lower left (x_min, y_min)
            *self.grid_to_world((self.rows, self.cols))      # upper right (x_max, y_max)
        ]
        extent = [extent[0], extent[2], extent[1], extent[3]]  # reorder for imshow: [x_min, x_max, y_min, y_max]

        cmap = plt.cm.viridis
        img = ax.imshow(display_map, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, extent=extent)

        # Draw obstacles as black rectangles
        for obstacle in self.obstacles:
            drawing = obstacle.pyplot_drawing(0.7)
            ax.add_patch(drawing)

        # Convert path to world and plot
        if path:
            path_world = [self.grid_to_world(p) for p in path]
            x_coords, y_coords = zip(*path_world)
            ax.plot(x_coords, y_coords, color='cyan', linewidth=2, label='Path')

        # Start and goal markers (converted to world)
        if start:
            sx, sy = start
            ax.plot(sx, sy, 'go', markersize=8, label='Start')
        if goal:
            gx, gy = goal
            ax.plot(gx, gy, 'ro', markersize=8, label='Goal')

        # Axis labels
        ax.set_title("Costmap and A* Path")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.grid(True)
        ax.axis('equal')

        # Legend
        obstacle_patch = mpatches.Patch(color='black', label='Obstacle')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(obstacle_patch)
        ax.legend(handles=handles, loc='upper left')

        # Colorbar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Distance from start", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    start = (-7, 0.01)
    goal = (4, 0)
    # goal = (-7, 0.01)
    # start = (4, 0)
    env_config = EnvConfig(
        pixels_per_meter=50 * np.array([1, -1]),
        screen_width=800,
        screen_height=800
    )
    robot = RobotBase(
        width=1,
        height=1,
        env_config=env_config,
        pos_center_start=np.array(start),
        pos_goal=np.array(goal)
    )
    rect_obstacle = RectangleObstacle(
        width=2,
        height=2,
        pos_center=np.array([2, 0]),
        env_config=env_config,
        robot=robot
    )
    circ_obstacle = CircleObstacle(
        radius=3,
        pos_center=np.array([-2, 3]),
        env_config=env_config,
        robot=robot
    )
    # obstacles = [rect_obstacle, circ_obstacle]
    obstacles = [
        RectangleObstacle(1, 6, np.array([0, 0.0]), env_config, robot),
        # CircleObstacle(2, np.array([0.0, 0.0]), env_config, robot),
        # CircleObstacle(1, np.array([4.0, 4.0]), env_config, robot)
    ]

    planner = AStarPlanner(
        costmap_size=(20, 20),
        grid_size=0.5,
        obstacles=obstacles
    )

    path = planner.plan(start, goal)
    planner.plot_costmap(path=path["path_grid"], start=start, goal=goal)