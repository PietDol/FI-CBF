import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from obstacles import RectangleObstacle
from robot_base import RobotBase
from env_config import EnvConfig

class AStarPlanner:
    def __init__(self, costmap, diagonal_movement=True, heuristic='euclidean'):
        self.costmap = costmap
        self.rows, self.cols = costmap.shape
        self.diagonal_movement = diagonal_movement
        self.heuristic = heuristic

        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1)
        ]
        if diagonal_movement:
            self.directions += [
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]

    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.costmap[x, y] < np.inf

    def heuristic_cost(self, a, b):
        if self.heuristic == 'manhattan':
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        else:  # Euclidean by default
            return np.linalg.norm(np.array(a) - np.array(b))

    def plan(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid(*neighbor):
                    continue

                move_cost = np.linalg.norm([dx, dy])
                tentative_g = g_score[current] + self.costmap[neighbor] * move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic_cost(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

        return None  # No path found

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
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
        """Visualize the costmap or distance-from-start map with optional path and color legend."""
        fig, ax = plt.subplots(figsize=(6, 6))

        # Compute distance map if requested
        if use_distance_map and start:
            display_map = self.compute_distance_map(start)
        else:
            display_map = self.costmap.copy()

        # Mask obstacles
        obstacle_mask = np.isinf(self.costmap)
        display_map = display_map.copy()
        if np.any(obstacle_mask):
            max_val = np.max(display_map[~obstacle_mask])
            display_map[obstacle_mask] = max_val + 10  # make obstacles stand out
            vmax = max_val + 10
            vmin = np.min(display_map[~obstacle_mask])
        else:
            vmax = np.max(display_map)
            vmin = np.min(display_map)

        # Show distance/cost map with colormap
        cmap = plt.cm.viridis
        img = ax.imshow(display_map, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

        # Overlay obstacles as black squares
        obstacle_overlay = np.full_like(self.costmap, np.nan, dtype=float)
        obstacle_overlay[obstacle_mask] = 1.0  # Only obstacle cells will be shown

        # Overlay using a black colormap on obstacle cells only
        ax.imshow(obstacle_overlay, cmap='gray', origin='lower', vmin=0, vmax=1, alpha=1.0)

        # Overlay path
        if path:
            y_coords, x_coords = zip(*path)
            ax.plot(x_coords, y_coords, color='cyan', linewidth=2, label='Path')

        # Start and goal markers
        if start:
            ax.plot(start[1], start[0], 'go', markersize=8, label='Start')
        if goal:
            ax.plot(goal[1], goal[0], 'ro', markersize=8, label='Goal')

        # Legend with obstacle patch
        handles, labels = ax.get_legend_handles_labels()

        obstacle_patch = mpatches.Patch(color='black', label='Obstacle')
        handles.append(obstacle_patch)
        ax.legend(handles=handles, loc='upper left')

        # Add colorbar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Distance from start", rotation=270, labelpad=15)

        ax.set_title("Costmap and A* Path")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    size = np.ones((10, 10))
    grid_size = 1
    costmap = np.ones((int(size.shape[0] / grid_size), int(size.shape[1] / grid_size))) * grid_size
    env_config = EnvConfig(
        pixels_per_meter=50 * np.array([1, -1]),
        screen_width=800,
        screen_height=800
    )

    robot = RobotBase(
        width=1,
        height=1,
        env_config=env_config,
        pos_goal=np.array([9, 9])
    )
    rect_obstacle = RectangleObstacle(
        width=2,
        height=2,
        pos_center=np.array([2, 2]),
        env_config=env_config,
        robot=robot
    )
    # costmap[4, 2:8] = np.inf  # obstacle
    costmap = rect_obstacle.add_obstacle_to_costmap(costmap, grid_size)

    start = (0, 0)
    goal = (9, 9)

    planner = AStarPlanner(costmap)
    path = planner.plan(start, goal)

    planner.plot_costmap(path=path, start=start, goal=goal)