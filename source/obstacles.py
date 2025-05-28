import numpy as np
from abc import ABC, abstractmethod
import jax.numpy as jnp
import matplotlib.patches as patches


class Obstacle(ABC):
    """Abstract class to create all different kinds of obstacls"""

    @abstractmethod
    def h(self):
        # method to get the value of the cbf
        pass

    @abstractmethod
    def pyplot_drawing(self):
        # method to return a plot drawing for the trajectory
        pass

    @abstractmethod
    def check_collision(self):
        # method to check if the robot is in collision with the object
        pass

    @abstractmethod
    def check_goal_position(self):
        # method to check if the goal position of the robot is feasible
        pass

    @abstractmethod
    def add_obstacle_to_costmap(self):
        # function to add the obstacle to the costmap for the planner
        pass


class RectangleObstacle(Obstacle):
    # object for obstacles
    def __init__(self, width, height, robot_radius, pos_center, id: int):
        self.width = width  # in m
        self.height = height  # in m
        self.robot_radius = robot_radius  # in m (space needed by the robot)
        self.pos_center = pos_center  # in m shape (2,)
        self.id = id

    def pyplot_drawing(self, opacity=1.0):
        cx, cy = self.pos_center
        bottom_left_x = cx - self.width / 2
        bottom_left_y = cy - self.height / 2

        square = patches.Rectangle(
            (bottom_left_x, bottom_left_y),
            self.width,
            self.height,
            edgecolor="red",
            facecolor="black",
            alpha=opacity,
            fill=True,
        )
        return square

    def add_obstacle_to_costmap(self, costmap, origin_offset, grid_size=1):
        # adds this rectangular obstacle to the given costmap.
        # calculate the x_min, x_max, y_min and y_max values
        x_min = self.pos_center[0] - (self.width / 2)
        x_max = self.pos_center[0] + (self.width / 2)
        y_min = self.pos_center[1] - (self.height / 2)
        y_max = self.pos_center[1] + (self.height / 2)

        # convert to grid (int does the floor conversion)
        # x-axis -> cols
        # y-axis -> rows
        col_min = min(
            max(int((x_min + origin_offset[0] * grid_size) / grid_size), 0),
            costmap.shape[1],
        )
        col_max = max(
            min(
                int((x_max + origin_offset[0] * grid_size) / grid_size),
                costmap.shape[1],
            ),
            0,
        )
        row_min = min(
            max(int((y_min + origin_offset[1] * grid_size) / grid_size), 0),
            costmap.shape[0],
        )
        row_max = max(
            min(
                int((y_max + origin_offset[1] * grid_size) / grid_size),
                costmap.shape[0],
            ),
            0,
        )

        # set the obstacles to inf costs
        costmap[row_min:row_max, col_min:col_max] = np.inf
        return costmap

    def h(self, z, safety_margin=0.0):
        # batched -> faster
        if z.ndim == 1:
            z = z[None, :]  # Reshape to (1, 4)

        # Z: shape (N, 4)
        px = z[:, 0]
        py = z[:, 1]

        closest_point = self.find_closest_point_to_obstacle(px, py)  # (N, 2)

        vectors = jnp.stack([px, py], axis=1) - closest_point  # (N, 2)
        norm = jnp.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6
        normal_vectors = vectors / norm

        h_values = (
            jnp.sum(normal_vectors * vectors, axis=1)
            - self.robot_radius
            - safety_margin
        )
        return h_values  # shape (N,)

    def check_collision(
        self,
        robot_position: np.ndarray,
        robot_width: float,
        robot_height: float,
        safety_margin=0.0,
    ):
        # function to check if the robot is in collision with this obstacle
        # get the centers of the robot and the obstacle
        cx_robot, cy_robot = robot_position
        cx_obstacle, cy_obstacle = self.pos_center

        # Check x-axis overlap
        x_overlap = (
            abs(cx_robot - cx_obstacle)
            <= (robot_width / 2) + (self.width / 2) + safety_margin
        )

        # Check y-axis overlap
        y_overlap = (
            abs(cy_robot - cy_obstacle)
            <= (robot_height / 2) + (self.height / 2) + safety_margin
        )

        return x_overlap and y_overlap

    def find_closest_point_to_obstacle(self, px, py):
        # batched
        # px, py: (N,)
        cx_obstacle, cy_obstacle = self.pos_center
        closest_x = jnp.clip(
            px, cx_obstacle - (self.width / 2), cx_obstacle + (self.width / 2)
        )
        closest_y = jnp.clip(
            py, cy_obstacle - (self.height / 2), cy_obstacle + (self.height / 2)
        )
        return jnp.stack([closest_x, closest_y], axis=1)  # (N, 2)

    def check_goal_position(
        self,
        goal_position: np.ndarray,
        robot_width: float,
        robot_height: float,
        extra_safety_margin=0.5,
    ):
        # check if the goal position is feasible
        cx_robot, cy_robot = goal_position
        cx_obstacle, cy_obstacle = self.pos_center

        # Check x-axis overlap
        x_overlap = (
            abs(cx_robot - cx_obstacle)
            <= (robot_width / 2) + (self.width / 2) + extra_safety_margin
        )

        # Check y-axis overlap
        y_overlap = (
            abs(cy_robot - cy_obstacle)
            <= (robot_height / 2) + (self.height / 2) + extra_safety_margin
        )

        return not (x_overlap and y_overlap)


class CircleObstacle(Obstacle):
    # object for circular obstacles
    def __init__(self, radius, robot_radius, pos_center, id: int):
        self.radius = radius  # in m
        self.robot_radius = robot_radius  # in m (space needed by the robot)
        self.pos_center = pos_center  # in m shape (2,)
        self.id = id

    def add_obstacle_to_costmap(self, costmap, origin_offset, grid_size=1):
        """
        Adds a circular obstacle to the costmap.
        The circle is conservatively approximated as a filled square + circular mask.
        """
        # Convert center to grid coordinates
        gx = (self.pos_center[0] + origin_offset[0] * grid_size) / grid_size
        gy = (self.pos_center[1] + origin_offset[1] * grid_size) / grid_size
        col = int(np.round(gx))  # x-axis → col
        row = int(np.round(gy))  # y-axis → row
        rows = costmap.shape[0]
        cols = costmap.shape[1]

        # Compute radius in grid cells, conservatively round UP
        radius_cells = int(np.ceil(self.radius / grid_size))

        # iterate over the columns and rows
        for r in range(radius_cells):
            for c in range(radius_cells):
                x = (col + c) * grid_size - origin_offset[0] * grid_size
                y = (row + r) * grid_size - origin_offset[1] * grid_size
                dist = np.linalg.norm(np.array([x, y]) - self.pos_center)

                if dist < self.radius:
                    costmap[
                        min(max(row + r, 0), rows - 1), min(max(col + c, 0), cols - 1)
                    ] = np.inf
                    costmap[
                        min(max(row + r, 0), rows - 1),
                        min(max(col - c - 1, 0), cols - 1),
                    ] = np.inf
                    costmap[
                        min(max(row - r - 1, 0), rows - 1),
                        min(max(col + c, 0), cols - 1),
                    ] = np.inf
                    costmap[
                        min(max(row - r - 1, 0), rows - 1),
                        min(max(col - c - 1, 0), cols - 1),
                    ] = np.inf
        return costmap

    def h(self, z, safety_margin=0.0):
        if z.ndim == 1:
            z = z[None, :]

        px = z[:, 0]
        py = z[:, 1]
        delta = jnp.stack([px, py], axis=1) - self.pos_center  # (N, 2)

        dist = jnp.linalg.norm(delta, axis=1)
        buffer = self.robot_radius + self.radius + safety_margin
        h_values = dist - buffer
        return h_values  # shape (N,)

    def pyplot_drawing(self, opacity=1.0):
        circle = patches.Circle(
            self.pos_center,
            self.radius,
            edgecolor="red",
            facecolor="black",
            alpha=opacity,
        )
        return circle

    def check_collision(
        self,
        robot_position: np.ndarray,
        robot_width: float,
        robot_height: float,
        safety_margin=0.0,
    ):
        # checks whehter the robot collides with this object
        distance = np.linalg.norm(robot_position - np.array(self.pos_center))
        return distance <= (self.robot_radius + self.radius + safety_margin)

    def check_goal_position(
        self,
        goal_position: np.ndarray,
        robot_width: float,
        robot_height: float,
        extra_safety_margin=0.5,
    ):
        # check if the goal position is feasible
        # return true if it is feasible and false otherwise
        distance = np.linalg.norm(goal_position - np.array(self.pos_center))
        return distance >= (self.robot_radius + self.radius + extra_safety_margin)
