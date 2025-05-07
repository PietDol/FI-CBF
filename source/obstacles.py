import numpy as np
from abc import ABC, abstractmethod
import pygame
from robot_base import RobotBase
from env_config import EnvConfig
import jax.numpy as jnp
import matplotlib.patches as patches

class Obstacle(ABC):
    """ Abstract class to create all different kinds of obstacls
    """
    @abstractmethod
    def h(self):
        # method to get the value of the cbf
        pass

    @abstractmethod
    def pygame_drawing(self):
        # method to return a pygame drawing
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
    def __init__(self, width, height, pos_center, env_config: EnvConfig, robot: RobotBase, id=int):
        self.width = width                  # in m
        self.height = height                # in m
        self.pos_center = pos_center        # in m shape (2,)
        self.px_per_meter = env_config.pixels_per_meter     # in m
        self.screen_width = env_config.screen_width         # in px
        self.screen_height = env_config.screen_height       # in px
        self.robot = robot                  # to which robot this obstacle is linked
        self.id = id
    
    @property
    def width_px(self):
        # returns the width in pixels
        return int(self.width * self.px_per_meter[0])
    
    @property
    def height_px(self):
        # returns the height in pixels
        return int(self.height * self.px_per_meter[0])
    
    @property
    def pos_px(self):
        """
        Returns the obstacle's top-left corner position in pixels,
        correctly centered in the pygame coordinate system.
        """
        left_top = self.pos_center + np.array([-self.width / 2, self.height / 2])
        # Convert obstacle position from meters to pixels
        left_top *= self.px_per_meter

        # Adjust so that (0,0) is at the center of the screen
        left_top += np.array([self.screen_width / 2, self.screen_height / 2])

        return int(left_top[0]), int(left_top[1])

    def pygame_drawing(self, screen, color):
        pos_x, pos_y = self.pos_px
        drawing = pygame.Rect(pos_x, pos_y, self.width_px, self.height_px)
        pygame.draw.rect(screen, color, drawing)

    def pyplot_drawing(self, opacity=1.0):
        cx, cy = self.pos_center
        bottom_left_x = cx - self.width / 2
        bottom_left_y = cy - self.height / 2

        square = patches.Rectangle((bottom_left_x, bottom_left_y), 
                                self.width, self.height, edgecolor='red',
                                facecolor='black', alpha=opacity, fill=True)
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
        col_min = min(max(int((x_min + origin_offset[0] * grid_size) / grid_size), 0), costmap.shape[1])
        col_max = max(min(int((x_max + origin_offset[0] * grid_size) / grid_size), costmap.shape[1]), 0)
        row_min = min(max(int((y_min + origin_offset[1] * grid_size) / grid_size), 0), costmap.shape[0])
        row_max = max(min(int((y_max + origin_offset[1] * grid_size) / grid_size), costmap.shape[0]), 0)

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

        h_values = jnp.sum(normal_vectors * vectors, axis=1) - self.robot.radius - safety_margin
        return h_values  # shape (N,)

    def check_collision(self, robot: RobotBase, safety_margin=0.0):
        # function to check if the robot is in collision with this obstacle
        # get the centers of the robot and the obstacle
        cx_robot, cy_robot = robot.position 
        cx_obstacle, cy_obstacle = self.pos_center

        # Check x-axis overlap
        x_overlap = abs(cx_robot - cx_obstacle) <= (robot.width / 2) + (self.width / 2) + safety_margin

        # Check y-axis overlap
        y_overlap = abs(cy_robot - cy_obstacle) <= (robot.height / 2) + (self.height / 2) + safety_margin

        return x_overlap and y_overlap
    
    def find_closest_point_to_obstacle(self, px, py):
        # batched
        # px, py: (N,)
        cx_obstacle, cy_obstacle = self.pos_center
        closest_x = jnp.clip(px, cx_obstacle - (self.width / 2), cx_obstacle + (self.width / 2))
        closest_y = jnp.clip(py, cy_obstacle - (self.height / 2), cy_obstacle + (self.height / 2))
        return jnp.stack([closest_x, closest_y], axis=1)  # (N, 2)
    
    def check_goal_position(self, robot: RobotBase, extra_safety_margin=0.5):
        # check if the goal position is feasible
        cx_robot, cy_robot = robot.pos_goal 
        cx_obstacle, cy_obstacle = self.pos_center

        # Check x-axis overlap
        x_overlap = abs(cx_robot - cx_obstacle) <= (robot.width / 2) + (self.width / 2) + extra_safety_margin

        # Check y-axis overlap
        y_overlap = abs(cy_robot - cy_obstacle) <= (robot.height / 2) + (self.height / 2) + extra_safety_margin

        return not (x_overlap and y_overlap)


class CircleObstacle(Obstacle):
    # object for circular obstacles
    def __init__(self, radius, pos_center, env_config: EnvConfig, robot: RobotBase, id: int):
        self.radius = radius            # in m
        self.pos_center = pos_center    # in m shape (2,)            
        self.env_config = env_config    
        self.robot = robot
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
                    costmap[min(max(row+r, 0), rows - 1), min(max(col+c, 0), cols - 1)] = np.inf
                    costmap[min(max(row+r, 0), rows - 1), min(max(col-c-1, 0), cols - 1)] = np.inf
                    costmap[min(max(row-r-1, 0), rows - 1), min(max(col+c, 0), cols - 1)] = np.inf
                    costmap[min(max(row-r-1, 0), rows - 1), min(max(col-c-1, 0), cols - 1)] = np.inf
        return costmap

    def h(self, z, safety_margin=0.0):
        if z.ndim == 1:
            z = z[None, :]

        px = z[:, 0]
        py = z[:, 1]
        delta = jnp.stack([px, py], axis=1) - self.pos_center  # (N, 2)

        dist = jnp.linalg.norm(delta, axis=1)
        buffer = self.robot.radius + self.radius + safety_margin
        h_values = dist - buffer
        return h_values # shape (N,)

    def pygame_drawing(self, screen, color):
        radius_px = int(self.env_config.pixels_per_meter[0] * self.radius)
        pos_px = self.pos_center * self.env_config.pixels_per_meter
        pos_px += np.array([self.env_config.screen_width / 2, self.env_config.screen_height / 2])
        pygame.draw.circle(screen, color, (pos_px[0], pos_px[1]), radius_px)

    def pyplot_drawing(self, opacity=1.0):
        circle = patches.Circle(self.pos_center, self.radius, edgecolor='red', facecolor='black', alpha=opacity)
        return circle

    def check_collision(self, robot: RobotBase, safety_margin=0.0):
        # checks whehter the robot collides with this object
        distance = np.linalg.norm(np.array(robot.position) - np.array(self.pos_center))
        return distance <= (robot.radius + self.radius + safety_margin)

    def check_goal_position(self, robot: RobotBase, extra_safety_margin=0.5):
        # check if the goal position is feasible
        # return true if it is feasible and false otherwise
        distance = np.linalg.norm(np.array(robot.pos_goal) - np.array(self.pos_center))
        return distance >= (robot.radius + self.radius + extra_safety_margin)
