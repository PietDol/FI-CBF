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

class RectangleObstacle(Obstacle):
    # object for obstacles
    def __init__(self, width, height, pos_center, env_config: EnvConfig, robot: RobotBase):
        self.width = width                  # in m
        self.height = height                # in m
        self.pos_center = pos_center        # in m shape (2,)
        self.px_per_meter = env_config.pixels_per_meter     # in m
        self.screen_width = env_config.screen_width         # in px
        self.screen_height = env_config.screen_height       # in px
        self.robot = robot                  # to which robot this obstacle is linked
    
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

    def pyplot_drawing(self):
        cx, cy = self.pos_center
        bottom_left_x = cx - self.width / 2
        bottom_left_y = cy - self.height / 2

        square = patches.Rectangle((bottom_left_x, bottom_left_y), 
                                self.width, self.height, 
                                color='black', fill=True)
        return square
    
    def add_obstacle_to_costmap(self, costmap, grid_size=1):
        # add the obstacles to the costmap
        # gridsize is the length in m of 1 grid (assumes it is squared)
        x_min = int(self.pos_center[0] / grid_size - self.width / (2 * grid_size))
        x_max = int(self.pos_center[0] / grid_size + self.width / (2 * grid_size))
        y_min = int(self.pos_center[1] / grid_size - self.height / (2 * grid_size))
        y_max = int(self.pos_center[1] / grid_size + self.height / (2 * grid_size))
        costmap[x_min:x_max, y_min:y_max] = np.inf
        return costmap

    def h(self, z):
        px, py, _, _ = z

        # get the closest point
        closest_point = self.find_closest_point_to_obstacle(px, py)

        # compute vector from closest point to robot
        vector_closest_to_robot = jnp.array([px, py]) - closest_point

        # normalize it
        norm = jnp.linalg.norm(vector_closest_to_robot) + 1e-6
        normal_vector = vector_closest_to_robot / norm  # add small term to prevent division by zero

        # calculate value for h
        h_value = jnp.dot(normal_vector, vector_closest_to_robot) - self.robot.radius - self.robot.safety_margin
        return jnp.array(h_value)

    def check_collision(self, robot: RobotBase):
        # function to check if the robot is in collision with this obstacle
        # get the centers of the robot and the obstacle
        cx_robot, cy_robot = robot.position 
        cx_obstacle, cy_obstacle = self.pos_center

        # Check x-axis overlap
        x_overlap = abs(cx_robot - cx_obstacle) <= (robot.width / 2) + (self.width / 2)

        # Check y-axis overlap
        y_overlap = abs(cy_robot - cy_obstacle) <= (robot.height / 2) + (self.height / 2)

        return x_overlap and y_overlap
    
    def find_closest_point_to_obstacle(self, cx_robot, cy_robot):
        # function to find the closest point between the obstacles edge and the robot
        cx_obstacle, cy_obstacle = self.pos_center
        closest_x = jnp.clip(cx_robot, cx_obstacle - (self.width / 2), cx_obstacle + (self.width / 2))
        closest_y = jnp.clip(cy_robot, cy_obstacle - (self.height / 2), cy_obstacle + (self.height / 2))
        return jnp.array([closest_x, closest_y])
    
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
    def __init__(self, radius, pos_center, env_config: EnvConfig, robot: RobotBase):
        self.radius = radius            # in m
        self.pos_center = pos_center    # in m shape (2,)            
        self.env_config = env_config    
        self.robot = robot
    
    def h(self, z):
        px, py, _, _ = z
        delta = jnp.array([px, py]) - self.pos_center
        h = jnp.dot(delta, delta) - (self.robot.radius + self.radius + self.robot.safety_margin)**2
        return h

    def pygame_drawing(self, screen, color):
        radius_px = int(self.env_config.pixels_per_meter[0] * self.radius)
        pos_px = self.pos_center * self.env_config.pixels_per_meter
        pos_px += np.array([self.env_config.screen_width / 2, self.env_config.screen_height / 2])
        pygame.draw.circle(screen, color, (pos_px[0], pos_px[1]), radius_px)

    def pyplot_drawing(self):
        circle = patches.Circle(self.pos_center, self.radius, color='black')
        return circle

    def check_collision(self, robot: RobotBase):
        # checks whehter the robot collides with this object
        distance = np.linalg.norm(np.array(robot.position) - np.array(self.pos_center))
        return distance <= (robot.radius + self.radius)

    def check_goal_position(self, robot: RobotBase, extra_safety_margin=0.5):
        # check if the goal position is feasible
        # return true if it is feasible and false otherwise
        distance = np.linalg.norm(np.array(robot.pos_goal) - np.array(self.pos_center))
        return distance >= (robot.radius + self.radius + extra_safety_margin)
