import numpy as np
from abc import ABC, abstractmethod
import pygame
from robot_base import RobotBase
import jax.numpy as jnp

class Obstacle(ABC):
    """ Abstract class to create all different kinds of obstacls
    """
    @property
    @abstractmethod
    def width_px(self) -> int:
        """
        returns the width in pixels
        """
        pass

    @property
    @abstractmethod
    def height_px(self) -> int:
        # returns the height in pixels
        pass

    @property
    @abstractmethod
    def pos_px(self):
        # returns the postiton of the pixels from the obstacles
        pass

    @abstractmethod
    def generate_cbfs(self):
        # method to create the cbfs
        pass

    @abstractmethod
    def pygame_drawing(self):
        # method to return a pygame drawing
        pass

class RectangleObstacle(Obstacle):
    # object for obstacles
    def __init__(self, width, height, pos_center):
        self.width = width                  # in m
        self.height = height                # in m
        self.pos_center = pos_center        # in m shape (2,)
        self.px_per_meter = None           # in m
        self.screen_width = None           # in px
        self.screen_height = None          # in px
    
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

    def pygame_drawing(self):
        pos_x, pos_y = self.pos_px
        drawing = pygame.Rect(pos_x, pos_y, self.width_px, self.height_px)
        return drawing

    def generate_cbfs(self):
        # TODO: make function to generate the cbfs for this obstacle
        pass

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
    
    def find_closest_point_to_obstacle(self, robot: RobotBase):
        # function to find the closest point between the obstacles edge and the robot
        cx_robot, cy_robot = robot.position
        cx_obstacle, cy_obstacle = self.pos_center
        closest_x = np.clip(cx_robot, cx_obstacle - (self.width / 2), cx_obstacle + (self.width / 2))
        closest_y = np.clip(cy_robot, cy_obstacle - (self.height / 2), cy_obstacle + (self.height / 2))
        return np.array([closest_x, closest_y])
