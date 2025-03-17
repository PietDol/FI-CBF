import numpy as np
from abc import ABC, abstractmethod
import pygame

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
    def __init__(self, width, height, pos_center, px_per_meter, screen_width, screen_height):
        self.width = width                  # in m
        self.height = height                # in m
        self.pos_center = pos_center        # in m shape (2,)
        self.px_per_meter = px_per_meter    # in m
        self.screen_width = screen_width    # in px
        self.screen_height = screen_height  # in px
    
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