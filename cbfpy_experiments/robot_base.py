import numpy as np
import pygame

class RobotBase:
    def __init__(self, width, height, px_per_meter, screen_width, screen_height, pos_center_start, vel_center_start):
        self.width = width                  # in m
        self.height = height                # in m
        self.pos_center = pos_center_start  # in m shape (2,)
        self.vel_center = vel_center_start  # in m/s shape (2,)
        self.px_per_meter = px_per_meter    # in m
        self.screen_width = screen_width    # in px
        self.screen_height = screen_height  # in px

    @property
    def position(self):
        return self.pos_center
    
    @position.setter
    def position(self, pos):
        self.pos_center = pos
    
    @property
    def velocity(self):
        return self.vel_center
    
    @velocity.setter
    def velocity(self, vel):
        self.vel_center = vel
    
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
        width_px = self.width * self.px_per_meter[0]
        height_px = self.height * self.px_per_meter[0]
        drawing = pygame.Rect(self.pos_px(), (width_px, height_px))
        return drawing