import numpy as np
import pygame

class RobotBase:
    def __init__(self, width, height, env_config, pos_goal, pos_center_start=np.zeros(2), vel_center_start=np.zeros(2), safety_margin=0.5):
        self.width = width                  # in m
        self.height = height                # in m
        self.radius = np.sqrt((self.width / 2)**2 + (self.height / 2)**2)    # in m (radius of circle that the robot uses)
        self._pos_center = pos_center_start  # in m shape (2,)
        self._vel_center = vel_center_start  # in m/s shape (2,)
        self.px_per_meter = env_config.pixels_per_meter    # in m
        self.screen_width = env_config.screen_width    # in px
        self.screen_height = env_config.screen_height  # in px
        self.pos_goal = pos_goal
        self.safety_margin = safety_margin

    @property
    def position(self):
        return self._pos_center
    
    @position.setter
    def position(self, pos):
        self._pos_center = pos
    
    @property
    def velocity(self):
        return self._vel_center
    
    @velocity.setter
    def velocity(self, vel):
        self._vel_center = vel
    
    def pos_px(self):
        """
        Returns the obstacle's top-left corner position in pixels,
        correctly centered in the pygame coordinate system.
        """
        left_top = self._pos_center + np.array([-self.width / 2, self.height / 2])
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

    def check_goal_reached(self, tolerance=0.01):
        # function to check if the goal is reached
        distance = np.linalg.norm(np.array(self.position) - np.array(self.pos_goal))
        return distance <= tolerance