import numpy as np
import pygame

class RobotBase:
    def __init__(self, width, height, env_config, pos_goal, path=[], pos_center_start=np.zeros(2), vel_center_start=np.zeros(2), safety_margin=0.0, u_min_max=np.array([-np.inf, np.inf])):
        self.width = width                  # in m
        self.height = height                # in m
        self.radius = np.sqrt((self.width / 2)**2 + (self.height / 2)**2)    # in m (radius of circle that the robot uses)
        self._pos_center = pos_center_start  # in m shape (2,)
        self._vel_center = vel_center_start  # in m/s shape (2,)
        self.px_per_meter = env_config.pixels_per_meter    # in m
        self.screen_width = env_config.screen_width    # in px
        self.screen_height = env_config.screen_height  # in px
        self.pos_goal = pos_goal            # goal position in m shape (2,)
        self.path = path                    # list of the path coordinates
        self.path_idx = 0                   # index which location is the current goal location
        self.safety_margin = safety_margin  # in m
        self.u_min_max = u_min_max

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
    
    def inter_pos_goal(self, tolerance=1.5):
        # calculate the intermediate goal location
        # check if the goal is reached
        pos_inter = np.array(self.path[self.path_idx])
        reached = self.check_goal_reached(tolerance=tolerance, pos_inter=pos_inter)

        if len(self.path) < 1:
            return self.pos_goal

        if reached:
            self.path_idx = min([len(self.path) - 1, self.path_idx + 1])

        return self.path[self.path_idx]
    
    def add_path(self, path):
        # method to add the path
        self.path = path
    
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

    def check_goal_reached(self, tolerance=0.01, pos_inter=np.array([])):
        # function to check if the goal is reached
        if pos_inter.shape[0] < 1:
            distance = np.linalg.norm(np.array(self.position) - np.array(self.pos_goal))
        else:
            distance = np.linalg.norm(np.array(self.position) - np.array(pos_inter))
        
        return distance <= tolerance