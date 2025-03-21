"""
This file is used to play around with the CBFpy package. 
In this file we try to create an environment where a robot drives around in an environment with obstacles
"""
import numpy as np
import pygame

from cbfpy.envs.base_env import BaseEnv
from cbfpy import CBF, CLFCBF

from obstacles import RectangleObstacle
from robot_base import RobotBase
from visualization import VisualizeCBF
from robot_base_config import RobotBaseCBFConfig, RobotBaseCLFCBFConfig

import jax.numpy as jnp

class RobotBaseEnv(BaseEnv):
    # Constants for the environment
    PIXELS_PER_METER = 50 * np.array([1, -1])

    # Screen dimensions
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800

    # Colors
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)

    # Robot properties
    ROBOT_WIDTH_M = 1
    ROBOT_HEIGHT_M = 1.5
    ROBOT_WIDTH_PX = PIXELS_PER_METER[0] * ROBOT_WIDTH_M
    ROBOT_HEIGHT_PX = PIXELS_PER_METER[0] * ROBOT_HEIGHT_M

    def __init__(
        self, 
        pos_des,
        obstacles, 
        pos_start=np.zeros(2), 
        vel_start=np.zeros(2),
        # pos_obstacle = np.array([10000, 10000]),
        u_min_max=np.array([-np.inf, np.inf])
    ):
        self.pos_des = pos_des          # desired position [x, y] in m
        self.pos_current = pos_start    # current position [x, y] in m
        self.vel_current = vel_start    # current velocity [vx, vy] in m/s
        self.u_min_max = u_min_max      # min max velocity input

        # set up the environment
        # Initialize Pygame
        pygame.init()
        # Set up the display
        self.screen = pygame.display.set_mode(
            (RobotBaseEnv.SCREEN_WIDTH, RobotBaseEnv.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Robot base experiment")

        # add robot base
        self.robot_base  = RobotBase(
            width=self.ROBOT_WIDTH_M,
            height=self.ROBOT_HEIGHT_M,
            px_per_meter=self.PIXELS_PER_METER,
            screen_width=self.SCREEN_WIDTH,
            screen_height=self.SCREEN_HEIGHT,
            pos_center_start=self.pos_current,
            vel_center_start=self.vel_current
        )

        # add obstacles
        self.obstacles = obstacles
        for obstacle in self.obstacles:
            obstacle.px_per_meter = self.PIXELS_PER_METER
            obstacle.screen_height = self.SCREEN_HEIGHT
            obstacle.screen_width = self.SCREEN_WIDTH

        # position robot base and goal in display
        self.goal_x, self.goal_y = self.position_to_pixels(self.pos_des)

        # some other pygame parameters
        self.font = pygame.font.SysFont("Arial", 20)
        self.fps = 60
        self.dt = 1 / self.fps
        self.running = True

    def position_to_pixels(self, pos):
        # helper function to convert the [x, y] position in m to px
        px = pos * self.PIXELS_PER_METER + np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        return px.astype(int)

    def get_state(self):
        return np.concatenate((self.robot_base.position, self.robot_base.velocity))
    
    def get_desired_state(self):
        return self.pos_des
    
    def apply_control(self, u) -> None:
        # just add the control input to the 
        u = np.clip(u, self.u_min_max[0], self.u_min_max[1])
        vel_current = self.robot_base.velocity + u
        pos_current = self.robot_base.position + vel_current * self.dt
        self.robot_base.velocity = vel_current
        self.robot_base.position = pos_current

        # check collision
        for obstacle in self.obstacles:
            collision = obstacle.check_collision(self.robot_base)
            if collision:
                print('Collision')
    
    def step(self):
        # Handle events
        # This includes where the speed of the main controlled by the user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_UP:
                #     self.leader_vel_des += 1
                # elif event.key == pygame.K_DOWN:
                #     self.leader_vel_des -= 1
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                
        # check for collision
        if self.check_goal_reached():
            print('Goal reached')
            self.running = False
            return
        
        # Clear the screen
        self.screen.fill(RobotBaseEnv.WHITE)   

        # Draw the road
        pygame.draw.rect(
            self.screen,
            RobotBaseEnv.GRAY,
            (
                0,
                0,
                RobotBaseEnv.SCREEN_WIDTH,
                RobotBaseEnv.SCREEN_HEIGHT,
            ),
        )

        # draw the obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, RobotBaseEnv.BLACK, obstacle.pygame_drawing())

        # draw the robot
        robot_base_drawing = self.robot_base.pygame_drawing()
        pygame.draw.rect(self.screen, RobotBaseEnv.RED, robot_base_drawing)

        # Draw the goal as a blue dot
        pygame.draw.circle(self.screen, RobotBaseEnv.BLUE, (self.goal_x, self.goal_y), 5)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(self.fps)
    
    def check_goal_reached(self, tolerance=0.01):
        # function to check if the goal is reached
        distance = np.linalg.norm(np.array(self.robot_base.position) - np.array(self.pos_des))
        return distance <= tolerance
    
def pd_controller(pos_des, pos_current, vel_current, Kp=0.5, Kd=0.3):
    """
    PD controller to drive the robot towards the goal with less overshoot.
    
    Parameters:
        pos_des (np.array): Desired position [x, y]
        pos_current (np.array): Current position [x, y]
        vel_current (np.array): Current velocity [vx, vy]
        Kp (float): Proportional gain
        Kd (float): Derivative gain

    Returns:
        u (np.array): Control input [ux, uy]
    """
    error = pos_des - pos_current
    damping = -Kd * vel_current  # Damping term to reduce overshoot
    u = Kp * error + damping
    return u

def main():
    mode = 0 # 0: PD + CBF, 1: CLF + CBF
    pos_goal = np.array([7, 0])
    obstacles = [
        RectangleObstacle(1, 4, np.array([0, -2.001])),
        RectangleObstacle(2, 2, np.array([3, 4]))
    ]
    env = RobotBaseEnv(pos_goal, pos_start=np.array([-7, 0]), obstacles=obstacles)
    if mode == 0:
        config = RobotBaseCBFConfig(env.obstacles, env.robot_base)
        cbf = CBF.from_config(config)
    elif mode == 1:
        config = RobotBaseCLFCBFConfig(env.obstacles, env.robot_base, pos_goal)
        clf_cbf = CLFCBF.from_config(config)
    else:
        raise f'Incorrect mode!'
    visualizer = VisualizeCBF(obstacles)

    while env.running:
        current_state = env.get_state()
        pos_des = env.get_desired_state()
        if mode == 0:
            nominal_control = pd_controller(pos_des, current_state[:2], current_state[2:])
        elif mode == 1:
            pos_des = np.concatenate((pos_des, np.zeros(2)))
            nominal_control = clf_cbf.controller(current_state, pos_des)

        # safe data for visualizer
        h = config.h_1(current_state)
        h = config.alpha(h)
        visualizer.data['h'].append(np.array(h))
        visualizer.data['robot_pos'].append(current_state[:2])

        # apply safety filter
        if mode == 0:
            u = cbf.safety_filter(current_state, nominal_control)
        elif mode == 1:
            u = nominal_control

        # safe control data for visualizer
        visualizer.data['control_input']['u_cbf'].append(u)
        visualizer.data['control_input']['u_nominal'].append(nominal_control)
        
        # change environment
        env.apply_control(u)
        env.step()
    
    # add last information for visualizer
    visualizer.data['control_input']['u_cbf'].append(u)
    visualizer.data['control_input']['u_nominal'].append(nominal_control)
    h = config.h_1(current_state)
    h = config.alpha(h)
    visualizer.data['h'].append(np.array(h))
    visualizer.data['robot_pos'].append(current_state[:2])

    # ensure Pygame window is fully closed before proceeding
    pygame.quit()

    # generate drawings
    visualizer.create_plot(['control_input', 'h', 'robot_pos'])

if __name__ == "__main__":
    main()