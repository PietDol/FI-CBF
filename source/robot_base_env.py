"""
This file is used to play around with the CBFpy package. 
In this file we try to create an environment where a robot drives around in an environment with obstacles
"""
import numpy as np
import pygame

from cbfpy.envs.base_env import BaseEnv
from cbfpy import CBF, CLFCBF

from obstacles import RectangleObstacle, CircleObstacle
from robot_base import RobotBase
from visualization import VisualizeSimulation
from robot_base_config import RobotBaseCBFConfig, RobotBaseCLFCBFConfig
from env_config import EnvConfig
from a_star import AStarPlanner

import jax.numpy as jnp
from loguru import logger

class RobotBaseEnv(BaseEnv):
    def __init__(
        self, 
        env_config: EnvConfig,
        robot_base: RobotBase,
        obstacles: list, 
        pygame_screen=True,
        fps=60
    ):
        self.env_config = env_config
        self.pygame_screen = pygame_screen

        # set up the environment
        # Initialize Pygame
        pygame.init()
        if self.pygame_screen:
            # Set up the display
            self.screen = pygame.display.set_mode(
                (self.env_config.screen_width, self.env_config.screen_height)
            )
            pygame.display.set_caption("Robot base experiment")
            self.font = pygame.font.SysFont("Arial", 20)

        # add robot base
        self.robot_base  = robot_base

        # add obstacles
        self.obstacles = obstacles

        # position robot base and goal in display
        self.goal_x, self.goal_y = self.position_to_pixels(self.robot_base.pos_goal)

        # some other pygame parameters
        self.fps = fps
        self.dt = 1 / self.fps
        self.running = True
    
    def position_to_pixels(self, pos):
        # helper function to convert the [x, y] position in m to px
        px = pos * self.env_config.pixels_per_meter + np.array([self.env_config.screen_width / 2, self.env_config.screen_height / 2])
        return px.astype(int)

    def get_state(self) -> np.array:
        return np.concatenate((self.robot_base.position, self.robot_base.velocity))
    
    def get_desired_state(self):
        return self.robot_base.inter_pos_goal()
    
    def apply_control(self, u) -> None:
        # just add the control input to the 
        u = np.clip(u, self.robot_base.u_min_max[0], self.robot_base.u_min_max[1])
        vel_current = self.robot_base.velocity + u
        pos_current = self.robot_base.position + vel_current * self.dt
        self.robot_base.velocity = vel_current
        self.robot_base.position = pos_current
    
    def step(self):
        # Handle events
        # This includes where the speed of the main controlled by the user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                
        # check for collision
        if self.robot_base.check_goal_reached():
            self.running = False
            return
        
        # Clear the screen
        if self.pygame_screen:
            self.screen.fill(self.env_config.white)   

            # Draw the road
            pygame.draw.rect(
                self.screen,
                self.env_config.gray,
                (
                    0,
                    0,
                    self.env_config.screen_width,
                    self.env_config.screen_height,
                ),
            )

            # draw the obstacles
            for obstacle in self.obstacles:
                obstacle.pygame_drawing(self.screen, self.env_config.black)

            # draw the robot
            robot_base_drawing = self.robot_base.pygame_drawing()
            pygame.draw.rect(self.screen, self.env_config.red, robot_base_drawing)

            # Draw the goal as a blue dot
            pygame.draw.circle(self.screen, self.env_config.blue, (self.goal_x, self.goal_y), 5)

            # Update the display
            pygame.display.flip()

            # Cap the frame rate
            pygame.time.Clock().tick(self.fps)

    
def pd_controller(pos_des, pos_current, vel_current, Kp=0.2, Kd=0.1):
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
    # create env config
    env_config = EnvConfig(
        pixels_per_meter=50 * np.array([1, -1]),
        screen_width=800,
        screen_height=800
    )

    # create robot
    pos_goal = np.array([10.0, 9.27])
    start = np.array([-1.44, -1.75])
    robot_base = RobotBase(
        width=1,
        height=1,
        env_config=env_config,
        pos_goal=pos_goal,
        pos_center_start=start,
        safety_margin=0.1
    )

    # create obstacles
    obstacles = [
        # RectangleObstacle(1, 6, np.array([0, 0.0]), env_config, robot_base, 0),
        # CircleObstacle(2, np.array([0.0, 0.0]), env_config, robot_base, 1),
        # CircleObstacle(1, np.array([4.0, 4.0]), env_config, robot_base, 2)
        # RectangleObstacle(3, 1, np.array([2, 2.5]), env_config, robot_base, 3),
        # RectangleObstacle(3, 1, np.array([2, -2.5]), env_config, robot_base, 4)
        # RectangleObstacle(2.67, 2.22, np.array([7.77, -3.93]), env_config, robot_base, 5),
        # RectangleObstacle(2.37, 1.23, np.array([7.13, -0.42]), env_config, robot_base, 6),
        RectangleObstacle(1.64, 2.8, np.array([-2.24, -4.37]), env_config, robot_base, 7)
    ]

    # create path
    planner = AStarPlanner(
        costmap_size=(20, 20),
        grid_size=0.1,
        obstacles=obstacles
    )
    path = planner.plan(start, pos_goal)
    robot_base.add_path(path["path_world"])

    # create environment
    env = RobotBaseEnv(
        env_config=env_config, 
        robot_base=robot_base, 
        obstacles=obstacles, 
        pygame_screen=True
    )

    # create the cbf configes 
    mode = 0 # 0: PD + CBF, 1: CLF + CBF
    if mode == 0:
        config = RobotBaseCBFConfig(obstacles, robot_base)
        cbf = CBF.from_config(config)
    elif mode == 1:
        config = RobotBaseCLFCBFConfig(obstacles, robot_base)
        clf_cbf = CLFCBF.from_config(config)
    else:
        raise f'Incorrect mode!'
    
    # create the visualizer object
    visualizer = VisualizeSimulation(pos_goal, planner, obstacles)

    # add path and costmap to visualizer
    visualizer.data.path = path["path_world"]
    visualizer.data.costmap = planner.costmap
    visualizer.data.display_map = planner.compute_distance_map(start=start)

    while env.running:
        current_state = env.get_state()
        pos_des = env.get_desired_state()
        if mode == 0:
            nominal_control = pd_controller(pos_des, current_state[:2], current_state[2:])
        elif mode == 1:
            pos_des = np.concatenate((pos_des, np.zeros(2)))
            u = clf_cbf.controller(current_state, pos_des)

        # safe data for visualizer
        h = config.h_1(current_state)
        h = config.alpha(h)
        visualizer.data.h_true.append(np.array(h))
        visualizer.data.robot_pos.append(current_state[:2])

        # apply safety filter
        if mode == 0:
            u = cbf.safety_filter(current_state, nominal_control)
        elif mode == 1:
            # calculate the nominal control for the visualization
            nominal_control = config.V_1(current_state)

        # safe control data for visualizer
        visualizer.data.u_cbf.append(u)
        visualizer.data.u_nominal.append(nominal_control)
        
        # change environment
        env.apply_control(u)
        env.step()
    
    # add last information for visualizer
    visualizer.data.u_cbf.append(u)
    visualizer.data.u_nominal.append(nominal_control)
    h = config.h_1(current_state)
    h = config.alpha(h)
    visualizer.data.h_true.append(np.array(h))
    visualizer.data.robot_pos.append(current_state[:2])

    # ensure Pygame window is fully closed before proceeding
    pygame.quit()

    # generate drawings
    visualizer.create_full_plot()

if __name__ == "__main__":
    main()