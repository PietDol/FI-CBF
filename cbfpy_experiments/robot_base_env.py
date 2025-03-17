"""
This file is used to play around with the CBFpy package. 
In this file we try to create an environment where a robot drives around in an environment with obstacles
"""
import numpy as np
import pygame

from cbfpy.envs.base_env import BaseEnv

from obstacles import RectangleObstacle

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
        pos_start=np.zeros(2), 
        vel_start=np.zeros(2),
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
        self.robot_base = pygame.Surface(
            (RobotBaseEnv.ROBOT_WIDTH_PX, RobotBaseEnv.ROBOT_HEIGHT_PX)
        )
        self.robot_base.fill(RobotBaseEnv.RED)

        # add obstacle
        self.obstacle = RectangleObstacle(1, 12, np.array([3, 0]), self.PIXELS_PER_METER, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.obstacle_drawing = self.obstacle.pygame_drawing()

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
    
    def get_robot_px(self):
        px =  self.position_to_pixels(
            self.pos_current + 0.5 * np.array([-self.ROBOT_WIDTH_M, self.ROBOT_HEIGHT_M])
        )
        return px[0], px[1]

    def get_state(self):
        return self.pos_current, self.vel_current
    
    def get_desired_state(self):
        return self.pos_des
    
    def apply_control(self, u) -> None:
        # just add the control input to the 
        u = np.clip(u, self.u_min_max[0], self.u_min_max[1])
        self.vel_current += u
        self.pos_current += self.vel_current * self.dt
    
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

        # draw the obstacle
        pygame.draw.rect(self.screen, RobotBaseEnv.BLACK, self.obstacle_drawing)

        # draw the robot
        self.screen.blit(self.robot_base, self.get_robot_px())

        # Draw the goal as a blue dot
        pygame.draw.circle(self.screen, RobotBaseEnv.BLUE, (self.goal_x, self.goal_y), 5)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(self.fps)

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
    pos_goal = np.array([6, 0])
    env = RobotBaseEnv(pos_goal)

    while env.running:
        pos_current, vel_current = env.get_state()
        pos_des = env.get_desired_state()
        u = pd_controller(pos_des, pos_current, vel_current)
        env.apply_control(u)
        env.step()


if __name__ == "__main__":
    main()