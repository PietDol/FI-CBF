from cbfpy import CBFConfig, CLFCBFConfig
import jax.numpy as jnp
import numpy as np

class RobotBaseCBFConfig(CBFConfig):
    def __init__(self, obstacle, robot):
        self.obstacle = obstacle
        self.robot = robot
        super().__init__(n=4, m=2)
    
    def f(self, z):
        px, py, vx, vy = z
        return jnp.array([vx, vy, 0, 0])
    
    def g(self, z):
        px, py, vx, vy = z
        g = np.zeros((4, 2))
        g[0, 0] = 1
        g[1, 1] = 1
        return jnp.array(g)
    
    def h_1(self, z):
        # px, py, vx, vy = z
        # c_x, c_y = self.obstacle.pos_center

        # h_x = abs(px - c_x) - ((self.obstacle.width / 2) + (self.robot.width / 2))
        # h_y = abs(py - c_y) - ((self.obstacle.height / 2) + (self.robot.height / 2))

        # return jnp.array([h_x, h_y])
    
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        c_obs = self.obstacle.pos_center
        dist_between_centers = jnp.linalg.norm(c_obs - c_robot)
        diff = jnp.array([(self.obstacle.height / 2) + (self.robot.height / 2)])
        h = dist_between_centers - diff
        return h

class RobotBaseCLFCBFConfig(CLFCBFConfig):
    def __init__(self, obstacle, robot, pos_goal):
        self.obstacle = obstacle
        self.robot = robot
        self.pos_goal = pos_goal
        super().__init__(n=4, m=2)
    
    def f(self, z):
        px, py, vx, vy = z
        return jnp.array([vx, vy, 0, 0])
    
    def g(self, z):
        px, py, vx, vy = z
        g = np.zeros((4, 2))
        g[0, 0] = 1
        g[1, 1] = 1
        return jnp.array(g)
    
    def h_1(self, z):
        # px, py, vx, vy = z
        # c_x, c_y = self.obstacle.pos_center

        # h_x = abs(px - c_x) - ((self.obstacle.width / 2) + (self.robot.width / 2))
        # h_y = abs(py - c_y) - ((self.obstacle.height / 2) + (self.robot.height / 2))

        # return jnp.array([h_x, h_y])
    
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        c_obs = self.obstacle.pos_center
        dist_between_centers = jnp.linalg.norm(c_obs - c_robot)
        diff = jnp.array([(self.obstacle.height / 2) + (self.robot.height / 2)])
        h = dist_between_centers - diff
        return h
    
    def V_1(self, z):
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        return jnp.array((c_robot - self.pos_goal) **2)

    