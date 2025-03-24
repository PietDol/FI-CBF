from cbfpy import CBFConfig, CLFCBFConfig
import jax.numpy as jnp
import numpy as np

class RobotBaseCBFConfig(CBFConfig):
    def __init__(self, obstacles, robot):
        self.obstacles = obstacles
        self.robot = robot
        super().__init__(n=4, m=2, relax_cbf=False)
    
    def f(self, z):
        px, py, vx, vy = z
        return jnp.array([vx, vy, 0, 0])
    
    def g(self, z):
        return jnp.block([[jnp.eye(2)], [jnp.zeros((2, 2))]])
    
    def h_1(self, z):
        h_values = []

        for obstacle in self.obstacles:
            h_value = obstacle.h(z)
            h_values.append(h_value)
        return jnp.array(h_values)
    
    # def alpha(self, h):
    #     return jnp.array([1.0]) * h

class RobotBaseCLFCBFConfig(CLFCBFConfig):
    def __init__(self, obstacles, robot):
        self.obstacles = obstacles
        self.robot = robot
        super().__init__(n=4, m=2, relax_cbf=False)
    
    def f(self, z):
        px, py, vx, vy = z
        return jnp.array([vx, vy, 0, 0])
    
    def g(self, z):
        return jnp.block([[jnp.eye(2)], [jnp.zeros((2, 2))]])
    
    def h_1(self, z):
        h_values = []

        for obstacle in self.obstacles:
            h_value = obstacle.h(z)
            h_values.append(h_value)
        return jnp.array(h_values)
    
    def V_1(self, z):
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        return jnp.array((c_robot - self.robot.pos_goal) **2)

    