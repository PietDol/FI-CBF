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
        # return jnp.zeros(self.n)
    
    def g(self, z):
        return jnp.block([[jnp.eye(2)], [jnp.zeros((2, 2))]])
    
    def h_1(self, z):
        px, py, vx, vy = z
        h_values = []

        for obstacle in self.obstacles:
            # get the closest point 
            # closest_point = obstacle.find_closest_point_to_obstacle(px, py)

            # # compute vector from closest point to robot
            # vector_closest_to_robot = jnp.array([px, py]) - closest_point

            # # normalize it
            # norm = jnp.linalg.norm(vector_closest_to_robot) + 1e-6
            # normal_vector = vector_closest_to_robot / norm  # add small term to prevent division by zero

            # # calculate value for h
            # h_value = jnp.dot(normal_vector, vector_closest_to_robot) - self.robot.radius - self.robot.safety_margin
            h_value = obstacle.h(z)
            h_values.append(h_value)
        return jnp.array(h_values)
    
    # def alpha(self, h):
    #     return jnp.array([1.0]) * h

class RobotBaseCLFCBFConfig(CLFCBFConfig):
    def __init__(self, obstacles, robot, pos_goal):
        self.obstacles = obstacles
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
        px, py, vx, vy = z
        h_values = []

        for obstacle in self.obstacles:
            # get the closest point 
            closest_point = obstacle.find_closest_point_to_obstacle(px, py)

            # compute vector from closest point to robot
            vector_closest_to_robot = jnp.array([px, py]) - closest_point

            # normalize it
            norm = jnp.linalg.norm(vector_closest_to_robot) + 1e-6
            normal_vector = vector_closest_to_robot / norm  # add small term to prevent division by zero

            # calculate value for h
            h_value = jnp.dot(normal_vector, vector_closest_to_robot) - self.robot.radius - self.safety_margin
            h_values.append(h_value)
        return jnp.array(h_values)
    
    def V_1(self, z):
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        return jnp.array((c_robot - self.pos_goal) **2)

    