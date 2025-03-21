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
        px, py, vx, vy = z

        # get the closest point 
        closest_point = self.obstacle.find_closest_point_to_obstacle(self.robot)

        # compute vector from closest point to robot
        vector_closest_to_robot = jnp.array([px, py]) - closest_point

        # make sure that no term is zero
        # epsilon = 1e-2
        # print('before', vector_closest_to_robot)
        # vector_closest_to_robot = jnp.where(jnp.abs(vector_closest_to_robot) < epsilon, -epsilon, vector_closest_to_robot)
        # print('after', vector_closest_to_robot)

        # normalize it
        norm = jnp.linalg.norm(vector_closest_to_robot) + 1e-6
        normal_vector = vector_closest_to_robot / norm  # add small term to prevent division by zero

        # calculate value for h
        h_value = jnp.dot(normal_vector, vector_closest_to_robot) - self.robot.radius
        return jnp.array([h_value])
            
        # px, py, vx, vy = z
        # c_robot = jnp.array([px, py])
        # c_obs = self.obstacle.pos_center
        # dist_between_centers = jnp.linalg.norm(c_obs - c_robot)
        # diff = jnp.array([np.sqrt((self.obstacle.height / 2)**2 + (self.robot.height / 2)**2)])
        # h = dist_between_centers - diff
        # return h
    
    def alpha(self, h):
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

        # h_x = (px - c_x)**2 - (self.obstacle.width / 2 + self.robot.width / 2)**2
        # h_y = (py - c_y)**2 - (self.obstacle.height/2)**2
        # h = -1 * h_x * h_y
        # # h_x = abs(px - c_x) - ((self.obstacle.width / 2) + (self.robot.width / 2))
        # # h_y = abs(py - c_y) - ((self.obstacle.height / 2) + (self.robot.height / 2))

        # return jnp.array([h])
    
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        c_obs = self.obstacle.pos_center
        dist_between_centers = jnp.linalg.norm(c_obs - c_robot)
        diff = jnp.array([np.sqrt((self.obstacle.height / 2)**2 + (self.robot.height / 2)**2)])
        h = dist_between_centers - diff
        return h
    
    def V_1(self, z):
        px, py, vx, vy = z
        c_robot = jnp.array([px, py])
        return jnp.array((c_robot - self.pos_goal) **2)

    