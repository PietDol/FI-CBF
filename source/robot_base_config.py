from cbfpy import CBFConfig, CLFCBFConfig
import jax.numpy as jnp

class RobotBaseCBFConfig(CBFConfig):
    def __init__(self, obstacles, robot):
        self.obstacles = obstacles
        self.robot = robot
        super().__init__(n=4, m=2, relax_cbf=False)
    
    def f(self, z):
        px, py, vx, vy = z
        return jnp.array([vx, vy, 0, 0])
    
    def g(self, z):
        # return jnp.block([[jnp.zeros((2, 2))], [jnp.eye(2)]])
        return jnp.block([[jnp.eye(2)], [jnp.zeros((2, 2))]])
    
    def h_1(self, z, safety_margin=0.0, batched=False):
        # batched -> faster for costmap calculation
        if z.ndim == 1:
            z = z[None, :]  # Reshape to (1, 4)

        # z: (N, 4)
        h_values = []
        for obstacle in self.obstacles:
            h_value = obstacle.h(z, safety_margin)  # (N,)
            h_values.append(h_value)
        
        h_values = jnp.stack(h_values, axis=1)  # (N, num_obstacles)

        if not batched:
            h_values = jnp.squeeze(h_values, 0)
        return h_values
    
    def alpha_batch(self, h_values):
        # h_values: (N, num_obstacles)
        # Apply alpha elementwise
        return jnp.vectorize(self.alpha)(h_values)


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

    