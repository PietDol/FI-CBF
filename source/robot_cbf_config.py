from cbfpy import CBFConfig
import jax.numpy as jnp
import numpy as np


class RobotCBFConfig(CBFConfig):
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.num_obstacles = len(obstacles)
        init_safety_margin = (np.ones(self.num_obstacles), False)
        super().__init__(n=4, m=2, relax_cbf=False, init_args=init_safety_margin)

    def f(self, z):
        px, py, vx, vy = z
        return jnp.array([vx, vy, 0, 0])

    def g(self, z):
        # return jnp.block([[jnp.zeros((2, 2))], [jnp.eye(2)]])
        return jnp.block([[jnp.eye(2)], [jnp.zeros((2, 2))]])

    def h_1(self, z, safety_margin, batched=False):
        # batched -> faster for costmap calculation
        # N is the number of points in the batch
        if z.ndim == 1:
            z = z[None, :]  # Reshape to (1, 4)

        # z: (N, 4)
        h_values = []
        for i, obstacle in enumerate(self.obstacles):
            h_value = obstacle.h(z, safety_margin[i])  # (N,)
            h_values.append(h_value)

        h_values = jnp.stack(h_values, axis=1)  # (N, num_obstacles)

        if not batched:
            h_values = jnp.squeeze(h_values, 0)
        return h_values

    def alpha_batch(self, h_values):
        # h_values: (N, num_obstacles)
        # Apply alpha elementwise
        return jnp.vectorize(self.alpha)(h_values)
