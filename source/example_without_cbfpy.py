import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.05
steps = 200
alpha = 5.0

# Obstacle (circle)
c = np.array([0.0, 0.0])  # center
r = 1.0                   # radius

# Robot state
x = np.array([-3.0, 0.0])   # start left of the obstacle
goal = np.array([3.0, 0.0])  # go to the right

trajectory = [x.copy()]

def h(x):
    return np.linalg.norm(x - c)**2 - r**2

def dh_dx(x):
    return 2 * (x - c)

for _ in range(steps):
    # Nominal controller: go straight to goal
    u_nom = goal - x
    u_nom = u_nom / np.linalg.norm(u_nom) * 1.0  # normalize speed

    # Check CBF constraint
    h_val = h(x)
    dh = dh_dx(x)
    lhs = dh @ u_nom + alpha * h_val

    # If safe, apply nominal
    if lhs >= 0:
        u = u_nom
    else:
        # Project u_nom onto the safe set: solve QP
        # min ||u - u_nom||^2
        # s.t. dh @ u + alpha * h >= 0
        # Solution: constrained projection (analytic)
        lambda_val = -(dh @ u_nom + alpha * h_val) / (np.dot(dh, dh) + 1e-6)
        u = u_nom + lambda_val * dh  # project onto constraint boundary

    # Update state
    x = x + u * dt
    trajectory.append(x.copy())

trajectory = np.array(trajectory)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(c[0] + r * np.cos(theta), c[1] + r * np.sin(theta), 'k--', label='Obstacle')
ax.plot(trajectory[:, 0], trajectory[:, 1], label='Robot trajectory')
ax.plot(goal[0], goal[1], 'go', label='Goal')
ax.set_aspect('equal')
ax.grid()
ax.legend()
plt.title("CBF-protected Trajectory")
plt.show()