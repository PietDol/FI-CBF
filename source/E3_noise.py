import numpy as np

# this file is created to analyze the data and make the plots for E3
# in this experiment we vary the maximum noise and see how it affects
# our proposed method
noise_dirs = ["015", "02"]
noise_values = [0.15, 0.2]
close_circles = [
    {"x": 8.0, "y": 3.375, "r": 2.5},
    {"x": 2.0, "y": 3.525, "r": 2.5},
]


def calculate_w_min(x, x1, y1, r):
    # calculate w_min = 2*y1 - 2 * sqrt(r^2 - (x - x1)^2)
    dx = np.asarray(x) - x1
    inside = r**2 - dx**2
    # where line intersects the discs, take sqrt; otherwise contribute 0
    sqrt_term = np.sqrt(np.maximum(inside, 0.0))
    gap = 2 * y1 - 2 * sqrt_term
    return np.maximum(gap, 0.0)


for i, (noise_dir, noise_value) in enumerate(zip(noise_dirs, noise_values)):
    path = f"./runs/E3_noise/noise_{noise_dir}/simulation_results/gap_experiment_3_seed_7/simulation_data/robot_pos.npy"
    robot_pos = np.load(path)
    x_max = np.amax(robot_pos[:, 0])
    w_min = calculate_w_min(
        x_max, close_circles[i]["x"], close_circles[i]["y"], close_circles[i]["r"]
    )
    print(f"w_min @ {np.round(x_max, 2)}: {np.round(w_min, 2)}")
