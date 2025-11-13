import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches


def grid_to_world(idx, costmap_size):
    ij = np.array(idx[::-1])
    grid_size = 0.1
    origin_offset = np.array(costmap_size) / (2 * grid_size)
    pos = (ij * grid_size) + (0.5 * grid_size) - (np.array(origin_offset) * grid_size)
    return tuple(pos)


def plot_env(params):
    # function to plot the env
    # Embed fonts nicely in PDF
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
        }
    )

    W = 7.16  # your full \textwidth in inches
    H = 0.7 * W  # good aspect for 2 stacked plots
    fig, ax = plt.subplots(1, 1, figsize=(W, H), sharex=True, constrained_layout=True)

    # load all the data
    path = np.load(f'{params["home_dir"]}{params["variables"][0]}')
    sensor_pos = np.load(f'{params["home_dir"]}{params["variables"][1]}')
    cbf_costmap = np.load(f'{params["home_dir"]}{params["variables"][2]}')
    noise_costmap = np.load(f'{params["home_dir"]}{params["variables"][3]}')
    obstacles = params["obstacles"]
    start = params["start"]
    goal = params["goal"]
    costmap_size = params["costmap_size"]

    # plot start, goal, and planned trajectory
    ax.plot(start[0], start[1], "ro", label="Start")
    ax.plot(goal[0], goal[1], "go", label="Goal")
    ax.plot(path[:, 0], path[:, 1], "--", color="cyan", label="Planned path")

    # add obstacles
    for obstacle in obstacles:
        patch = patches.Circle(
            obstacle[:2],
            obstacle[2],
            edgecolor="black",
            facecolor="gray",
            alpha=1.0,
        )
        ax.add_patch(patch)

    # add sensor noise and sensors
    ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], c="k", label="Sensor pos")
    # Basic min/max values for colormap
    vmax = np.max(noise_costmap)
    vmin = np.min(noise_costmap)

    # Define extent in world coordinates
    rows = int(costmap_size[0] / 0.1)  # grid size = 0.1
    cols = int(costmap_size[0] / 0.1)  # grid size = 0.1
    extent = [
        *grid_to_world((0, 0), costmap_size),
        *grid_to_world((rows, cols), costmap_size),
    ]
    extent = [extent[0], extent[2], extent[1], extent[3]]  # reorder for imshow

    # Plot costmap
    img = ax.imshow(
        noise_costmap,
        cmap="plasma",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    # plot contour for h=0 values
    X = np.linspace(extent[0], extent[1], cbf_costmap.shape[1])
    Y = np.linspace(extent[2], extent[3], cbf_costmap.shape[0])
    X, Y = np.meshgrid(X, Y)

    contour = ax.contour(X, Y, cbf_costmap, levels=[0], colors="white", linewidths=2)
    ax.clabel(contour, fmt="h=0", colors="white", fontsize=9)

    # styling
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_title("Cluttered Environment")
    ax.margins(x=0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axis("equal")
    # ax.set_ylim(-2, 2)

    # Colorbar
    fig = ax.get_figure()
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(r"$\sigma$ [m]", rotation=270, labelpad=15)

    fig.savefig(params["filename"], bbox_inches="tight", transparent=True)


# parameters for gap env
gap_params = {
    "home_dir": "./runs/E1_overall_performance/gap_env/simulation_results/gap_experiment_0_seed_7/simulation_data/",
    "variables": [
        "path.npy",
        "sensor_positions.npy",
        "cbf_costmap.npy",
        "noise_costmap.npy",
    ],
    "obstacles": np.array(
        [
            [-7.0, -3.75, 2.5],
            [-7.0, 3.75, 2.5],
            [-4.0, -3.675, 2.5],
            [-4.0, 3.675, 2.5],
            [-1.0, -3.6, 2.5],
            [-1.0, 3.6, 2.5],
            [2.0, -3.525, 2.5],
            [2.0, 3.525, 2.5],
            [5.0, -3.45, 2.5],
            [5.0, 3.45, 2.5],
            [8.0, -3.375, 2.5],
            [8.0, 3.375, 2.5],
        ]
    ),
    "start": np.array([-12, 0]),
    "goal": np.array([11, 0]),
    "costmap_size": np.array([26, 26]),
    "filename": "gap_env.pdf",
}
cluttered_params = {
    "home_dir": "./runs/E1_overall_performance/cluttered_env/simulation_results/cluttered_experiment_0_seed_7/simulation_data/",
    "variables": [
        "path.npy",
        "sensor_positions.npy",
        "cbf_costmap.npy",
        "noise_costmap.npy",
    ],
    "obstacles": np.array(
        [
            [-6.0, 4.0, 3.5],
            [-1.0, -1.0, 4.0],
            [7.0, -7.0, 2.0],
            [5.0, -1.0, 2.0],
            [6.0, 6.0, 3.0],
            [-0.75, 6.0, 2.0],
            [-6.0, -8.0, 2.0],
        ]
    ),
    "start": np.array([-9.0, -9.0]),
    "goal": np.array([-4.0, 8.0]),
    "costmap_size": np.array([20, 20]),
    "filename": "cluttered_env.pdf",
}

# plot envs
# plot_env(gap_params)
plot_env(cluttered_params)
