import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os

def make_robot_safety_video_two(
    robot_xy_1, safety_margin_1,
    robot_xy_2, safety_margin_2,
    dt, obstacles, start, goal,
    fps=10, outfile="robot_compare.mp4",
    label1="MR-CBF", label2="New method",
):
    """
    robot_xy_1, robot_xy_2      : (N_i, 2) arrays of positions
    safety_margin_1, _2        : (N_i,) or (N_i, K_i) arrays
    dt                         : simulation timestep (seconds), assumed same for both
    obstacles                  : (M, 3) array [x, y, r] for circular obstacles
    start, goal                : (2,) arrays
    fps                        : frames per second of the video
    outfile                    : path to output mp4
    label1, label2             : legend labels for both robots
    """

    # --- Ensure positions are (N,2) ---
    def ensure_xy(arr, name):
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"{name} must be (N,2) or (N,>=2), got {arr.shape}")
        return arr[:, :2]

    robot_xy_1 = ensure_xy(robot_xy_1, "robot_xy_1")
    robot_xy_2 = ensure_xy(robot_xy_2, "robot_xy_2")

    # --- Handle safety margins possibly being (N,K) ---
    def to_scalar_margin(sm, name):
        if sm.ndim == 2:
            return sm.min(axis=1)
        elif sm.ndim == 1:
            return sm
        else:
            raise ValueError(f"{name} must be 1D or 2D, got shape {sm.shape}")

    sm1 = to_scalar_margin(safety_margin_1, "safety_margin_1")
    sm2 = to_scalar_margin(safety_margin_2, "safety_margin_2")

    N1 = len(robot_xy_1)
    N2 = len(robot_xy_2)

    t1 = np.arange(N1) * dt
    t2 = np.arange(N2) * dt

    # Stop when the first robot is done
    T_final = min(t1[-1], t2[-1])
    n_frames = int(np.ceil(T_final * fps))

    # ----- Figure and axes -----
    fig, (ax_env, ax_sm) = plt.subplots(1, 2, figsize=(10, 5))

    # ================= LEFT: ENVIRONMENT / ROBOTS =================

    # Obstacles: filled gray circles with black edge
    for ox, oy, r in obstacles:
        circle = plt.Circle(
            (ox, oy),
            r,
            fill=True,
            facecolor="0.8",   # light gray
            edgecolor="k",
            linewidth=1.0,
        )
        ax_env.add_patch(circle)

    # Start & goal
    ax_env.plot(start[0], start[1], "gx")
    ax_env.plot(goal[0], goal[1], "go")

    # Trajectory so far (dynamic)
    traj1_line, = ax_env.plot([], [], "-",  lw=2, color="tab:red",   label=f"{label1} trajectory")
    traj2_line, = ax_env.plot([], [], "-",  lw=2, color="tab:blue",  label=f"{label2} trajectory")

    # Robot circles (dynamic), radius = 0.7 m (diameter 1.4 m)
    robot_radius = 0.7
    robot1_circle = plt.Circle(
        (0.0, 0.0),
        robot_radius,
        fill=True,
        facecolor="tab:red",   # solid red robot
        edgecolor="k",         # thin black outline
        linewidth=0.2,
        label=label1,
    )
    robot2_circle = plt.Circle(
        (0.0, 0.0),
        robot_radius,
        fill=True,
        facecolor="tab:blue",  # solid blue robot
        edgecolor="k",
        linewidth=0.2,
        label=label2,
    )
    ax_env.add_patch(robot1_circle)
    ax_env.add_patch(robot2_circle)

    # Limits
    xs = np.concatenate([robot_xy_1[:, 0], robot_xy_2[:, 0], [start[0], goal[0]]])
    margin = 1.0
    ax_env.set_xlim(xs.min() - margin, xs.max() + margin)
    ax_env.set_ylim(-8, 8)  # your chosen bounds
    ax_env.set_aspect("equal", "box")
    ax_env.set_title("Robots in environment")
    ax_env.set_xlabel("x [m]")
    ax_env.set_ylabel("y [m]")
    ax_env.grid(True)

    # Legend at top, two columns
    ax_env.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
    )

    # ================= RIGHT: SAFETY MARGINS =================

    sm1_line, = ax_sm.plot([], [], lw=1.5, color="tab:red",  label=label1)
    sm2_line, = ax_sm.plot([], [], lw=1.5, color="tab:blue", label=label2)

    ax_sm.axhline(0.0, color="k", lw=0.8)
    ax_sm.set_xlim(0.0, T_final)

    ymin = float(min(sm1.min(), sm2.min()))
    ymax = float(max(sm1.max(), sm2.max()))
    pad = 0.1 * (ymax - ymin + 1e-6)
    ax_sm.set_ylim(ymin - pad, ymax + pad)

    ax_sm.set_title("Safety margin over time")
    ax_sm.set_xlabel("Time [s]")
    ax_sm.set_ylabel("Safety margin [m]")
    ax_sm.grid(True)
    ax_sm.legend(loc="upper right")

    # ----- Helper: map time to nearest index for each robot -----
    def idx_for_time(t, N):
        return int(np.clip(round(t / dt), 0, N - 1))

    # ----- Update function for each frame -----
    def update(frame):
        t = frame / fps

        i1 = idx_for_time(t, N1)
        i2 = idx_for_time(t, N2)

        # Trajectories up to now
        traj1_line.set_data(robot_xy_1[:i1 + 1, 0], robot_xy_1[:i1 + 1, 1])
        traj2_line.set_data(robot_xy_2[:i2 + 1, 0], robot_xy_2[:i2 + 1, 1])

        # Update robot circle centers
        x1, y1 = robot_xy_1[i1, 0], robot_xy_1[i1, 1]
        x2, y2 = robot_xy_2[i2, 0], robot_xy_2[i2, 1]
        robot1_circle.center = (x1, y1)
        robot2_circle.center = (x2, y2)

        # Safety margins up to now
        sm1_line.set_data(t1[:i1 + 1], sm1[:i1 + 1])
        sm2_line.set_data(t2[:i2 + 1], sm2[:i2 + 1])

        return traj1_line, traj2_line, robot1_circle, robot2_circle, sm1_line, sm2_line

    # ----- Make the video -----
    writer = FFMpegWriter(fps=fps, bitrate=4000)

    with writer.saving(fig, outfile, dpi=300):
        for k in range(n_frames):
            update(k)
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved video to {outfile}")

def prep_inputs_for_video(r_dir):
    robot_xy = np.load(f"{r_dir}/robot_pos.npy")
    safety_margin= np.load(f"{r_dir}/safety_margin.npy")
    safety_margin = np.max(safety_margin, 1)
    dt = 1 / 50
    obstacles = np.array(
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
    )
    start = np.array([-12, 0])
    goal = np.array([11, 0])
    return robot_xy, safety_margin, dt, obstacles, start, goal


r0_dir = "./runs/E1_overall_performance/gap_env/simulation_results/gap_experiment_0_seed_7/simulation_data"
r2_dir = "./runs/E1_overall_performance/gap_env/simulation_results/gap_experiment_3_seed_7/simulation_data"
mr_robot_xy, mr_safety_margin, dt, obstacles, start, goal = prep_inputs_for_video(r0_dir)
new_robot_xy, new_safety_margin, dt, obstacles, start, goal = prep_inputs_for_video(r2_dir)

make_robot_safety_video_two(
    new_robot_xy, new_safety_margin,
    mr_robot_xy, mr_safety_margin,
    dt, obstacles, start, goal,
    fps=30,
    outfile="gap_env_compare_mrcbf_new.mp4",
    label1="CALM-CBF",
    label2="MR-CBF", 
)
