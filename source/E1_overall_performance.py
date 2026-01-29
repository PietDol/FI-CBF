import numpy as np
from dataclasses import dataclass


# script to calculate all the metrics for E1
def calculate_w_min_2d(x1, y1, r1, x2, y2, r2, x, y, vx, vy, eps=1e-12):
    """
    Gap along a line between boundaries of two circles.

    Line: passes through (x, y) with direction vector (vx, vy).
    Circles: (x1, y1, r1) and (x2, y2, r2).

    Returns nonnegative scalar (or np.ndarray if inputs are arrays).
    """
    p = np.array([x, y], dtype=float)
    v = np.array([vx, vy], dtype=float)
    nv = np.linalg.norm(v, axis=0) if np.ndim(v) else np.linalg.norm(v)
    if np.all(nv < eps):
        raise ValueError("Direction vector (vx, vy) must be non-zero.")
    v = v / nv  # unit direction

    def interval_for_circle(cx, cy, r):
        c = np.array([cx, cy], dtype=float)
        d = c - p  # vector from line point to center
        t_center = np.dot(v, d)  # projection along the line
        d_perp = d - t_center * v  # component perpendicular to line
        inside = r * r - np.dot(d_perp, d_perp)
        s = np.sqrt(max(inside, 0.0))  # half-length of intersection interval
        return t_center - s, t_center + s  # [start, end] in t

    a1, b1 = interval_for_circle(x1, y1, r1)
    a2, b2 = interval_for_circle(x2, y2, r2)

    # Distance between intervals [a1,b1] and [a2,b2]
    # If they overlap/touch => gap = 0
    if b1 < a2:
        return a2 - b1
    elif b2 < a1:
        return a1 - b2
    else:
        return 0.0

def detect_deadlock(
    positions: np.ndarray,
    dt: float,
    u_cbf: np.ndarray | None = None,
    *,
    v_thresh: float = 0.02,     # m/s: "near zero" speed
    window_sec: float = 2.0,    # seconds of sustained stall to call deadlock
    u_thresh: float = 0.05,     # optional: near-zero control (||u_cbf||)
    goal: np.ndarray | None = None,
    goal_tol: float = 0.10      # meters: if within this, we don't call deadlock
):
    """
    Detects first time index where the robot is in a deadlock.

    Parameters
    ----------
    positions : (T, D) array
        Robot positions over time (D=2 or 3). Assumes uniform dt.
    dt : float
        Sampling time [s].
    u_cbf : (T, M) array or None
        (Optional) CBF-filtered control per timestep. Only the norm is used.
    v_thresh : float
        Speed threshold to consider "stopped".
    window_sec : float
        Duration that speed must remain below threshold to trigger deadlock.
    u_thresh : float
        (If u_cbf provided) Mean ||u_cbf|| over the window must also be < u_thresh.
    goal : (D,) array or None
        (Optional) Goal position to suppress 'deadlock' if robot is at goal.
    goal_tol : float
        Distance under which we consider the goal reached.

    Returns
    -------
    deadlocked : bool
        True if a deadlock was detected.
    idx : int | None
        Index (in positions) where the deadlock window ends (first detection).
    info : dict
        Diagnostics: {'window_samples', 'mean_speed', 'mean_u_cbf', 'reason'}.
    """
    pos = np.asarray(positions)
    if pos.ndim != 2 or pos.shape[0] < 2:
        raise ValueError("positions must be (T, D) with T >= 2")

    # Speeds per step (T-1,)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt

    # Rolling mean speed over a window
    W = max(1, int(round(window_sec / dt)))
    if len(v) < W:
        return False, None, {"window_samples": W, "reason": "sequence too short"}

    # Efficient rolling mean via cumulative sum
    cs = np.cumsum(np.concatenate(([0.0], v)))
    roll_mean_v = (cs[W:] - cs[:-W]) / W  # length T-1-(W-1) = T-W

    # Candidate deadlock indices where mean speed < threshold
    cand = np.nonzero(roll_mean_v < v_thresh)[0]
    if cand.size == 0:
        return False, None, {"window_samples": W, "reason": "no low-speed window"}

    # Adjust to index in positions (window ends at i_end = i + W)
    for i in cand:
        i_end = i + W  # corresponds to positions index i_end
        # Optional: ignore if near goal
        if goal is not None:
            if np.linalg.norm(pos[i_end] - goal) <= goal_tol:
                continue

        # Optional: require also that CBF-filtered control is small
        mean_u = None
        if u_cbf is not None:
            u = np.asarray(u_cbf)
            if u.shape[0] != pos.shape[0]:
                raise ValueError("u_cbf must have the same length T as positions")
            # use [i_end-W, i_end) range in time, i.e., indices [i_end-W, ..., i_end-1]
            u_norm = np.linalg.norm(u[i_end-W:i_end], axis=1)
            mean_u = float(np.mean(u_norm))
            if mean_u >= u_thresh:
                # Control is not clamped small; likely not a CBF-induced deadlock
                # (could be actuator saturation or pushing—skip this candidate)
                continue

        # If we reach here, it’s a deadlock
        mean_v = float(roll_mean_v[i])
        return True, int(i_end), {
            "window_samples": W,
            "mean_speed": mean_v,
            "mean_u_cbf": mean_u,
            "reason": "low speed sustained" + (" with low u_cbf" if u_cbf is not None else "")
        }

    # No candidate satisfied all conditions
    return False, None, {
        "window_samples": W,
        "reason": "no window passed auxiliary checks (goal or u_cbf)"
    }    

@dataclass
class InterruptEvent:
    start: int
    end: int           # inclusive
    duration: int      # samples
    peak: float        # max ||u_cbf - u_nom||
    mean: float        # mean ||u_cbf - u_nom||
    area_L1: float     # sum ||u_cbf - u_nom|| * dt
    energy_L2: float   # sum ||u_cbf - u_nom||^2 * dt

@dataclass
class InterruptSummary:
    n_events: int
    fraction_time_interrupted: float
    n_interuptions: float
    total_area_L1: float
    total_energy_L2: float
    peak_overall: float
    events: list       # list[InterruptEvent]

def analyze_cbf_interruptions(
    u_nom: np.ndarray,
    u_cbf: np.ndarray,
    *,
    dt: float = 1.0,
    abs_tol: float = 1e-3,
    rel_tol: float = 0.05,
    min_duration: int = 1,
    min_separation: int = 1
) -> InterruptSummary:
    """
    Analyze when and how strongly the CBF modifies the nominal control.

    Parameters
    ----------
    u_nom, u_cbf : (T, M) arrays
        Time-aligned nominal and CBF-filtered commands.
    dt : float
        Sample period [s]. Used to scale L1 area and L2 energy.
    abs_tol : float
        Absolute tolerance for declaring an interruption.
    rel_tol : float
        Relative tolerance: interruption if ||Δu|| > abs_tol + rel_tol * ||u_nom||.
    min_duration : int
        Minimum consecutive samples (in one episode) to count as an event.
    min_separation : int
        Minimum off-gap (non-interrupted samples) to split two events.

    Returns
    -------
    InterruptSummary
    """
    u_nom = np.asarray(u_nom, dtype=float)
    u_cbf = np.asarray(u_cbf, dtype=float)
    if u_nom.shape != u_cbf.shape or u_nom.ndim != 2:
        raise ValueError("u_nom and u_cbf must be (T, M) with matching shapes")

    du = u_cbf - u_nom
    du_norm = np.linalg.norm(du, axis=1)
    nom_norm = np.linalg.norm(u_nom, axis=1)
    thresh = abs_tol + rel_tol * nom_norm
    interrupted = du_norm > thresh

    # Optionally merge short gaps (< min_separation) between interrupted segments
    if min_separation > 1 and interrupted.any():
        # Dilate the mask by closing gaps shorter than min_separation
        # (morphological closing via run-length encoding)
        mask = interrupted.astype(int)
        # find gaps
        diff = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1
        # fill short gaps between consecutive segments
        for k in range(len(starts) - 1):
            gap = starts[k+1] - ends[k] - 1
            if 0 < gap < min_separation:
                mask[ends[k]+1:starts[k+1]] = 1
        interrupted = mask.astype(bool)

    # Extract events (runs of True) and filter by min_duration
    diff = np.diff(np.concatenate(([0], interrupted.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1

    events: list[InterruptEvent] = []
    for s, e in zip(starts, ends):
        duration = e - s + 1
        if duration < min_duration:
            continue
        seg = du_norm[s:e+1]
        events.append(
            InterruptEvent(
                start=int(s),
                end=int(e),
                duration=int(duration),
                peak=float(seg.max()),
                mean=float(seg.mean()),
                area_L1=float(seg.sum() * dt),
                energy_L2=float(np.sum(seg**2) * dt),
            )
        )

    # Global stats
    frac_time = float(np.mean(interrupted)) if du_norm.size else 0.0
    num_interuptions = float(np.sum(interrupted)) if du_norm.size else 0.0
    total_L1 = float(np.sum([ev.area_L1 for ev in events])) if events else 0.0
    total_L2 = float(np.sum([ev.energy_L2 for ev in events])) if events else 0.0
    peak_all = float(np.max(du_norm)) if du_norm.size else 0.0

    return InterruptSummary(
        n_events=len(events),
        fraction_time_interrupted=frac_time,
        n_interuptions=num_interuptions,
        total_area_L1=total_L1,
        total_energy_L2=total_L2,
        peak_overall=peak_all,
        events=events,
    )


# gap env
def gap_env():
    # function for gap env metrics
    d_robot = (2)**0.5
    print("- - - Gap environment - - -")
    # important parameters
    home_dir = "./runs/E1_overall_performance/gap_env/simulation_results"
    dirs = [
        [
            f"{home_dir}/gap_experiment_0_seed_7",
            f"{home_dir}/gap_experiment_0_seed_15",
            f"{home_dir}/gap_experiment_0_seed_22",
        ],
        [
            f"{home_dir}/gap_experiment_1_seed_7",
            f"{home_dir}/gap_experiment_1_seed_15",
            f"{home_dir}/gap_experiment_1_seed_22",
        ],
        [
            f"{home_dir}/gap_experiment_3_seed_7",
            f"{home_dir}/gap_experiment_3_seed_15",
            f"{home_dir}/gap_experiment_3_seed_22",
        ],
    ]
    close_circles = [
        {
            "x1": -4.0,
            "y1": -3.675,
            "r1": 2.5,
            "x2": -4.0,
            "y2": 3.675,
            "r2": 2.5,
        },  # robot 0
        {
            "x1": 8.0,
            "y1": 3.375,
            "r1": 2.5,
            "x2": 8.0,
            "y2": -3.375,
            "r2": 2.5,
        },  # robot 1
    ]

    for i, dir in enumerate(dirs):
        w_mins = []
        l1_u_change = []
        l1_u_change_time = []
        frac_rate = []
        for d in dir:
            # get important data
            path_pos = f"{d}/simulation_data/robot_pos.npy"
            path_u_cbf = f"{d}/simulation_data/u_cbf.npy"
            path_u_nom = f"{d}/simulation_data/u_nominal.npy"
            robot_pos = np.load(path_pos)
            u_cbf = np.load(path_u_cbf)
            u_nom = np.load(path_u_nom)
            # calculate w_min
            if i <= 1:
                x_max = np.amax(robot_pos[:, 0])
                w_min = calculate_w_min_2d(
                    close_circles[i]["x1"],
                    close_circles[i]["y1"],
                    close_circles[i]["r1"],
                    close_circles[i]["x2"],
                    close_circles[i]["y2"],
                    close_circles[i]["r2"],
                    x_max,
                    0.0,
                    0.0,
                    1.0
                )
                w_mins.append(w_min)
            else:
                w_mins.append(1.75)
            
            if i <= 1:
                # calculate u_cbf summaries
                deadlock_idx = detect_deadlock(
                    robot_pos,
                    0.02,
                    # u_cbf,
                    v_thresh=0.05,
                )
                # print(f"Deadlock @ {deadlock_idx[1]}: {robot_pos[deadlock_idx[1]]}")
                # get reports
                u_change = analyze_cbf_interruptions(
                    u_nom[:deadlock_idx[1]+1],
                    u_cbf[:deadlock_idx[1]+1],
                    abs_tol=0.01,
                    rel_tol=0.0
                )
                l1_u_change.append(u_change.total_area_L1)
                l1_u_change_time.append(u_change.total_area_L1/u_change.n_interuptions)
                frac_rate.append(u_change.fraction_time_interrupted)
            else:
                u_change = analyze_cbf_interruptions(
                    u_nom,
                    u_cbf,
                    abs_tol=0.01,
                    rel_tol=0.0
                )
                l1_u_change.append(u_change.total_area_L1)
                frac_rate.append(u_change.fraction_time_interrupted)
                l1_u_change_time.append(u_change.total_area_L1/u_change.n_interuptions)
        
        # print wmin
        print(f"Robot {i}:")
        print(f"Avg c_eff: {np.round(np.mean(np.array(w_mins))-d_robot, 2)}, {np.array(w_mins)-d_robot}")
        print(f"Avg ||u||: {np.round(np.mean(np.array(l1_u_change)), 2)}, {l1_u_change}")
        print(f"Avg ||u||/T: {np.round(np.mean(np.array(l1_u_change_time)), 2)}, {l1_u_change_time}")
        print(f"Avg frac_rate: {np.round(np.mean(np.array(frac_rate)), 3)}, {frac_rate}")

def cluttered_env():
    # function for gap env metrics
    d_robot = (2)**0.5
    print("- - - Cluttered environment - - -")
    # important parameters
    home_dir = "./runs/E1_overall_performance/cluttered_env/simulation_results"
    dirs = [
        [
            f"{home_dir}/cluttered_experiment_0_seed_7",
            f"{home_dir}/cluttered_experiment_0_seed_15",
            f"{home_dir}/cluttered_experiment_0_seed_22",
        ],
        [
            f"{home_dir}/cluttered_experiment_1_seed_7",
            f"{home_dir}/cluttered_experiment_1_seed_15",
            f"{home_dir}/cluttered_experiment_1_seed_22",
        ],
        [
            f"{home_dir}/cluttered_experiment_3_seed_7",
            f"{home_dir}/cluttered_experiment_3_seed_15",
            f"{home_dir}/cluttered_experiment_3_seed_22",
        ],
    ]
    close_circles = [
        {
            "x1": 7,
            "y1": -7,
            "r1": 2,
            "x2": 5,
            "y2": -1,
            "r2": 2,
        },  # robot 0
        {
            "x1": 6,
            "y1": 6,
            "r1": 3,
            "x2": -0.75,
            "y2": 6,
            "r2": 2,
        },  # robot 1
    ]

    for i, dir in enumerate(dirs):
        w_mins = []
        l1_u_change = []
        l1_u_change_time = []
        frac_rate = []
        for d in dir:
            # get important data
            path_pos = f"{d}/simulation_data/robot_pos.npy"
            path_u_cbf = f"{d}/simulation_data/u_cbf.npy"
            path_u_nom = f"{d}/simulation_data/u_nominal.npy"
            robot_pos = np.load(path_pos)
            u_cbf = np.load(path_u_cbf)
            u_nom = np.load(path_u_nom)
            # calculate everything
            if i <= 1:
                # calculate u_cbf summaries
                deadlock_idx = detect_deadlock(
                    robot_pos,
                    0.02,
                    # u_cbf,
                    v_thresh=0.07,
                )
                print(deadlock_idx)
                # print(f"Deadlock @ {deadlock_idx[1]}: {robot_pos[deadlock_idx[1]]}")
                # get reports
                u_change = analyze_cbf_interruptions(
                    u_nom[:deadlock_idx[1]+1],
                    u_cbf[:deadlock_idx[1]+1],
                    abs_tol=0.01,
                    rel_tol=0.0
                )
                l1_u_change.append(u_change.total_area_L1)
                l1_u_change_time.append(u_change.total_area_L1/u_change.n_interuptions)
                frac_rate.append(u_change.fraction_time_interrupted)

                # calculate w_min
                final_pos = robot_pos[deadlock_idx[1]]
                w_min = calculate_w_min_2d(
                    close_circles[i]["x1"],
                    close_circles[i]["y1"],
                    close_circles[i]["r1"],
                    close_circles[i]["x2"],
                    close_circles[i]["y2"],
                    close_circles[i]["r2"],
                    final_pos[0],
                    final_pos[1],
                    close_circles[i]["x2"]-close_circles[i]["x1"],
                    close_circles[i]["y2"]-close_circles[i]["y1"]
                )
                w_mins.append(w_min)
            else:
                # u_cbf summaries
                u_change = analyze_cbf_interruptions(
                    u_nom,
                    u_cbf,
                    abs_tol=0.01,
                    rel_tol=0.0
                )
                l1_u_change.append(u_change.total_area_L1)
                frac_rate.append(u_change.fraction_time_interrupted)
                l1_u_change_time.append(u_change.total_area_L1/u_change.n_interuptions)

                # w_min
                w_mins.append(1.75)
        
        # print wmin
        print(f"Robot {i}:")
        print(f"Avg c_eff: {np.round(np.mean(np.array(w_mins))-d_robot, 2)}, {np.array(w_mins)-d_robot}")
        print(f"Avg ||u||: {np.round(np.mean(np.array(l1_u_change)), 2)}, {l1_u_change}")
        print(f"Avg ||u||/T: {np.round(np.mean(np.array(l1_u_change_time)), 2)}, {l1_u_change_time}")
        print(f"Avg frac_rate: {np.round(np.mean(np.array(frac_rate)), 3)}, {frac_rate}")
    

# run envs
gap_env()
cluttered_env()