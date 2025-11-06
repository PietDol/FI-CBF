from perception import Perception
from planners import AStarPlanner, CBFInfusedAStar
from cbf_costmap import CBFCostmap
from robot_cbf_config import RobotCBFConfig
from visualization import VisualizeSimulation
from confidence_manager import ConfidenceManager
from loguru import logger
from cbfpy import CBF
import numpy as np
import jax
import jax.numpy as jnp
import time
from collections import deque


class Robot:
    def __init__(
        self,
        costmap_size: np.ndarray,
        grid_size: float,
        planner_mode: str,
        width: float,
        height: float,
        min_values_state: np.ndarray,
        max_values_state: np.ndarray,
        min_sensor_noise: float,
        max_sensor_noise: float,
        magnitude_threshold: float,
        control_fps: float,
        state_estimation_fps: float,
        cbf_state_uncertainty_mode: str,
        cbf_switch_velocity_thres: float = None,
        cbf_switch_control_diff_thres: float = None,
        cbf_switch_nominal_control_mag: float = None,
        cbf_confidence_config: dict = None,
        cbf_percentile: float = None,
        noise_cost_gain: float = 0.0,
        goal_tolerance: float = 0.1,
        Kp: float = 0.5,
        Kd: float = 0.1,
        u_min_max: np.ndarray = np.array([-1000, 1000]),
        initial_state: np.ndarray = np.zeros(4),
        sensors: list = None,
        obstacles: list = None,
        env_folder: str = None,
    ):
        # this class represents the robot
        # create cbf object
        self.cbf_config = RobotCBFConfig(
            obstacles=obstacles,
        )
        self.cbf = CBF.from_config(self.cbf_config)

        # create the confidence manager
        self.confidence_manager = ConfidenceManager(cbf_confidence_config)

        # create perception module
        # we create the perception module with given sensors (not with random generation)
        self.perception = Perception(
            costmap_size=costmap_size,
            grid_size=grid_size,
            cbf=self.cbf,
            obstacles=obstacles,
            env_dir=env_folder,
            confidence_config=cbf_confidence_config,
            min_values_state=min_values_state,
            max_values_state=max_values_state,
            min_sensor_noise=min_sensor_noise,
            max_sensor_noise=max_sensor_noise,
            magnitude_threshold=magnitude_threshold,
            num_samples_per_dim=4,  # normally take 4
            sensors=sensors,
            # load_lipschitz_grid_path="./runs/experiment_fabric_success/simulation_results/fabric_experiment_0",
            # load_lipschitz_grid_path="./runs/experiment_fake_success/simulation_results/fake_experiment_0",
            # load_lipschitz_grid_path="./runs/experiment_cluttered_success/simulation_results/cluttered_experiment_0",
            # load_lipschitz_grid_path="./runs/gap_experiments_debug/simulation_results/gap_experiment_0_seed_7",
            # load_lipschitz_grid_path="./runs/experiments_debug/simulation_results/debug_experiment_3_seed_7",
            load_lipschitz_grid_path="./runs/gap_exp_new/simulation_results/robot_3/gap_experiment_3_seed_7",
        )

        # create cbf costmap
        self._cbf_costmap = CBFCostmap(
            costmap_size=costmap_size,
            grid_size=grid_size,
            cbf_config=self.cbf_config,
            cbf_reduction="min",
        )

        if planner_mode == "A*":
            self.planner = AStarPlanner(
                costmap_size=costmap_size,
                grid_size=grid_size,
                obstacles=obstacles,
                diagonal_movement=True,
                noise_costmap=self.perception.noise_costmap,
                noise_cost_gain=noise_cost_gain,
            )
        elif planner_mode == "CBF infused A*":
            self.planner = CBFInfusedAStar(
                costmap_size=costmap_size,
                grid_size=grid_size,
                obstacles=obstacles,
                cbf_costmap=self._cbf_costmap,
                noise_costmap=self.perception.noise_costmap,
                noise_cost_gain=noise_cost_gain,
            )

        # create the visualizer
        self.visualizer = VisualizeSimulation(
            pos_goal=None, obstacles=obstacles, show_plot=False
        )

        # some other handy attributes
        self._obstacles = obstacles
        self._width = width
        self._height = height
        self._initial_state = initial_state.copy()
        self._true_state = initial_state.copy()
        self._estimated_state = self.perception.get_estimated_state(
            true_state=self._true_state
        )
        self._path = None
        self._goal_position = None
        self._path_idx = 0
        self._goal_tolerance = goal_tolerance
        self._cbf_state_uncertainty_mode = cbf_state_uncertainty_mode
        self._cbf_switch_velocity_thres = cbf_switch_velocity_thres
        self._cbf_switch_control_diff_thres = cbf_switch_control_diff_thres
        self._cbf_switch_nominal_control_mag = cbf_switch_nominal_control_mag
        self._cbf_percentile = np.round(cbf_percentile, 1)  # round for dict key
        self._env_folder = env_folder
        self._k = cbf_confidence_config["k"]

        # control parameters
        self._u_min_max = u_min_max
        self._Kp = Kp
        self._Kd = Kd

        # fps and time
        self._control_fps = control_fps
        self._control_dt = 1 / control_fps
        self._state_estimation_fps = state_estimation_fps
        self._state_esimation_dt = 1 / state_estimation_fps
        self._t_control = 0.0
        self._t_estimation = 0.0

        # costmaps
        self._work_domain = np.array(
            [
                [-costmap_size[0] / 2, costmap_size[0] / 2],  # x min max
                [-costmap_size[1] / 2, costmap_size[1] / 2],  # y min max
            ]
        )
        self.costmaps = self.get_costmaps()

        # cbf switch mechanism
        self._prev_dist_to_goal = -1
        self._timesteps_passed = 0
        self._lower_conf_timesteps_passed = 0
        self._normal_cooldown = 0
        self._cbf_state = "normal"
        self._relax_hold = 0
        self._active_percentile = self._cbf_percentile
        self._ramp_counter = 0
        self._calculate_grid_per_level = cbf_confidence_config[
            "calculate_grid_per_level"
        ]
        self._percentile_velocity_dict = {}
        for i in range(len(cbf_confidence_config["percentiles"])):
            self._percentile_velocity_dict[
                f"{cbf_confidence_config['percentiles'][i]}"
            ] = cbf_confidence_config["percentile_velocity"][i]

        # log
        logger.success("Robot created")

    #########################################################
    # PROPERTIES
    #########################################################
    # use property because you only want to change the attributes in the class
    @property
    def true_state(self):
        return self._true_state

    @property
    def estimated_state(self):
        return self._estimated_state

    @property
    def path(self):
        return self._path

    @property
    def goal_position(self):
        return self._goal_position

    #########################################################
    # HELPER METHODS
    #########################################################
    def reset(self):
        # reset the robot to the initial state
        # TODO: check what else should be reset e.g. the visualizer
        self._true_state = self._initial_state

    def plan(self, goal_pos: np.ndarray):
        # return true if path is found and false otherwise
        start_pos = self._initial_state[:2]
        path_output = self.planner.plan(start_coords=start_pos, goal_coords=goal_pos)

        # if no path is found
        if path_output is None:
            logger.error(f"No path found for given goal location: {goal_pos}")
            return False

        # set the path, goal location and planner costmap
        self._path = path_output["path_world"]
        self._goal_position = goal_pos
        self.visualizer.pos_goal = goal_pos
        self.visualizer.data.path = self._path
        self.planner.create_costmap(start_pos)
        self.costmaps = self.get_costmaps()
        logger.success("Path planned and costmaps created.")
        return True

    def get_intermediate_position(self):
        # tolerance is no taken as 1.5
        tolerance = 1.5

        # get the intermediate position
        current_pos = self.estimated_state[:2]
        # current_pos = self._true_state[2:]
        inter_pos = self._path[self._path_idx]
        distance = np.linalg.norm(current_pos - inter_pos)

        # logic to which position is returned
        if len(self._path) < 1:
            return self._goal_position

        if distance <= tolerance:
            self._path_idx = min([len(self._path) - 1, self._path_idx + 1])

        if self._path_idx == len(self.path) - 1:
            return self._goal_position
        else:
            return self._path[self._path_idx]

    def pd_controller(self, target_pos: np.ndarray, v_max: float = None):
        # if v_max is given,
        position = self.estimated_state[:2]
        velocity = self.estimated_state[2:]

        error = target_pos - position
        damping = -self._Kd * velocity
        u = self._Kp * error + damping

        # clip based predefined min and max set by the user
        return np.clip(u, self._u_min_max[0], self._u_min_max[1])

    def check_goal_reached(self):
        # function to check if the goal is reached
        # use estimated state because in reality robot also only knows it estimate
        position = self._estimated_state[:2]
        distance = np.linalg.norm(position - self._goal_position)
        return distance <= self._goal_tolerance

    def get_costmaps(self):
        # function to return all the costmaps in one dict
        # also add the costmaps to the visualizer
        costmaps = {
            "perception_magnitude_costmap": self.perception.perception_magnitude_costmap,
            "noise_costmap": self.perception.noise_costmap,
            "planner_costmap": self.planner.costmap,
            "cbf_costmap": self._cbf_costmap.costmap,
        }

        # add costmaps to visualizer
        self.visualizer.data.perception_magnitude_costmap = costmaps[
            "perception_magnitude_costmap"
        ]
        self.visualizer.data.noise_costmap = costmaps["noise_costmap"]
        self.visualizer.data.planner_costmap = costmaps["planner_costmap"]
        self.visualizer.data.cbf_costmap = costmaps["cbf_costmap"]
        return costmaps

    # def calculate_safety_filter_constraints(
    #     self,
    #     v_max: float,
    #     noise: float,
    #     steps_ahead: float = 1.0,
    #     # work_domain: np.ndarray = np.array([[-10, 10], [-10, 10]]),
    # ):
    #     # function to calculate the reachable set of the robot and the constraint matrices for the QP
    #     # take 99.7% confidence interval (3 sigma around)
    #     x_min = (
    #         self._estimated_state[0]
    #         - 3 * noise
    #         - steps_ahead * v_max * self._control_dt
    #     )
    #     x_max = (
    #         self._estimated_state[0]
    #         + 3 * noise
    #         + steps_ahead * v_max * self._control_dt
    #     )
    #     y_min = (
    #         self._estimated_state[1]
    #         - 3 * noise
    #         - steps_ahead * v_max * self._control_dt
    #     )
    #     y_max = (
    #         self._estimated_state[1]
    #         + 3 * noise
    #         + steps_ahead * v_max * self._control_dt
    #     )

    #     # make sure robot stays within the working env
    #     x_min = max(self._work_domain[0, 0], x_min)
    #     x_max = min(self._work_domain[0, 1], x_max)
    #     y_min = max(self._work_domain[1, 0], y_min)
    #     y_max = min(self._work_domain[1, 1], y_max)

    #     # create the matrices for that: Gu <= h (https://github.com/kevin-tracy/qpax)
    #     G = jnp.array(
    #         [
    #             # stay within working domain
    #             [-steps_ahead * self._control_dt, 0],  # x_min
    #             [steps_ahead * self._control_dt, 0],  # x_max
    #             [0, -steps_ahead * self._control_dt],  # y_min
    #             [0, steps_ahead * self._control_dt],  # y_max
    #             # v < v_max
    #             [-1, 0],  # > -v_max
    #             [1, 0],  # < v_max
    #             [0, -1],  # > -v_max
    #             [0, 1],  # < v_max
    #             # stay within reachable set
    #             [-steps_ahead * self._control_dt, 0],  # x_min
    #             [steps_ahead * self._control_dt, 0],  # x_max
    #             [0, -steps_ahead * self._control_dt],  # y_min
    #             [0, steps_ahead * self._control_dt],  # y_max
    #         ]
    #     )
    #     h = jnp.array(
    #         [
    #             # stay within working domain
    #             -self._work_domain[0, 0]
    #             + self._estimated_state[0]
    #             + steps_ahead * self._estimated_state[2] * self._control_dt,  # x_min
    #             self._work_domain[0, 1]
    #             - self._estimated_state[0]
    #             - steps_ahead * self._estimated_state[2] * self._control_dt,  # x_max
    #             -self._work_domain[1, 0]
    #             + self._estimated_state[1]
    #             + steps_ahead * self._estimated_state[3] * self._control_dt,  # y_min
    #             self._work_domain[1, 1]
    #             - self._estimated_state[1]
    #             - steps_ahead * self._estimated_state[3] * self._control_dt,  # y_max
    #             # v < v_max
    #             v_max + self._estimated_state[2],  # > -v_max
    #             v_max - self._estimated_state[2],  # < v_max
    #             v_max + self._estimated_state[3],  # > -v_max
    #             v_max - self._estimated_state[3],  # < v_max
    #             # stay within reachable set
    #             steps_ahead * (v_max + self._estimated_state[2]) * self._control_dt
    #             + 3 * noise,  # x_min
    #             steps_ahead * (v_max - self._estimated_state[2]) * self._control_dt
    #             + 3 * noise,  # x_max
    #             steps_ahead * (v_max + self._estimated_state[3]) * self._control_dt
    #             + 3 * noise,  # y_min
    #             steps_ahead * (v_max - self._estimated_state[3]) * self._control_dt
    #             + 3 * noise,  # y_max
    #         ]
    #     )
    #     return np.array([[x_min, x_max], [y_min, y_max]]), G, h

    def calculate_safety_filter_constraints(
        self,
        v_max: float,
        noise: float,
        steps_ahead: float = 1.0,
    ):
        dt = self._control_dt * steps_ahead
        x_hat = self._estimated_state[0]
        y_hat = self._estimated_state[1]

        pad = 3.0 * noise + steps_ahead * v_max * dt
        x_min = x_hat - pad
        x_max = x_hat + pad
        y_min = y_hat - pad
        y_max = y_hat + pad

        # extra work-domain rows
        xwd_min, xwd_max = self._work_domain[0]
        ywd_min, ywd_max = self._work_domain[1]

        G = jnp.vstack(
            [
                # work-domain
                [-dt, 0],
                [dt, 0],
                [0, -dt],
                [0, dt],
                # |u| <= v_max
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1],
                # reachable set
                [-dt, 0],
                [dt, 0],
                [0, -dt],
                [0, dt],
            ]
        )
        h = jnp.hstack(
            [
                x_hat - xwd_min,
                xwd_max - x_hat,
                y_hat - ywd_min,
                ywd_max - y_hat,
                v_max,
                v_max,
                v_max,
                v_max,
                x_hat - x_min,
                x_max - x_hat,
                y_hat - y_min,
                y_max - y_hat,
            ]
        )

        return np.array([[x_min, x_max], [y_min, y_max]]), G, h

    def get_cbf_percentile(
        self,
        experiment_mode: int,
        *,
        # --- config (same defaults you posted) ---
        T: int = 50,  # samples in the window (~1 s @50Hz)
        HOLD: int = 50,  # min frames to stay RELAXED before exit
        COOLDOWN: int = 50,  # min frames to stay NORMAL before re-enter
        ETA_ENTER: float = 0.02,  # m progress over T to ENTER RELAXED
        ETA_EXIT: float = 0.05,  # m progress over T to EXIT RELAXED
        H_ENTER: float = 0.02,  # m, near boundary to ENTER
        H_EXIT: float = 0.06,  # m, safely away to EXIT  (> H_ENTER)
        PCT_FLOOR: float = 60.0,
        PCT_STEP: float = 10.0,
        RAMP_EVERY: int = 25,  # frames between ramp steps (~0.5 s)
        RELAX_REEVAL_EVERY: int = 50,  # frames between further relax checks (~1 s)
    ):
        # mechanism to prevent deadlocks be decreasing the percentile for the Lipschitz constants
        # return the percentile and the corresponding maximum velocity
        # for experiment 0 and 1 robust safety -> 100%
        # for experiment 2 we decide to take 90% globally
        if experiment_mode <= 1:
            return 100.0, self._percentile_velocity_dict["100.0"]
        elif experiment_mode == 2:
            return 90.0, self._percentile_velocity_dict["90.0"]

        # experiment mode 3
        # -------- one-time state init --------
        if not hasattr(self, "_dist_hist"):
            self._dist_hist = deque(maxlen=T + 1)
        if not hasattr(self, "_h_hist"):
            self._h_hist = deque(maxlen=T)

        # get last estimated h value and the distance to the goal
        first_run_done = True
        try:
            h_now = float(
                np.min(self.visualizer.data.h_estimated[-1])
            ) 
            dist_to_goal = float(
                np.linalg.norm(self._estimated_state[:2] - self._goal_position)
            )
        except:
            first_run_done = False

        # if it crashes return 100.0 percentile and corresponding maximum velocity
        if not first_run_done:
            return 100.0, self._percentile_velocity_dict["100.0"]

        # Keep nominal up to date (confidence can change over time)
        self._cbf_percentile = self._cbf_percentile

        # -------- update rolling windows --------
        self._dist_hist.append(float(dist_to_goal))
        self._h_hist.append(float(h_now))

        # Default values in case window not yet full
        progress_window = 0.0
        h_min = h_now

        # -------- state machine only after warm-up --------
        if len(self._dist_hist) == self._dist_hist.maxlen:
            progress_window = (
                self._dist_hist[0] - self._dist_hist[-1]
            )  # POSITIVE = progress
            h_min = min(self._h_hist) if len(self._h_hist) else 1e9

            if self._cbf_state == "normal":
                # Cooldown to avoid immediate re-entry into RELAXED
                if self._normal_cooldown > 0:
                    self._normal_cooldown -= 1
                else:
                    # Enter RELAXED if little progress and hugging the boundary
                    enter = (progress_window < ETA_ENTER) and (h_min < H_ENTER)
                    if enter:
                        self._cbf_state = "relaxed"
                        self._relax_hold = 0
                        # Step down once on entry (or keep it low if already below)
                        self._active_percentile = max(
                            self._active_percentile - PCT_STEP, PCT_FLOOR
                        )
                        self._ramp_counter = 0
                        logger.debug(
                            f"Switch to relaxed mode @ t={self._t_control:.2f}: percentile -> {self._active_percentile}"
                        )

            else:  # RELAXED
                self._relax_hold += 1

                # Consider further relaxation every RELAX_REEVAL_EVERY frames
                still_stuck = (progress_window < ETA_ENTER) and (h_min < H_ENTER)
                time_to_reeval = (self._relax_hold % RELAX_REEVAL_EVERY) == 0
                can_step_down = (self._active_percentile - PCT_STEP) >= PCT_FLOOR
                if time_to_reeval and still_stuck and can_step_down:
                    self._active_percentile = max(
                        self._active_percentile - PCT_STEP, PCT_FLOOR
                    )
                    logger.debug(
                        f"Further relax @ t={self._t_control:.2f}: percentile -> {self._active_percentile}"
                    )

                # Exit when held long enough AND we’re away from the boundary AND making progress
                exit_ok = (h_min > H_EXIT) and (progress_window >= ETA_EXIT)
                if (self._relax_hold >= HOLD) and exit_ok:
                    self._cbf_state = "normal"
                    self._normal_cooldown = COOLDOWN
                    # Do NOT snap percentile back; ramp in NORMAL
                    logger.debug(
                        f"Switch to normal mode @ t={self._t_control:.2f}: percentile -> {self._active_percentile}"
                    )

        # -------- choose percentile for this tick --------
        if self._cbf_state == "normal":
            # Gradually ramp toward nominal only when progress is healthy
            if len(self._dist_hist) == self._dist_hist.maxlen:
                if (
                    progress_window >= ETA_EXIT
                    and self._active_percentile < self._cbf_percentile
                ):
                    self._ramp_counter += 1
                    if self._ramp_counter >= RAMP_EVERY:
                        self._active_percentile = min(
                            self._active_percentile + PCT_STEP, self._cbf_percentile
                        )
                        self._ramp_counter = 0
                        logger.debug(
                            f"Ramp up percentile @ t={self._t_control:.2f}: -> {self._active_percentile}"
                        )

            cbf_percentile = self._active_percentile
        else:
            cbf_percentile = self._active_percentile  # low value while RELAXED

        return (
            cbf_percentile,
            self._percentile_velocity_dict[f"{np.round(self._active_percentile, 1)}"],
        )

    #########################################################
    # MAIN METHODS
    #########################################################
    def control_update(self, experiment_mode: int):
        # method to apply the control input to the system
        # check if there is a path
        if self._path is None:
            logger.warning("No path to follow. No control input applied.")
            return

        # get control input and apply the safety filter
        target_pos = self.get_intermediate_position()
        noise_true = self.perception.get_perception_noise(x_true=self._true_state[:2])
        noise = self.perception.get_noise_upper_bound_in_ball(x_hat=self._estimated_state[:2])
        if noise_true > noise:
            logger.debug(f"Optimistic noise used: {noise} ->  true noise {noise_true}")

        # implementation of confidence manager
        conf_level, conf_velocity = self.confidence_manager.get_confidence_info(noise)

        # get the cbf percentile
        cbf_percentile, percentile_velocity = self.get_cbf_percentile(
            experiment_mode=experiment_mode, 
        )

        # set the maximum velocity as the minimum of the two mechanism
        v_max = min(conf_velocity, percentile_velocity)

        # set confidence level to use for Lipschitz grid
        if self._calculate_grid_per_level:
            _conf_level = conf_level
        else:
            _conf_level = 1

        # calculate the reachable set
        reachable_set, G_constraints, h_constraints = (
            self.calculate_safety_filter_constraints(
                v_max=v_max, noise=noise, steps_ahead=2.0
            )
        )

        # calculate the nominal control
        u_nominal = self.pd_controller(target_pos, v_max)

        # calculate the safety margins based on the experiment mode
        safety_margin, L_Lfh, L_Lgh, G, h = self.perception.calculate_safety_margin(
            experiment_mode=experiment_mode,
            noise=noise,
            u_nominal=u_nominal,
            k=self._k,
            reachable_set=reachable_set,
            confidence_level=_conf_level,
            percentile=cbf_percentile,  # for now we take 80% percentile
            G=G_constraints,
            h=h_constraints,
        )

        # apply safety filter to the control input
        # new version
        u_cbf, h_est, h_true, Lfh_est, Lfh_true, Lgh_est, Lgh_true, t_qp = (
            self.cbf.safety_filter(
                self._estimated_state,
                u_nominal,
                safety_margin,
                self._true_state,
                G,
                h,
            )
        )

        # add all the data
        self.visualizer.data.Lfh_est.append(Lfh_est)
        self.visualizer.data.Lgh_est.append(Lgh_est)
        self.visualizer.data.Lfh_true.append(Lfh_true)
        self.visualizer.data.Lgh_true.append(Lgh_true)
        self.visualizer.data.L_Lfh.append(L_Lfh)
        self.visualizer.data.L_Lgh.append(L_Lgh)
        self.visualizer.data.t_qp.append(np.amax(np.array(t_qp)))
        self.visualizer.data.h_true.append(np.array(h_true))
        self.visualizer.data.h_estimated.append(np.array(h_est))
        self.visualizer.data.u_cbf.append(u_cbf)
        self.visualizer.data.u_nominal.append(u_nominal)
        self.visualizer.data.safety_margin.append(safety_margin)
        self.visualizer.data.noise.append(noise)
        self.visualizer.data.noise_true.append(noise_true)
        self.visualizer.data.v_max.append(v_max)
        self.visualizer.data.k.append(self._k)
        self.visualizer.data.conf_level.append(conf_level)
        self.visualizer.data.percentile_level.append(cbf_percentile)

        # update the state of the system
        # self._true_state[2:] += u_cbf
        # self._true_state[:2] += self._true_state[2:] * self._control_dt
        self._true_state[:2] += u_cbf * self._control_dt
        self._true_state[2:] = u_cbf

    def state_estimation_update(self):
        # method to get the state estimation of the robot
        estimated_state = self.perception.get_estimated_state(
            true_state=self._true_state
        )
        self._estimated_state = estimated_state

        # update the visualizer
        self.visualizer.data.robot_pos_estimated.append(self._estimated_state[:2])
        self.visualizer.data.robot_pos.append(self._true_state[:2].copy())
        self.visualizer.data.robot_vel.append(self._true_state[2:].copy())

    def run_simulation(self, sim_time: float, env_folder: str, experiment_mode: int):
        if self.path is None:
            return None
        self._t_control = 0.0
        self._t_estimation = 0.0
        t = 0.0
        dt = min(self._control_dt, self._state_esimation_dt)

        # apply the loop
        # in general: if both are in the same loop -> first estimation then apply control
        logger.info("Simulation started...")
        while t < sim_time and not self.check_goal_reached():
            # check order
            if self._t_control < self._t_estimation and t >= self._t_control:
                self.control_update(experiment_mode)
                self._t_control += self._control_dt

            # check if estimation needs to be updated
            if t >= self._t_estimation:
                self.state_estimation_update()
                self._t_estimation += self._state_esimation_dt

            # check if control needs to be updated
            if t >= self._t_control:
                self.control_update(experiment_mode)
                self._t_control += self._control_dt

            # check for collision
            for i, obstacle in enumerate(self._obstacles):
                collision = obstacle.check_collision(
                    self._true_state[:2], self._width, self._height
                )
                if collision:
                    logger.error(
                        f"Collision between robot and obstacle (id={obstacle.id})! Robot true state: {self._true_state}"
                    )

            # update the time
            t += dt

        # save data
        # last information to visualizer
        self.visualizer.data.sensor_positions = self.perception.sensor_positions
        self.get_costmaps()
        self.visualizer.data.to_numpy()
        self.visualizer.data.control_time = (
            np.arange(self.visualizer.data.u_nominal.shape[0]) * self._control_dt
        )
        self.visualizer.data.state_estimation_time = (
            np.arange(self.visualizer.data.robot_pos_estimated.shape[0])
            * self._state_esimation_dt
        )
        self.visualizer.data.save_data(dir=f"{env_folder}/simulation_data")

        if self.check_goal_reached():
            logger.success(f"Goal reached in {t} seconds")
            return True
        else:
            distance = np.linalg.norm(self._goal_position - self._true_state[:2])
            logger.warning(
                f"Goal not reached after {t} seconds. Distance to goal: {distance} m"
            )
            return False

    def plot(self, filename: str):
        # create the plots
        self.visualizer.create_full_plot(planner=self.planner, filename=filename)
        self.visualizer.plot_lipschitz(
            f"{self._env_folder}/lipschitz_constants_time.png"
        )
