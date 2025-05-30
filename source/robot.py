from perception import Perception
from planners import AStarPlanner, CBFInfusedAStar
from cbf_costmap import CBFCostmap
from robot_cbf_config import RobotCBFConfig
from visualization import VisualizeSimulation
from loguru import logger
from cbfpy import CBF
import numpy as np


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
        cbf_state_uncertainty_mode: str,
        control_fps: float,
        state_estimation_fps: float,
        goal_tolerance: float = 0.1,
        Kp: float = 0.5,
        Kd: float = 0.1,
        u_min_max: np.ndarray = np.array([-1000, 1000]),
        initial_state: np.ndarray = np.zeros(4),
        sensors: list = None,
        obstacles: list = None,
    ):
        # this class represents the robot
        # create cbf object
        self.cbf_config = RobotCBFConfig(
            obstacles=obstacles,
        )
        self.cbf = CBF.from_config(self.cbf_config)

        # create perception module
        # we create the perception module with given sensors (not with random generation)
        self.perception = Perception(
            costmap_size=costmap_size,
            grid_size=grid_size,
            cbf=self.cbf,
            min_values_state=min_values_state,
            max_values_state=max_values_state,
            min_sensor_noise=min_sensor_noise,
            max_sensor_noise=max_sensor_noise,
            magnitude_threshold=magnitude_threshold,
            sensors=sensors,
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
            )
        elif planner_mode == "CBF infused A*":
            self.planner = CBFInfusedAStar(
                costmap_size=costmap_size,
                grid_size=grid_size,
                obstacles=obstacles,
                cbf_costmap=self._cbf_costmap,
                diagonal_movement=True,
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

        # control parameters
        self._u_min_max = u_min_max
        self._Kp = Kp
        self._Kd = Kd

        # fps and time
        self._control_fps = control_fps
        self._control_dt = 1 / control_fps
        self._state_estimation_fps = state_estimation_fps
        self._state_esimation_dt = 1 / state_estimation_fps

        # costmaps
        self.costmaps = self.get_costmaps()

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
        self.planner.compute_distance_map(start_pos)
        self.costmaps = self.get_costmaps()
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

    def pd_controller(self, target_pos: np.ndarray):
        # TODO: maybe change controller? Check for jitter behavior
        # subtract position and velocity from estimated state
        position = self.estimated_state[:2]
        velocity = self.estimated_state[2:]

        error = target_pos - position
        damping = -self._Kd * velocity  # Damping term to reduce overshoot
        u = self._Kp * error + damping
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
            "planner_costmap": self.planner.distance_map,
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

    #########################################################
    # MAIN METHODS
    #########################################################
    def control_update(self):
        # method to apply the control input to the system
        # check if there is a path
        if self._path is None:
            logger.warning("No path to follow. No control input applied.")
            return

        # get control input and apply the safety filter
        target_pos = self.get_intermediate_position()
        u_nominal = self.pd_controller(target_pos)
        noise = self.perception.get_perception_noise(self._true_state[2:])
        safety_margin = self.perception.calculate_safety_margin(
            noise=noise, u_nominal=u_nominal, mode=self._cbf_state_uncertainty_mode
        )
        u_cbf = self.cbf.safety_filter(self._estimated_state, u_nominal, safety_margin)

        # add data to visualizer
        h_true = self.cbf_config.alpha(
            self.cbf_config.h_1(
                self._true_state, np.zeros(self.cbf_config.num_obstacles)
            )
        )
        self.visualizer.data.h_true.append(np.array(h_true))
        h_estimated = self.cbf_config.alpha(
            self.cbf_config.h_1(self._estimated_state, safety_margin)
        )
        self.visualizer.data.h_estimated.append(np.array(h_estimated))
        # self.visualizer.data.robot_pos.append(self._true_state[:2])
        # self.visualizer.data.robot_vel.append(self._true_state[2:])
        self.visualizer.data.u_cbf.append(u_cbf)
        self.visualizer.data.u_nominal.append(u_nominal)
        self.visualizer.data.safety_margin.append(safety_margin)

        # update the state of the system
        self._true_state[2:] += u_cbf
        self._true_state[:2] += self._true_state[2:] * self._control_dt

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

    def run_simulation(self, sim_time: float, env_folder: str):
        if self.path is None:
            return None
        t_control = 0.0
        t_estimation = 0.0
        t = 0.0
        dt = min(self._control_dt, self._state_esimation_dt)

        # apply the loop
        # in general: if both are in the same loop -> first estimation then apply control
        while t < sim_time and not self.check_goal_reached():
            # check order
            if t_control < t_estimation and t >= t_control:
                self.control_update()
                t_control += self._control_dt

            # check if estimation needs to be updated
            if t >= t_estimation:
                self.state_estimation_update()
                t_estimation += self._state_esimation_dt

            # check if control needs to be updated
            if t >= t_control:
                self.control_update()
                t_control += self._control_dt

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
            logger.warning(f"Goal not reached after {t} seconds")
            return False

    def plot(self, filename: str):
        # create the plot
        self.visualizer.create_full_plot(planner=self.planner, filename=filename)
