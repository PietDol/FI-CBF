# python file to random generate environments
from loguru import logger
from env_config import EnvConfig
from robot_base_env import RobotBaseEnv, pd_controller
from robot_base_config import RobotBaseCBFConfig
from robot_base import RobotBase
from cbf_costmap import CBFCostmap
from cbf_infused_a_star import CBFInfusedAStar
from planner_comparison import PlannerComparison
from a_star import AStarPlanner
from visualization import VisualizeSimulation
from cbfpy import CBF
import random
from obstacles import RectangleObstacle, CircleObstacle
import numpy as np
import os
import json
from perception import Perception, Sensor


class EnvGeneratorConfig:
    def __init__(
        self,
        number_of_simulations: int,
        environment_size: tuple,
        grid_size: float,
        max_obstacle_size: dict,
        max_duration_of_simulation: float,
        min_goal_distance: float,
        work_dir: str,
        use_safety_margin: bool,
        max_sensor_noise: float,
        cbf_state_uncertainty_mode: str,
        fps=60,
        robot_size=(1, 1),
        min_number_of_obstacles=1,
        max_number_of_obstacles=1,
        min_number_of_sensors=1,
        max_number_of_sensors=1,
        cbf_reduction="min",
        cbf_infused_a_star=False,
        planner_comparison=False,
    ):
        # general parameters
        self.number_of_simulations = number_of_simulations
        self.max_duration_of_simulation = max_duration_of_simulation  # in seconds
        self.min_goal_distance = min_goal_distance  # in m
        self.work_dir = work_dir
        self.fps = fps
        self.planner_comparison = planner_comparison  # bool

        # environment parameters (robot and obstacles)
        self.environment_size = environment_size  # in m
        self.robot_size = robot_size  # in m
        self.min_number_of_obstacles = min_number_of_obstacles
        self.max_number_of_obstacles = max_number_of_obstacles
        self.max_obstacle_size = max_obstacle_size  # in m

        # costmap parameters
        self.grid_size = grid_size  # in m

        # cbf parameters
        self.cbf_state_uncertainty_mode = (
            cbf_state_uncertainty_mode  # choose between robust or probabilistic
        )
        self.use_safety_margin = use_safety_margin  # bool whether to use safety margin
        self.cbf_reduction = cbf_reduction  # the reduction to get the cbf in one grid, options: ['min', 'mean', 'sum']
        self.cbf_infused_a_star = cbf_infused_a_star  # bool

        # perception parameters
        self.min_number_of_sensors = min_number_of_sensors
        self.max_number_of_sensors = max_number_of_sensors
        self.max_sensor_noise = max_sensor_noise  # in m

    def log_information(self):
        logger.info(f"Workdir: {self.work_dir}")
        logger.info(f"Environment size: {self.environment_size}")
        logger.info(f"FPS: {self.fps}")
        logger.info(f"Grid size: {self.grid_size} m")
        logger.info(f"Max duration for simulation: {self.max_duration_of_simulation} s")
        logger.info(f"Minimum distance to the goal: {self.min_goal_distance} m")
        logger.info(f"Robot size: {self.robot_size}")
        logger.info(f"Use safety margin: {self.use_safety_margin}")
        logger.info(f"CBF reduction mode: {self.cbf_reduction}")
        logger.info(f"CBF infused A*: {self.cbf_infused_a_star}")
        logger.info(f"CBF state uncertainty mode: {self.cbf_state_uncertainty_mode}")
        logger.info(f"Max  noise (standard deviation): {self.max_sensor_noise}")
        logger.info(f"Min number of sensors: {self.min_number_of_sensors}")
        logger.info(f"Max number of sensors: {self.max_number_of_sensors}")
        logger.info(f"Min number of obstascles: {self.min_number_of_obstacles}")
        logger.info(f"Max number of obstascles: {self.max_number_of_obstacles}")
        for key, val in self.max_obstacle_size.items():
            logger.info(f"Max obstacle size for {key}: {val} m")

    def save_to_file(self, path: str):
        # Convert all class attributes to a serializable dict
        config_dict = self.__dict__.copy()
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
        logger.success(f"Configuration saved to {path}")

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        logger.success(f"Configuration loade from {path}")
        return cls(**config_dict)


class EnvGenerator:
    # this class is able to do multiple runs behind each other and creates nice logs
    # and visualization for that
    def __init__(self, config: EnvGeneratorConfig):
        self.config = config
        self.x_range = self.config.environment_size[0] / 2  # + - range for x dimenstion
        self.y_range = self.config.environment_size[1] / 2  # + - range for y dimension

        # create the env_config (mainly needed for switching the x and y dimension)
        self.env_config = EnvConfig(
            pixels_per_meter=50 * np.array([1, -1]), screen_width=800, screen_height=800
        )
        self.pygame_screen = False  # we dont want the pygame screen to pops up
        self.work_dir_available = self._create_work_dir()

        # create workdir to save all the combined image (only if there is no comparison)
        if not self.config.planner_comparison:
            os.makedirs(
                f"{self.config.work_dir}/simulation_results/all_envs", exist_ok=True
            )

        # log all the important information
        self.config.log_information()

        # save the information
        self.config.save_to_file(f"{self.config.work_dir}/env_config.json")

    def _create_work_dir(self):
        # function to create work_dir
        try:
            os.makedirs(f"{self.config.work_dir}/simulation_results", exist_ok=False)
            logger.info("New simulation is started!")
            return True
        except FileExistsError:
            logger.error(f"Directory already exists: {self.config.work_dir}")
            logger.info("Try similar forlder name")
            for i in range(10):
                try:
                    new_work_dir = f"{self.config.work_dir}_{i+1}"
                    os.makedirs(f"{new_work_dir}/simulation_results", exist_ok=False)
                    self.config.work_dir = new_work_dir
                    logger.success(f"Similar work dir created: {self.config.work_dir}")
                    return True
                except FileExistsError:
                    continue

            return False

    def save_env_information(
        self, dir, robot: RobotBase, obstacles: list, sensors: list
    ):
        # Prepare dictionary with only JSON-serializable data
        env_dict = {
            "start_pos": robot.pos_center_start.tolist(),
            "start_vel": robot.vel_center_start.tolist(),
            "goal_pos": robot.pos_goal.tolist(),
            "obstacles": [],
            "sensors": [],
        }

        for obstacle in obstacles:
            if isinstance(obstacle, CircleObstacle):
                env_dict["obstacles"].append(
                    {
                        "type": "circle",
                        "center": obstacle.pos_center.tolist(),
                        "radius": obstacle.radius,
                    }
                )
            elif isinstance(obstacle, RectangleObstacle):
                env_dict["obstacles"].append(
                    {
                        "type": "rectangle",
                        "center": obstacle.pos_center.tolist(),
                        "height": obstacle.height,
                        "width": obstacle.width,
                    }
                )

        for sensor in sensors:
            env_dict["sensors"].append(
                {
                    "center": sensor.sensor_position.tolist(),
                    "max_distance": sensor.max_distance,
                }
            )

        # Save to JSON file
        json_path = f"{dir}/env_data.json"
        with open(json_path, "w") as f:
            json.dump(env_dict, f, indent=4)

        logger.success(f"Environment data saved: {json_path}")

    def load_env_information(self, json_path):
        with open(json_path, "r") as f:
            env_dict = json.load(f)

        # Convert robot state to NumPy arrays
        start_pos = np.array(env_dict["start_pos"])
        start_vel = np.array(env_dict["start_vel"])
        goal_pos = np.array(env_dict["goal_pos"])

        # Reconstruct obstacles
        obstacles = []
        for i, obs in enumerate(env_dict["obstacles"]):
            if obs["type"] == "circle":
                obstacles.append(
                    CircleObstacle(
                        pos_center=np.array(obs["center"]),
                        radius=obs["radius"],
                        env_config=self.env_config,
                        robot=None,
                        id=i,
                    )
                )
            elif obs["type"] == "rectangle":
                obstacles.append(
                    RectangleObstacle(
                        pos_center=np.array(obs["center"]),
                        height=obs["height"],
                        width=obs["width"],
                        env_config=self.env_config,
                        robot=None,
                        id=i,
                    )
                )

        # reconstruct sensors
        sensors = []
        for sensor in env_dict["sensors"]:
            sensors.append(
                Sensor(
                    sensor_position=np.array(sensor["center"]),
                    max_distance=sensor["max_distance"],
                )
            )

        logger.success(f"Environment data loaded: {json_path}")

        return start_pos, start_vel, goal_pos, obstacles, sensors

    def _generate_goal(self, robot_pos, max_tries=1000):
        cx, cy = robot_pos
        max_dist = 0

        for _ in range(max_tries):
            gx = np.round(random.uniform(-self.x_range, self.x_range), 2)
            gy = np.round(random.uniform(-self.y_range, self.y_range), 2)
            dist = np.linalg.norm([gx - cx, gy - cy])
            if dist >= self.config.min_goal_distance:
                return np.array([gx, gy])
            elif dist > max_dist:
                max_goal = np.array([gx, gy])
                max_dist = dist

        logger.error(
            f"Unable to find goal location after {max_tries} tries. Max goal used with distance {max_dist}: {(gx, gy)} "
        )
        return max_goal

    def _check_collision(self, robot, obstacles, generation=False):
        # checks if there is a collision between robot and obstacles
        obs_to_keep = []
        for i, obstacle in enumerate(obstacles):
            if generation:
                # only add small safety margin when generating the obstacles -> prevent robot to be to close to obstacle
                collision = obstacle.check_collision(robot, safety_margin=0.2)
            else:
                collision = obstacle.check_collision(robot)
            goal_feasible = obstacle.check_goal_position(robot)

            if generation:
                if collision:
                    logger.info(
                        f"Robot in collision with obstacle (id={obstacle.id}), remove obstacle."
                    )
                    obs_to_keep.append(False)
                elif not goal_feasible:
                    logger.info(
                        f"Goal position in collision with obstacle (id={obstacle.id}), remove obstacle."
                    )
                    obs_to_keep.append(False)
                else:
                    obs_to_keep.append(True)
            elif not generation and collision:
                logger.error(
                    f"Collision between robot and obstacle (id={obstacle.id})! Robot position: {robot.position}"
                )

        if generation:
            filtered_obstacles = [
                obstacle for obstacle, keep in zip(obstacles, obs_to_keep) if keep
            ]
            return filtered_obstacles

    def _generate_obstacles(self, robot: RobotBase, max_tries=1000):
        obstacles = []
        for i in range(max_tries):
            number_of_obstacles = random.randint(
                self.config.min_number_of_obstacles, self.config.max_number_of_obstacles
            )

            for i in range(number_of_obstacles):
                shape = random.choice(["circle", "rectangle"])
                cx = np.round(random.uniform(-self.x_range, self.x_range), 2)
                cy = np.round(random.uniform(-self.y_range, self.y_range), 2)
                pos_center = np.array([cx, cy])

                if shape == "circle":
                    radius = np.round(
                        random.uniform(1, self.config.max_obstacle_size["circle"]), 2
                    )
                    obstacle = CircleObstacle(
                        radius=radius,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=robot,
                        id=i,
                    )

                    logger.info(
                        f"Generated CircleObstacle (id={i}) - center: {pos_center}, radius: {radius}"
                    )
                elif shape == "rectangle":
                    width = np.round(
                        random.uniform(
                            1, self.config.max_obstacle_size["rectangle"][0]
                        ),
                        2,
                    )
                    height = np.round(
                        random.uniform(
                            1, self.config.max_obstacle_size["rectangle"][1]
                        ),
                        2,
                    )
                    obstacle = RectangleObstacle(
                        width=width,
                        height=height,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=robot,
                        id=i,
                    )
                    logger.info(
                        f"Generated RectangleObstacle (id={i}) - center: {pos_center}, width: {width}, height: {height}"
                    )

                obstacles.append(obstacle)

            # check if they are in collision
            obstacles = self._check_collision(robot, obstacles, generation=True)

            if len(obstacles) > 0:
                return obstacles

        logger.error(
            f"Not able to generate obstacles after {max_tries} tries. This iteration will be skipped."
        )
        return []

    def _generate_env_elements(self, loaded_env_dir: str = None):
        if loaded_env_dir is None:
            # generate the robot object
            robot_x = np.round(random.uniform(-self.x_range, self.x_range), 2)
            robot_y = np.round(random.uniform(-self.y_range, self.y_range), 2)
            pos_goal = self._generate_goal((robot_x, robot_y))

            robot = RobotBase(
                width=self.config.robot_size[0],
                height=self.config.robot_size[1],
                env_config=self.env_config,
                pos_goal=pos_goal,
                pos_center_start=np.array([robot_x, robot_y]),
            )
            logger.info(f"Robot start: {robot.position}, goal: {robot.pos_goal}")

            # generate the obstacles
            obstacles = self._generate_obstacles(robot)

            # set sensors to None to create sensors
            sensors = None
        else:
            start_pos, start_vel, pos_goal, obstacles, sensors = (
                self.load_env_information(loaded_env_dir)
            )
            robot = RobotBase(
                width=self.config.robot_size[0],
                height=self.config.robot_size[1],
                env_config=self.env_config,
                pos_goal=pos_goal,
                pos_center_start=start_pos,
                vel_center_start=start_vel,
            )

            # add robot to obstacles
            for obstacle in obstacles:
                obstacle.robot = robot

        # create config and cbf
        config = RobotBaseCBFConfig(obstacles, robot)
        cbf = CBF.from_config(config)

        cbf_costmap = CBFCostmap(
            costmap_size=self.config.environment_size,
            grid_size=self.config.grid_size,
            cbf=config,
            cbf_reduction=self.config.cbf_reduction,
        )

        # create perception module (costmap included)
        num_sensors = random.randint(
            self.config.min_number_of_sensors, self.config.max_number_of_sensors
        )
        perception = Perception(
            costmap_size=self.config.environment_size,
            grid_size=self.config.grid_size,
            cbf=cbf,
            num_sensors=num_sensors,
            min_values_state=np.array([-10, -10, -1.5, -1.5]),
            max_values_state=np.array([10, 10, 1.5, 1.5]),
            max_sensor_noise=self.config.max_sensor_noise,
            num_samples_per_dim=4,
            sensors=sensors,
        )

        # generate the planner
        planners = {}
        if self.config.cbf_infused_a_star:
            planner = CBFInfusedAStar(
                costmap_size=self.config.environment_size,
                grid_size=self.config.grid_size,
                obstacles=obstacles,
                cbf_costmap=cbf_costmap,
            )
            planners["CBF infused A*"] = planner
        else:
            planner = AStarPlanner(
                costmap_size=self.config.environment_size,
                grid_size=self.config.grid_size,
                obstacles=obstacles,
            )
            planners["A*"] = planner

        if self.config.planner_comparison:
            if isinstance(planner, CBFInfusedAStar):
                other_planner = AStarPlanner(
                    costmap_size=self.config.environment_size,
                    grid_size=self.config.grid_size,
                    obstacles=obstacles,
                )
                planners["A*"] = other_planner
            else:
                other_planner = CBFInfusedAStar(
                    costmap_size=self.config.environment_size,
                    grid_size=self.config.grid_size,
                    obstacles=obstacles,
                    cbf_costmap=cbf_costmap,
                )
                planners["CBF infused A*"] = other_planner

        # create environment
        env = RobotBaseEnv(
            env_config=self.env_config,
            robot_base=robot,
            obstacles=obstacles,
            pygame_screen=self.pygame_screen,
            fps=self.config.fps,
        )

        visualizers = {}
        for key in planners.keys():
            visualizer = VisualizeSimulation(
                pos_goal=robot.pos_goal, obstacles=obstacles, show_plot=False
            )
            visualizers[key] = visualizer

        return env, visualizers, config, cbf, planners, cbf_costmap, perception

    def _run_planner(
        self,
        env: RobotBaseEnv,
        visualizer: VisualizeSimulation,
        config: RobotBaseCBFConfig,
        cbf: CBF,
        planner: AStarPlanner | CBFInfusedAStar,
        cbf_costmap: CBFCostmap,
        perception: Perception,
        env_folder: str,
    ):
        max_timesteps = env.fps * self.config.max_duration_of_simulation
        timestep = 0

        # add path and costmaps to visualizer
        visualizer.data.path = env.robot_base.path
        visualizer.data.planner_costmap = planner.compute_distance_map(
            start=env.robot_base.position
        )
        visualizer.data.cbf_costmap = cbf_costmap.costmap
        visualizer.data.perception_magnitude_costmap = (
            perception.perception_magnitude_costmap
        )
        visualizer.data.noise_costmap = perception.noise_costmap
        visualizer.data.sensor_positions = perception.sensor_positions

        # run env
        while env.running and timestep < max_timesteps:
            current_state = env.get_state()
            estimated_state = perception.get_estimated_state(current_state)

            # calculate nominal control
            # nominal_control = pd_controller(pos_des, estimated_state[:2], estimated_state[2:])
            nominal_control = env.robot_base.velocity_pd_controller()
            if np.isnan(current_state).any():
                logger.warning(
                    f"[{timestep}]: Nan value in current_state: {current_state}"
                )
                break

            # calculate the safety margin
            if self.config.use_safety_margin:
                noise = perception.get_perception_noise(x_true=current_state[:2])
                safety_margin = perception.calculate_safety_margin(
                    noise=noise,
                    u_nominal=nominal_control,
                    mode=self.config.cbf_state_uncertainty_mode,
                )
            else:
                safety_margin = np.zeros(config.num_obstacles)

            # safe data for visualizer
            # for h take the true state to calculate h
            visualizer.data.timestep.append(timestep)
            h_true = config.h_1(
                current_state, np.zeros(config.num_obstacles)
            )  # don't add safety margin here (because state is true)
            h_true = config.alpha(h_true)
            visualizer.data.h_true.append(np.array(h_true))
            h_estimated = config.h_1(estimated_state, safety_margin)
            h_estimated = config.alpha(h_estimated)
            visualizer.data.h_estimated.append(np.array(h_estimated))
            visualizer.data.robot_pos.append(current_state[:2])
            visualizer.data.robot_pos_estimated.append(estimated_state[:2])
            visualizer.data.robot_vel.append(current_state[2:])

            # apply safety filter
            u = cbf.safety_filter(estimated_state, nominal_control, safety_margin)

            # safe control data for visualizer
            visualizer.data.u_cbf.append(u)
            visualizer.data.u_nominal.append(nominal_control)
            visualizer.data.safety_margin.append(safety_margin)

            # change environment
            env.apply_control(u)
            self._check_collision(
                robot=env.robot_base, obstacles=env.obstacles, generation=False
            )
            env.step()

            # increment timestep
            timestep += 1

        # check the maximum tolerance as an effect of the state estimation uncertainty
        if env.robot_base.check_goal_reached(
            tolerance=self.config.grid_size + self.config.max_sensor_noise + 0.01
        ):
            logger.success(f"Goal reached in {timestep} timesteps.")
            goal_reached = True
        else:
            logger.warning(
                f"Simulation ended without reaching the goal. Timesteps: {timestep}"
            )
            goal_reached = False

        # add last information for visualizer
        visualizer.data.timestep.append(timestep)
        visualizer.data.u_cbf.append(u)
        visualizer.data.u_nominal.append(nominal_control)
        visualizer.data.safety_margin.append(safety_margin)
        h_true = config.h_1(current_state, np.zeros(config.num_obstacles))
        h_true = config.alpha(h_true)
        visualizer.data.h_true.append(np.array(h_true))
        h_estimated = config.h_1(estimated_state, safety_margin)
        h_estimated = config.alpha(h_estimated)
        visualizer.data.h_estimated.append(np.array(h_estimated))
        visualizer.data.robot_pos.append(current_state[:2])
        visualizer.data.robot_pos_estimated.append(estimated_state[:2])
        visualizer.data.robot_vel.append(current_state[2:])

        # set planner filenames
        if isinstance(planner, CBFInfusedAStar):
            planner_filename = "cbf_infused_a_star"
        else:
            planner_filename = "a_star"

        # save the data
        visualizer.data.save_data(dir=f"{env_folder}/{planner_filename}")

        # generate drawings
        if goal_reached:
            filename = f"{env_folder}/{planner_filename}_success.png"
        else:
            filename = f"{env_folder}/{planner_filename}_fail.png"

        filename = [filename]
        if not self.config.planner_comparison:
            env_number = env_folder.split("_")[-1]
            if goal_reached:
                _filename = f"{self.config.work_dir}/simulation_results/all_envs/{env_number}_success.png"
            else:
                _filename = f"{self.config.work_dir}/simulation_results/all_envs/{env_number}_fail.png"

            filename.append(_filename)

        visualizer.create_full_plot(planner, filename)

        # return whether the goal is reached
        return goal_reached

    @logger.catch
    def _run_env(self, env_folder, loaded_env_dir=None):
        env, visualizers, config, cbf, planners, cbf_costmap, perception = (
            self._generate_env_elements(loaded_env_dir)
        )
        # perception.add_sensor(Sensor(sensor_position=np.array([6, -1])))
        success = {}

        # save the environment parameters
        self.save_env_information(
            dir=env_folder,
            robot=env.robot_base,
            obstacles=env.obstacles,
            sensors=perception.sensors,
        )

        for planner_name, planner in planners.items():
            logger.info(f"Planner: {planner_name}")

            # plan the path
            env.robot_base.reset()
            path = planner.plan(
                start_coords=env.robot_base.position,
                goal_coords=env.robot_base.pos_goal,
            )

            # if no path go to the next planner
            if path is None:
                logger.error(f"No path found! Skip this planner!")
                success[planner_name] = None
                continue

            # path found -> add the path, activate the environment and run the planner
            env.robot_base.add_path(path=planner.path_world)
            env.running = True
            goal_reached = self._run_planner(
                env=env,
                visualizer=visualizers[planner_name],
                config=config,
                cbf=cbf,
                planner=planner,
                cbf_costmap=cbf_costmap,
                perception=perception,
                env_folder=env_folder,
            )
            success[planner_name] = goal_reached

        if self.config.planner_comparison:
            planner_comparison = PlannerComparison(
                a_star_planner=planners["A*"],
                cbf_a_star_planner=planners["CBF infused A*"],
                visualizers=visualizers,
            )
            planner_comparison.plot_comparison(f"{env_folder}/planner_comparison.png")

        return success

    def run_env_from_file(self, env_file: str, env_folder: str):
        # run an environment from a file
        # create folder for this simulation
        env_folder = f"{self.config.work_dir}/simulation_results/{env_folder}"
        os.makedirs(env_folder, exist_ok=True)

        # run the env
        succeed = self._run_env(env_folder=env_folder, loaded_env_dir=env_file)

    def __call__(self):
        # check if work_dir is available
        if not self.work_dir_available:
            return

        results_per_planner = {}

        # iterate for all the different elements
        for i in range(self.config.number_of_simulations):
            logger.info(
                f"Start environment {(i+1)}/{self.config.number_of_simulations}"
            )

            # create folder for this simulation
            env_folder = f"{self.config.work_dir}/simulation_results/env_{i+1}"
            os.makedirs(env_folder, exist_ok=True)

            # run the env
            succeed = self._run_env(env_folder)

            if i == 0:
                for key in succeed.keys():
                    results_per_planner[key] = {
                        "reached": 0,
                        "not reached": 0,
                        "no path": 0,
                    }

            for key, reached in succeed.items():
                if reached and isinstance(reached, bool):
                    results_per_planner[key]["reached"] += 1
                elif not reached and isinstance(reached, bool):
                    results_per_planner[key]["not reached"] += 1
                elif reached is None:
                    results_per_planner[key]["no path"] += 1

        logger.success(
            f"Simulations done: {self.config.number_of_simulations} executed."
        )

        # log all the results
        for planner_name, results in results_per_planner.items():
            logger.info(f"Results for {planner_name} planner:")
            logger.info(f"Number goal reached: {results['reached']}")
            logger.info(f"Number goal not reached: {results['not reached']}")
            logger.info(f"No path found: {results['no path']}")


def main():
    directory = "./runs/debug_1"

    # config = EnvGeneratorConfig(
    #     number_of_simulations=1,
    #     fps=50,
    #     min_number_of_obstacles=5,
    #     max_number_of_obstacles=10,
    #     environment_size=(20, 20),
    #     grid_size=0.1,
    #     max_obstacle_size={
    #         'circle': 3.0,
    #         'rectangle': (3.0, 3.0)
    #     },
    #     max_duration_of_simulation=20,
    #     min_goal_distance=15,
    #     robot_size=(1, 1),
    #     use_safety_margin=True,
    #     cbf_reduction='min',
    #     work_dir=directory,
    #     cbf_infused_a_star=True,
    #     planner_comparison=False,
    #     max_number_of_sensors=5,
    #     max_sensor_noise=0.1
    # )
    config = EnvGeneratorConfig.from_file("./runs/debug/env_config.json")

    # create logger
    logger.add(f"{config.work_dir}/simulations.log", rotation="10 MB")

    # create the environment generator class
    envs = EnvGenerator(config=config)

    # apply same environment for debugging
    envs.run_env_from_file(
        env_file="./runs/debug/simulation_results/env_6/env_data.json",
        env_folder="loaded_env",
    )

    # apply the simulations
    # envs()


if __name__ == "__main__":
    main()
