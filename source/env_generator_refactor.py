# python file to random generate environments
from loguru import logger
from env_config import EnvConfig
import random
from obstacles import RectangleObstacle, CircleObstacle
import numpy as np
import os
import json
from perception import Sensor
from robot import Robot
from env_generator_config import EnvGeneratorConfig


class EnvGenerator:
    # this class is able to do multiple runs behind each other and creates nice logs
    # and visualization for that
    def __init__(self, config: EnvGeneratorConfig):
        self.config = config
        self.x_range = self.config.costmap_size[0] / 2  # + - range for x dimenstion
        self.y_range = self.config.costmap_size[1] / 2  # + - range for y dimension

        # create the env_config (mainly needed for switching the x and y dimension)
        self.env_config = EnvConfig(
            pixels_per_meter=50 * np.array([1, -1]), screen_width=800, screen_height=800
        )
        self.pygame_screen = False  # we dont want the pygame screen to pops up
        self.work_dir_available = self._create_work_dir()

        # create workdir to save all the combined image
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

    def save_env_information(self, dir, robot: Robot, obstacles: list, sensors: list):
        # Prepare dictionary with only JSON-serializable data
        env_dict = {
            "start_pos": robot.true_state[:2].tolist(),
            "start_vel": robot.true_state[2:].tolist(),
            "goal_pos": robot.goal_position.tolist(),
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
                        "robot_radius": obstacle.robot_radius,
                    }
                )
            elif isinstance(obstacle, RectangleObstacle):
                env_dict["obstacles"].append(
                    {
                        "type": "rectangle",
                        "center": obstacle.pos_center.tolist(),
                        "height": obstacle.height,
                        "width": obstacle.width,
                        "robot_radius": obstacle.robot_radius,
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
                        robot_radius=obs["robot_radius"],
                        env_config=self.env_config,
                        id=i,
                    )
                )
            elif obs["type"] == "rectangle":
                obstacles.append(
                    RectangleObstacle(
                        pos_center=np.array(obs["center"]),
                        height=obs["height"],
                        width=obs["width"],
                        robot_radius=obs["robot_radius"],
                        env_config=self.env_config,
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

    def _check_collision(
        self,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        robot_width: float,
        robot_height: float,
        obstacles: list,
        generation=False,
    ):
        # checks if there is a collision between robot and obstacles
        obs_to_keep = []
        for i, obstacle in enumerate(obstacles):
            if generation:
                # only add small safety margin when generating the obstacles -> prevent robot to be to close to obstacle
                collision = obstacle.check_collision(
                    robot_pos, robot_width, robot_height, safety_margin=0.2
                )
                # only check for goal location when generating
                goal_feasible = obstacle.check_goal_position(
                    goal_pos, robot_width, robot_height
                )
            else:
                collision = obstacle.check_collision(
                    robot_pos, robot_width, robot_height
                )

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
                    f"Collision between robot and obstacle (id={obstacle.id})! Robot position: {robot_pos}"
                )

        if generation:
            filtered_obstacles = [
                obstacle for obstacle, keep in zip(obstacles, obs_to_keep) if keep
            ]
            return filtered_obstacles

    def _generate_obstacles(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        robot_width: float,
        robot_height: float,
        max_tries=1000,
    ):
        obstacles = []
        robot_radius = np.linalg.norm(robot_width / 2 + robot_height / 2)
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
                        robot_radius=robot_radius,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=None,
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
                        robot_radius=robot_radius,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=None,
                        id=i,
                    )
                    logger.info(
                        f"Generated RectangleObstacle (id={i}) - center: {pos_center}, width: {width}, height: {height}"
                    )

                obstacles.append(obstacle)

            # check if they are in collision
            obstacles = self._check_collision(
                robot_pos=start_pos,
                goal_pos=goal_pos,
                robot_width=robot_width,
                robot_height=robot_height,
                obstacles=obstacles,
                generation=True,
            )

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
            start_pos = np.array([robot_x, robot_y])
            goal_pos = self._generate_goal(start_pos)

            # generate the obstacles
            obstacles = self._generate_obstacles(
                start_pos=start_pos,
                goal_pos=goal_pos,
                robot_width=self.config.robot_width,
                robot_height=self.config.robot_height,
            )

            # set sensors to None to create sensors
            num_sensors = random.randint(
                self.config.min_number_of_sensors, self.config.max_number_of_sensors
            )
            sensors = []
            for _ in range(num_sensors):
                sx = np.round(
                    random.uniform(
                        -self.config.costmap_size[0] / 2,
                        self.config.costmap_size[0] / 2,
                    ),
                    2,
                )
                sy = np.round(
                    random.uniform(
                        -self.config.costmap_size[1] / 2,
                        self.config.costmap_size[1] / 2,
                    ),
                    2,
                )
                sensors.append(Sensor(np.array([sx, sy])))
        else:
            start_pos, start_vel, goal_pos, obstacles, sensors = (
                self.load_env_information(loaded_env_dir)
            )

        # create the robot
        initial_state = np.array([start_pos, start_vel]).flatten()
        robot = Robot(
            costmap_size=self.config.costmap_size,
            grid_size=self.config.grid_size,
            planner_mode=self.config.planner_mode,
            width=self.config.robot_width,
            height=self.config.robot_height,
            min_values_state=self.config.min_values_state,
            max_values_state=self.config.max_values_state,
            max_sensor_noise=self.config.max_sensor_noise,
            cbf_state_uncertainty_mode=self.config.cbf_state_uncertainty_mode,
            control_fps=self.config.control_fps,
            state_estimation_fps=self.config.state_estimation_fps,
            goal_tolerance=self.config.goal_tolerance,
            Kp=self.config.Kp,
            Kd=self.config.Kd,
            u_min_max=self.config.u_min_max,
            initial_state=initial_state,
            sensors=sensors,
            obstacles=obstacles,
        )

        # add robot to obstacles
        for obstacle in obstacles:
            obstacle.robot = robot

        # plan the path
        robot.plan(goal_pos=goal_pos)

        return robot, obstacles, sensors

    @logger.catch
    def _run_env(self, env_folder, loaded_env_dir=None):
        robot, obstacles, sensors = self._generate_env_elements(loaded_env_dir)
        # perception.add_sensor(Sensor(sensor_position=np.array([6, -1])))

        # save the environment parameters
        self.save_env_information(
            dir=env_folder,
            robot=robot,
            obstacles=obstacles,
            sensors=sensors,
        )

        # run the simulation
        sim_output = robot.run_simulation(
            sim_time=self.config.max_duration_of_simulation,
            env_folder=env_folder
        )

        # create plot
        env_number = env_folder.split("_")[-1]
        # generate drawings
        if sim_output:
            filenames = [
                f"{env_folder}/simulation_success.png",
                f"{self.config.work_dir}/simulation_results/all_envs/{env_number}_success.png",
            ]
        else:
            filenames = [
                f"{env_folder}/simulation_fail.png",
                f"{self.config.work_dir}/simulation_results/all_envs/{env_number}_fail.png",
            ]

        robot.plot(filenames)
        return sim_output

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
    directory = "./runs/debug_refactor"

    # config = EnvGeneratorConfig(
    #     number_of_simulations=1,
    #     work_dir=directory,
    #     max_duration_of_simulation=20,
    #     min_goal_distance=15,
    #     min_number_of_obstacles=5,
    #     max_number_of_obstacles=10,
    #     max_obstacle_size={"circle": 3.0, "rectangle": [3.0, 3.0]},
    #     min_number_of_sensors=1,
    #     max_number_of_sensors=5,
    #     costmap_size=np.array([20, 20]),
    #     grid_size=0.1,
    #     planner_mode="CBF infused A*",
    #     robot_width=1.0,
    #     robot_height=1.0,
    #     min_values_state=np.array([-10, -10, -1.5, -1.5]),
    #     max_values_state=np.array([10, 10, 1.5, 1.5]),
    #     max_sensor_noise=0.1,
    #     cbf_state_uncertainty_mode="robust",
    #     control_fps=50,
    #     state_estimation_fps=50,
    #     goal_tolerance=0.1,
    #     Kp=0.5,
    #     Kd=0.1,
    #     u_min_max=np.array([-1000, 1000])
    # )
    config = EnvGeneratorConfig.from_file("./runs/baseline_hard/env_config.json")

    # create logger
    logger.add(f"{config.work_dir}/simulations.log", rotation="10 MB")

    # create the environment generator class
    envs = EnvGenerator(config=config)

    # apply same environment for debugging
    envs.run_env_from_file(
        env_file="./runs/baseline_hard/simulation_results/loaded_env/env_data.json",
        env_folder="loaded_env",
    )

    # apply the simulations
    # envs()


if __name__ == "__main__":
    main()
