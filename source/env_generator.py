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
from uncertainty_costmap import UncertaintyCostmap
from perception import Perception


class EnvGeneratorConfig:
    def __init__(self, 
                 number_of_simulations: int, 
                 max_number_of_obstacles: int,
                 environment_size: tuple,
                 grid_size: float,
                 max_obstacle_size: dict,
                 max_duration_of_simulation: float,
                 min_goal_distance: float,
                 work_dir: str, 
                 state_std: np.ndarray | float,
                 use_safety_margin: bool,
                 number_of_sensors: int,
                 max_sensor_noise: float,
                 fps=60,
                 robot_size=(1, 1),
                 min_number_of_obstacles=1,
                 cbf_reduction='min',
                 cbf_infused_a_star=False,
                 planner_comparison=False):
        self.number_of_simulations = number_of_simulations
        self.min_number_of_obstacles = min_number_of_obstacles
        self.max_number_of_obstacles = max_number_of_obstacles
        self.environment_size = environment_size                        # in m
        self.grid_size = grid_size                                      # in m
        self.max_obstacle_size = max_obstacle_size                      # in m
        self.max_duration_of_simulation = max_duration_of_simulation    # in seconds
        self.min_goal_distance = min_goal_distance      # in m
        self.work_dir = work_dir
        self.state_std = state_std          # in m
        self.robot_size = robot_size        # in m
        self.fps = fps
        self.use_safety_margin = use_safety_margin  # bool whether to use safety margin
        self.number_of_sensors = number_of_sensors  
        self.max_sensor_noise = max_sensor_noise    # in m
        self.cbf_reduction = cbf_reduction  # the reduction to get the cbf in one grid, options: ['min', 'mean', 'sum']
        self.cbf_infused_a_star = cbf_infused_a_star    # bool
        self.planner_comparison = planner_comparison    # bool

class EnvGenerator:
    # this class is able to do multiple runs behind each other and creates nice logs
    # and visualization for that
    def __init__(self, config: EnvGeneratorConfig):
        self.config = config
        self.x_range = self.config.environment_size[0] / 2  # + - range for x dimenstion
        self.y_range = self.config.environment_size[1] / 2  # + - range for y dimension

        # create the env_config (mainly needed for switching the x and y dimension)
        self.env_config = EnvConfig(
            pixels_per_meter=50 * np.array([1, -1]),
            screen_width=800,
            screen_height=800
        )
        self.pygame_screen = False  # we dont want the pygame screen to pops up
        self.work_dir_available = self._create_work_dir()

        # create workdir to save all the combined image (only if there is no comparison)
        if not self.config.planner_comparison:
            os.makedirs(f"{self.config.work_dir}/simulation_results/all_envs", exist_ok=True)

        # log all the important information
        self.log_information()

    def _create_work_dir(self):
        # function to create work_dir
        _foldername = self.config.work_dir.split("/")[-1]
        try:
            if _foldername == 'debug':
                os.makedirs(f"{self.config.work_dir}/simulation_results", exist_ok=True)
            else:
                os.makedirs(f"{self.config.work_dir}/simulation_results", exist_ok=False)
            logger.info("New simulation is started!")
            return True
        except FileExistsError:
            logger.error(f"Directory already exists: {self.config.work_dir}")
            return False
    
    def log_information(self):
        logger.info(f"Workdir: {self.config.work_dir}")
        logger.info(f"Environment size: {self.config.environment_size}")
        logger.info(f"FPS: {self.config.fps}")
        logger.info(f"Grid size: {self.config.grid_size} m")
        logger.info(f"Max duration for simulation: {self.config.max_duration_of_simulation} s")
        logger.info(f"Minimum distance to the goal: {self.config.min_goal_distance} m")
        logger.info(f"Robot size: {self.config.robot_size}")
        logger.info(f"Use safety margin: {self.config.use_safety_margin}")
        logger.info(f"State estimation std: {self.config.state_std}")
        logger.info(f"CBF reduction mode: {self.config.cbf_reduction}")
        logger.info(f"CBF infused A*: {self.config.cbf_infused_a_star}")
        logger.info(f"Max standard deviation: {self.config.max_sensor_noise}")
        logger.info(f"Number of sensors: {self.config.number_of_sensors}")
        logger.info(f"Min number of obstascles: {self.config.min_number_of_obstacles}")
        logger.info(f"Max number of obstascles: {self.config.max_number_of_obstacles}")
        for key, val in self.config.max_obstacle_size.items():
            logger.info(f"Max obstacle size for {key}: {val} m")
    
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

        logger.error(f"Unable to find goal location after {max_tries} tries. Max goal used with distance {max_dist}: {(gx, gy)} ")
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
                    logger.info(f"Robot in collision with obstacle (id={obstacle.id}), remove obstacle.")
                    obs_to_keep.append(False)
                elif not goal_feasible:
                    logger.info(f"Goal position in collision with obstacle (id={obstacle.id}), remove obstacle.")
                    obs_to_keep.append(False)
                else:
                    obs_to_keep.append(True)    
            elif not generation and collision:
                logger.error(f"Collision between robot and obstacle (id={obstacle.id})! Robot position: {robot.position}")

        if generation:
            filtered_obstacles = [obstacle for obstacle, keep in zip(obstacles, obs_to_keep) if keep]
            return filtered_obstacles
    
    def _generate_obstacles(self, robot: RobotBase, max_tries=1000):
        obstacles = []
        for i in range(max_tries):
            number_of_obstacles = random.randint(self.config.min_number_of_obstacles, self.config.max_number_of_obstacles)

            for i in range(number_of_obstacles):
                shape = random.choice(["circle", "rectangle"])
                cx = np.round(random.uniform(-self.x_range, self.x_range), 2)
                cy = np.round(random.uniform(-self.y_range, self.y_range), 2)
                pos_center = np.array([cx, cy])

                if shape == "circle":
                    radius = np.round(random.uniform(1, self.config.max_obstacle_size["circle"]), 2)
                    obstacle = CircleObstacle(
                        radius=radius,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=robot,
                        id=i
                    )

                    logger.info(f"Generated CircleObstacle (id={i}) - center: {pos_center}, radius: {radius}")
                elif shape == "rectangle":
                    width = np.round(random.uniform(1, self.config.max_obstacle_size["rectangle"][0]), 2)
                    height = np.round(random.uniform(1, self.config.max_obstacle_size["rectangle"][1]), 2)
                    obstacle = RectangleObstacle(
                        width=width,
                        height=height,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=robot,
                        id=i
                    )
                    logger.info(f"Generated RectangleObstacle (id={i}) - center: {pos_center}, width: {width}, height: {height}")
                
                obstacles.append(obstacle)
            
            # check if they are in collision
            obstacles = self._check_collision(robot, obstacles, generation=True)

            if len(obstacles) > 0:
                return obstacles
        
        logger.error(f"Not able to generate obstacles after {max_tries} tries. This iteration will be skipped.")
        return []        

    def _generate_env_elements(self):
        # generate the robot object
        robot_x = np.round(random.uniform(-self.x_range, self.x_range), 2)
        robot_y = np.round(random.uniform(-self.y_range, self.y_range), 2)
        pos_goal = self._generate_goal((robot_x, robot_y))

        robot = RobotBase(
            width=self.config.robot_size[0],
            height=self.config.robot_size[1],
            env_config=self.env_config,
            pos_goal=pos_goal,
            pos_center_start=np.array([robot_x, robot_y])
        )
        logger.info(f"Robot start: {robot.position}, goal: {robot.pos_goal}")

        # generate the obstacles
        obstacles = self._generate_obstacles(robot)

        # create config and cbf
        config = RobotBaseCBFConfig(obstacles, robot)
        cbf = CBF.from_config(config)
        
        cbf_costmap = CBFCostmap(
            costmap_size=self.config.environment_size,
            grid_size=self.config.grid_size,
            cbf=config,
            cbf_reduction=self.config.cbf_reduction
        )

        # create perception module 
        perception = Perception(
            costmap_size=self.config.environment_size,
            cbf=cbf,
            num_sensors=self.config.number_of_sensors,
            min_values_state=np.array([-10, -10, -1.5, -1.5]),
            max_values_state=np.array([10, 10, 1.5, 1.5]),
            max_sensor_noise=self.config.max_sensor_noise,
            num_samples_per_dim=4
        )

        # create uncertainty costmap
        uncertainty_costmap = UncertaintyCostmap(
            costmap_size=self.config.environment_size,
            grid_size=self.config.grid_size,
            perception=perception
        )

        # generate the planner
        planners = {}
        if self.config.cbf_infused_a_star:
            planner = CBFInfusedAStar(
                costmap_size=self.config.environment_size,
                grid_size=self.config.grid_size,
                obstacles=obstacles,
                cbf_costmap=cbf_costmap
            )
            planners["CBF infused A*"] = planner
        else:
            planner = AStarPlanner(
                costmap_size=self.config.environment_size,
                grid_size=self.config.grid_size,
                obstacles=obstacles
            )
            planners["A*"] = planner

        if self.config.planner_comparison:
            if isinstance(planner, CBFInfusedAStar):
                other_planner = AStarPlanner(
                    costmap_size=self.config.environment_size,
                    grid_size=self.config.grid_size,
                    obstacles=obstacles
                )
                planners["A*"] = other_planner
            else:
                other_planner = CBFInfusedAStar(
                    costmap_size=self.config.environment_size,
                    grid_size=self.config.grid_size,
                    obstacles=obstacles,
                    cbf_costmap=cbf_costmap
                )
                planners["CBF infused A*"] = other_planner
        
        # create environment
        env = RobotBaseEnv(
            env_config=self.env_config,
            robot_base=robot,
            obstacles=obstacles,
            pygame_screen=self.pygame_screen,
            fps=self.config.fps
        )

        visualizers = {}
        for key in planners.keys():
            visualizer = VisualizeSimulation(
                pos_goal=robot.pos_goal,
                obstacles=obstacles,
                show_plot=False
            )
            visualizers[key] = visualizer
        
        return env, visualizers, config, cbf, planners, cbf_costmap, perception, uncertainty_costmap
    
    def _run_planner(self, 
                     env: RobotBaseEnv, 
                     visualizer: VisualizeSimulation, 
                     config: RobotBaseCBFConfig, 
                     cbf: CBF, 
                     planner: AStarPlanner | CBFInfusedAStar, 
                     cbf_costmap: CBFCostmap, 
                     perception: Perception,
                     uncertainty_costmap: UncertaintyCostmap,
                     env_folder: str):
        max_timesteps = env.fps * self.config.max_duration_of_simulation
        timestep = 0

        # add path and costmaps to visualizer
        visualizer.data.path = env.robot_base.path
        visualizer.data.planner_costmap = planner.compute_distance_map(start=env.robot_base.position)
        visualizer.data.cbf_costmap = cbf_costmap.costmap
        visualizer.data.uncertainty_costmap = uncertainty_costmap.costmap
        visualizer.data.sensor_positions = perception.sensor_positions

        # run env
        while env.running and timestep < max_timesteps:
            current_state = env.get_state()
            estimated_state = env.get_estimated_state(std=self.config.state_std)
            pos_des = env.get_desired_state()
            
            # calculate nominal control
            # nominal_control = pd_controller(pos_des, estimated_state[:2], estimated_state[2:])
            nominal_control = env.robot_base.velocity_pd_controller()
            if np.isnan(current_state).any():
                logger.warning(f"[{timestep}]: Nan value in current_state: {current_state}")
                break
            
            # calculate the safety margin
            if self.config.use_safety_margin:
                safety_margin = perception.calculate_safety_margin(
                    epsilon=0.4,        # in the paper they use 0.4 (later this will be taken from the costmap)
                    u_nominal=nominal_control,
                    mode='robust'
                )
            else:
                safety_margin = np.zeros(config.num_obstacles)
            
            # safe data for visualizer  
            # for h take the true state to calculate h
            visualizer.data.timestep.append(timestep)
            h_true = config.h_1(current_state, np.zeros(config.num_obstacles))  # don't add safety margin here (because state is true)
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
            self._check_collision(robot=env.robot_base, obstacles=env.obstacles, generation=False)
            env.step()
            
            # increment timestep
            timestep += 1
        
        # check the maximum tolerance as an effect of the state estimation uncertainty
        if isinstance(self.config.state_std, float):
            max_tolerance_state_uncertainty = self.config.state_std
        elif isinstance(self.config.state_std, np.ndarray):
            max_tolerance_state_uncertainty = np.amax(self.config.state_std)
        
        if env.robot_base.check_goal_reached(tolerance=self.config.grid_size + max_tolerance_state_uncertainty + 0.01):
            logger.success(f"Goal reached in {timestep} timesteps.")
            goal_reached = True
        else:
            logger.warning(f"Simulation ended without reaching the goal. Timesteps: {timestep}")
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
        if isinstance(planner,  CBFInfusedAStar):
            planner_filename = 'cbf_infused_a_star'
        else:
            planner_filename = 'a_star'

        # save the data
        visualizer.data.save_data(dir=f"{env_folder}/{planner_filename}")

        # generate drawings
        if goal_reached:
            filename = f"{env_folder}/{planner_filename}_success.png"
        else:
            filename = f"{env_folder}/{planner_filename}_fail.png"
        
        filename = [filename]
        if not self.config.planner_comparison:
            env_number = env_folder.split('_')[-1]
            if goal_reached:
                _filename = f"{self.config.work_dir}/simulation_results/all_envs/{env_number}_success.png"
            else:
                _filename = f"{self.config.work_dir}/simulation_results/all_envs/{env_number}_fail.png"
            
            filename.append(_filename)

        visualizer.create_full_plot(planner, filename)

        # return whether the goal is reached
        return goal_reached

    @logger.catch
    def _run_env(self, env_id):
        env, visualizers, config, cbf, planners, cbf_costmap, perception, uncertainty_costmap = self._generate_env_elements()
        success = {}

        # create folder for this simulation
        env_folder = f"{self.config.work_dir}/simulation_results/env_{env_id}"
        os.makedirs(env_folder, exist_ok=True)

        # save the environment parameters
        env.save_env_information(dir=env_folder)

        for planner_name, planner in planners.items():
            logger.info(f"Planner: {planner_name}")

            # plan the path
            env.robot_base.reset()
            path = planner.plan(
                start_coords=env.robot_base.position,
                goal_coords=env.robot_base.pos_goal
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
                uncertainty_costmap=uncertainty_costmap,
                env_folder=env_folder
            )
            success[planner_name] = goal_reached
        
        if self.config.planner_comparison:
            planner_comparison = PlannerComparison(
                a_star_planner=planners["A*"],
                cbf_a_star_planner=planners["CBF infused A*"],
                visualizers=visualizers
            )
            planner_comparison.plot_comparison(f"{env_folder}/planner_comparison.png")

        return success
    
    def __call__(self):
        # check if work_dir is available
        if not self.work_dir_available:
            return
        
        results_per_planner = {}
        
        # iterate for all the different elements
        for i in range(self.config.number_of_simulations):
            logger.info(f"Start environment {(i+1)}/{self.config.number_of_simulations}")
            succeed = self._run_env(i+1)

            if i == 0:
                for key in succeed.keys():
                    results_per_planner[key] = {
                        'reached': 0,
                        'not reached': 0,
                        'no path': 0,
                    }

            for key, reached in succeed.items():
                if reached and isinstance(reached, bool):
                    results_per_planner[key]['reached'] += 1
                elif not reached and isinstance(reached, bool):
                    results_per_planner[key]['not reached'] += 1
                elif reached is None:
                    results_per_planner[key]['no path'] += 1
        
        logger.success(f"Simulations done: {self.config.number_of_simulations} executed.")

        # log all the results
        for planner_name, results in results_per_planner.items():
            logger.info(f"Results for {planner_name} planner:")
            logger.info(f"Number goal reached: {results['reached']}")
            logger.info(f"Number goal not reached: {results['not reached']}")
            logger.info(f"No path found: {results['no path']}")


def main():
    directory = './runs/debug'
    
    # cbf_mode 0: PD + CBF
    # cbf_mode 1: CLF + CBF
    config = EnvGeneratorConfig(
        number_of_simulations=1,
        fps=50,
        min_number_of_obstacles=5,
        max_number_of_obstacles=10,
        environment_size=(20, 20),
        grid_size=0.1,
        max_obstacle_size={
            'circle': 3.0,
            'rectangle': (3.0, 3.0)
        },
        max_duration_of_simulation=20,
        min_goal_distance=15,
        robot_size=(1, 1),
        state_std=np.array([0.1, 0.1, 0.0, 0.0]),
        use_safety_margin=True,
        cbf_reduction='min',
        work_dir=directory,
        cbf_infused_a_star=True,
        planner_comparison=False,
        number_of_sensors=10,
        max_sensor_noise=0.1
    )
    
    # create logger
    logger.add(f"{directory}/simulations.log", rotation="10 MB")

    # create the environment generator class
    envs = EnvGenerator(config=config)

    # apply the simulations
    envs()

if __name__ == '__main__':
    main()