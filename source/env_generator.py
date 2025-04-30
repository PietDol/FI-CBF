# python file to random generate environments
from loguru import logger
from env_config import EnvConfig
from robot_base_env import RobotBaseEnv, pd_controller
from robot_base_config import RobotBaseCBFConfig, RobotBaseCLFCBFConfig
from robot_base import RobotBase
from cbf_costmap import CBFCostmap
from a_star import AStarPlanner
from visualization import VisualizeCBF
from cbfpy import CBF, CLFCBF
import random
from obstacles import RectangleObstacle, CircleObstacle
import numpy as np
import os


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
                 robot_size=(1, 1),
                 safety_margin=0.0,
                 cbf_reduction='min',
                 cbf_mode=0):
        self.number_of_simulations = number_of_simulations
        self.max_number_of_obstacles = max_number_of_obstacles
        self.environment_size = environment_size                        # in m
        self.grid_size = grid_size                                      # in m
        self.max_obstacle_size = max_obstacle_size                      # in m
        self.max_duration_of_simulation = max_duration_of_simulation    # in seconds
        self.min_goal_distance = min_goal_distance
        self.work_dir = work_dir
        self.robot_size = robot_size
        self.safety_margin = safety_margin
        self.cbf_reduction = cbf_reduction  # the reduction to get the cbf in one grid, options: ['min', 'mean', 'sum']
        self.cbf_mode = cbf_mode

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
        logger.info(f"Grid size: {self.config.grid_size} m")
        logger.info(f"Max duration for simulation: {self.config.max_duration_of_simulation} s")
        logger.info(f"Minimum distance to the goal: {self.config.min_goal_distance} m")
        logger.info(f"Robot size: {self.config.robot_size}")
        logger.info(f"Safety margin: {self.config.safety_margin}")
        logger.info(f"CBF reduction mode: {self.config.cbf_reduction}")
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
            collision = obstacle.check_collision(robot, self.config.safety_margin)
            goal_feasible = obstacle.check_goal_position(robot)

            if generation and (collision or not goal_feasible):
                logger.info(f"Robot or goal position in collision with obstacle (id={obstacle.id}), remove obstacle.")
                obs_to_keep.append(False)
            elif not generation and collision:
                logger.error(f"Collision between robot and obstacle! Robot position: {robot.position}, obstacle index: {i}")
            elif generation:
                obs_to_keep.append(True)

        if generation:
            filtered_obstacles = [obstacle for obstacle, keep in zip(obstacles, obs_to_keep) if keep]
            return filtered_obstacles
    
    def _generate_obstacles(self, robot: RobotBase, max_tries=1000):
        obstacles = []
        for i in range(max_tries):
            number_of_obstacles = random.randint(1, self.config.max_number_of_obstacles)

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

        robot_base = RobotBase(
            width=self.config.robot_size[0],
            height=self.config.robot_size[1],
            env_config=self.env_config,
            pos_goal=pos_goal,
            pos_center_start=np.array([robot_x, robot_y]),
            safety_margin=self.config.safety_margin
        )
        logger.info(f"Robot start: {robot_base.position}, goal: {robot_base.pos_goal}")

        # generate the obstacles
        obstacles = self._generate_obstacles(robot_base)

        # generate the planner
        planner = AStarPlanner(
            costmap_size=self.config.environment_size,
            grid_size=self.config.grid_size,
            obstacles=obstacles
        )
        
        return robot_base, obstacles, planner
    
    def _generate_env(self):
        # function to generate the environment
        robot, obstacles, planner = self._generate_env_elements()

        # generate the path 
        path = planner.plan(
            start_coords=robot.position,
            goal_coords=robot.pos_goal
        )
        if path is None:
            # no path found
            return None, None, None, None, None, None
        robot.add_path(path["path_world"])

        # create environment
        env = RobotBaseEnv(
            env_config=self.env_config,
            robot_base=robot,
            obstacles=obstacles,
            pygame_screen=self.pygame_screen
        )

        # create visualizer
        visualizer = VisualizeCBF(
            pos_goal=robot.pos_goal,
            planner=planner,
            obstacles=obstacles,
            show_plot=False
        )

        # create config and cbf based on the mode
        if self.config.cbf_mode == 0:
            config = RobotBaseCBFConfig(obstacles, robot)
            cbf = CBF.from_config(config)
        elif self.config.cbf_mode == 1:
            config = RobotBaseCLFCBFConfig(obstacles, robot)
            cbf = CLFCBF.from_config(config)
        else:
            raise f'Incorrect CBF mode ({self.config.cbf_mode})'
        
        cbf_costmap = CBFCostmap(
            costmap_size=self.config.environment_size,
            grid_size=self.config.grid_size,
            cbf=config,
            cbf_reduction=self.config.cbf_reduction
        )

        return env, visualizer, config, cbf, planner, cbf_costmap

    @logger.catch
    def _run_env(self, filename='env.png'):
        # function to run the environment
        # generate the environment
        env, visualizer, config, cbf, planner, cbf_costmap = self._generate_env()
        if env is None:
            logger.error(f"No path found! Skip this environment!")
            return None
        max_timesteps = env.fps * self.config.max_duration_of_simulation
        timesteps = 0

         # add path and costmaps to visualizer
        visualizer.data["path"] = env.robot_base.path
        visualizer.data["costmap"] = planner.costmap
        visualizer.data["display_map"] = planner.compute_distance_map(start=env.robot_base.position)
        visualizer.data["cbf_costmap"] = cbf_costmap.costmap

        # run env
        while env.running and timesteps < max_timesteps:
            current_state = env.get_state()
            pos_des = env.get_desired_state()
            
            # calculate nominal control
            if self.config.cbf_mode == 0:
                nominal_control = pd_controller(pos_des, current_state[:2], current_state[2:])
            elif self.config.cbf_mode == 1:
                nominal_control = config.V_1(current_state)
            
            # safe data for visualizer
            h = config.h_1(current_state)
            h = config.alpha(h)
            visualizer.data['h'].append(np.array(h))
            visualizer.data['robot_pos'].append(current_state[:2])

            # apply safety filter
            if self.config.cbf_mode == 0:
                u = cbf.safety_filter(current_state, nominal_control)
            elif self.config.cbf_mode == 1:
                pos_des = np.concatenate((pos_des, np.zeros(2)))
                # no nominal input inserted here, since cbfpy does that under the hood
                u = cbf.controller(current_state, pos_des)  

            # safe control data for visualizer
            visualizer.data['control_input']['u_cbf'].append(u)
            visualizer.data['control_input']['u_nominal'].append(nominal_control)
            
            # change environment
            env.apply_control(u)
            self._check_collision(robot=env.robot_base, obstacles=env.obstacles, generation=False)
            env.step()
            
            # increment timestep
            timesteps += 1
        
        if env.robot_base.check_goal_reached(tolerance=self.config.grid_size+0.01):
            logger.success(f"Goal reached in {timesteps} timesteps.")
            goal_reached = True
        else:
            logger.warning(f"Simulation ended without reaching the goal. Timesteps: {timesteps}")
            goal_reached = False

        # add last information for visualizer
        visualizer.data['control_input']['u_cbf'].append(u)
        visualizer.data['control_input']['u_nominal'].append(nominal_control)
        h = config.h_1(current_state)
        h = config.alpha(h)
        visualizer.data['h'].append(np.array(h))
        visualizer.data['robot_pos'].append(current_state[:2])

        # generate drawings
        if goal_reached:
            filename = f"{filename.split('.')[0]}_success.png"
        else:
            filename = f"{filename.split('.')[0]}_fail.png"
        visualizer.create_plot(['control_input', 'h', 'robot_pos'], f"{self.config.work_dir}/simulation_results/{filename}")
        logger.success(f"Visualization saved: {filename}")

        # return whether the goal is reached
        return goal_reached
    
    def __call__(self):
        # check if work_dir is available
        if not self.work_dir_available:
            return

        goal_reached = 0
        goal_not_reached = 0
        error = 0

        # iterate for all the different elements
        for i in range(self.config.number_of_simulations):
            logger.info(f"Start environment {(i+1)}/{self.config.number_of_simulations}")
            succeed = self._run_env(f"env_{i}.png")

            if succeed and isinstance(succeed, bool):
                goal_reached += 1
            elif not succeed and isinstance(succeed, bool):
                goal_not_reached += 1
            elif succeed is None:
                # an error occured
                error += 1

        
        logger.success(f"Simulations done: {self.config.number_of_simulations} executed.")
        logger.info(f"Number goal reached: {goal_reached}")
        logger.info(f"Number goal not reached: {goal_not_reached}")
        logger.info(f"Number error: {error}")


def main():
    directory = './runs/debug'
    
    # cbf_mode 0: PD + CBF
    # cbf_mode 1: CLF + CBF
    config = EnvGeneratorConfig(
        number_of_simulations=100,
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
        safety_margin=0.1,
        cbf_mode=0,
        cbf_reduction='min',
        work_dir=directory
    )
    
    # create logger
    logger.add(f"{directory}/simulations.log", rotation="10 MB")

    # create the environment generator class
    
    envs = EnvGenerator(config=config)

    # apply the simulations
    envs()

if __name__ == '__main__':
    main()