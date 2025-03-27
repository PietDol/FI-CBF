# python file to random generate environments
from loguru import logger
from env_config import EnvConfig
from robot_base_env import RobotBaseEnv, pd_controller
from robot_base_config import RobotBaseCBFConfig, RobotBaseCLFCBFConfig
from robot_base import RobotBase
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
                 max_obstacle_size: dict,
                 max_duration_of_simulation: float,
                 min_goal_distance: float,
                 work_dir: str, 
                 robot_size=(1, 1),
                 safety_margin=0.0,
                 cbf_mode=0):
        self.number_of_simulations = number_of_simulations
        self.max_number_of_obstacles = max_number_of_obstacles
        self.environment_size = environment_size
        self.max_obstacle_size = max_obstacle_size
        self.max_duration_of_simulation = max_duration_of_simulation    # in seconds
        self.min_goal_distance = min_goal_distance
        self.work_dir = work_dir
        self.robot_size = robot_size
        self.safety_margin = safety_margin
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

    def _create_work_dir(self):
        # function to create work_dir
        try:
            os.makedirs(f"{self.config.work_dir}/simulation_results", exist_ok=False)
            logger.info(f"New simulation is started. Workdir: {self.config.work_dir}")
            return True
        except FileExistsError:
            logger.error(f"Directory already exists: {self.config.work_dir}")
            return False
    
    def _generate_goal(self, robot_pos, max_tries=1000):
        cx, cy = robot_pos
        max_dist = 0

        for _ in range(max_tries):
            gx = random.uniform(-self.x_range, self.x_range)
            gy = random.uniform(-self.y_range, self.y_range)
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
        for i, obstacle in enumerate(obstacles):
            collision = obstacle.check_collision(robot)
            goal_feasible = obstacle.check_goal_position(robot)

            if generation and (collision or not goal_feasible):
                logger.info(f"Robot or goal position in collision with obstacle, remove obstacle.")
                obstacles.pop(i)
            elif not generation and collision:
                logger.error(f"Collision between robot and obstacle! Robot position: {robot.position}, obstacle index: {i}")
        
        if generation:
            return obstacles
    
    def _generate_obstacles(self, robot: RobotBase, max_tries=1000):
        obstacles = []
        for i in range(max_tries):
            number_of_obstacles = random.randint(1, self.config.max_number_of_obstacles)

            for _ in range(number_of_obstacles):
                shape = random.choice(["circle", "rectangle"])
                cx = random.uniform(-self.x_range, self.x_range)
                cy = random.uniform(-self.y_range, self.y_range)
                pos_center = np.array([cx, cy])

                if shape == "circle":
                    radius = random.uniform(1, self.config.max_obstacle_size["circle"])
                    obstacle = CircleObstacle(
                        radius=radius,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=robot
                    )

                    logger.info(f"Generated CircleObstacle - center: {pos_center}, radius: {radius}")
                elif shape == "rectangle":
                    width = random.uniform(1, self.config.max_obstacle_size["rectangle"][0])
                    height = random.uniform(1, self.config.max_obstacle_size["rectangle"][1])
                    obstacle = RectangleObstacle(
                        width=width,
                        height=height,
                        pos_center=pos_center,
                        env_config=self.env_config,
                        robot=robot
                    )
                    logger.info(f"Generated RectangleObstacle - center: {pos_center}, width: {width}, height: {height}")
                
                obstacles.append(obstacle)
            
            # check if they are in collision
            obstacles = self._check_collision(robot, obstacles, generation=True)

            if len(obstacles) > 0:
                return obstacles
        
        logger.error(f"Not able to generate obstacles after {max_tries} tries. This iteration will be skipped.")
        return []        

    def _generate_env_elements(self):
        # generate the robot object
        robot_x = random.uniform(-self.x_range, self.x_range)
        robot_y = random.uniform(-self.y_range, self.y_range)
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
        
        return robot_base, obstacles
    
    def _generate_env(self):
        # function to generate the environment
        robot, obstacles = self._generate_env_elements()

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

        return env, visualizer, config, cbf

    @logger.catch
    def _run_env(self, filename='env.png'):
        # function to run the environment
        # generate the environment
        env, visualizer, config, cbf = self._generate_env()
        max_timesteps = env.fps * self.config.max_duration_of_simulation
        timesteps = 0

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
        
        if env.robot_base.check_goal_reached():
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
        visualizer.create_plot(['control_input', 'h', 'robot_pos'], f"{self.config.work_dir}/simulation_results/{filename}")
        logger.info(f"Visualization saved: {filename}")

        # return whether the goal is reached
        return goal_reached
    
    def __call__(self):
        # check if work_dir is available
        if not self.work_dir_available:
            return

        goal_reached = 0
        goal_not_reached = 0

        # iterate for all the different elements
        for i in range(self.config.number_of_simulations):
            logger.info(f"Start environment {(i+1)}/{self.config.number_of_simulations}")
            succeed = self._run_env(f"env_{i}.png")

            if succeed:
                goal_reached += 1
            elif not succeed:
                goal_not_reached += 1

        
        logger.success(f"Simulations done: {self.config.number_of_simulations} executed.")
        logger.info(f"Number goal reached: {goal_reached}")
        logger.info(f"Number goal not reached: {goal_not_reached}")


def main():
    directory = './runs'
    
    # cbf_mode 0: PD + CBF
    # cbf_mode 1: CLF + CBF
    config = EnvGeneratorConfig(
        number_of_simulations=1000,
        max_number_of_obstacles=10,
        environment_size=(20, 20),
        max_obstacle_size={
            'circle': 3.0,
            'rectangle': (3.0, 3.0)
        },
        max_duration_of_simulation=20,
        min_goal_distance=15,
        robot_size=(1, 1),
        safety_margin=0.1,
        cbf_mode=0,
        work_dir=directory
    )
    
    # create the environment generator class
    envs = EnvGenerator(config=config)

    # create logger
    logger.add(f"{directory}/simulations.log", rotation="10 MB")

    # apply the simulations
    envs()

if __name__ == '__main__':
    main()