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
                 max_number_of_simulations: int, 
                 max_number_of_obstacles: int,
                 environment_size: tuple,
                 max_obstacle_size: dict,
                 max_duration_of_simulation: float,
                 work_dir: str, 
                 robot_size=(1, 1),
                 safety_margin=0.0,
                 cbf_mode=0):
        self.max_number_of_simulations = max_number_of_simulations
        self.max_number_of_obstacles = max_number_of_obstacles
        self.environment_size = environment_size
        self.max_obstacle_size = max_obstacle_size
        self.max_duration_of_simulation = max_duration_of_simulation    # in seconds
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
            logger.warning(f"Directory already exists: {self.config.work_dir}")
            return False

    def _generate_env_elements(self):
        # generate the robot object
        robot_x = random.uniform(-self.x_range, self.x_range)
        robot_y = random.uniform(-self.y_range, self.y_range)
        goal_x = random.uniform(-self.x_range, self.x_range)
        goal_y = random.uniform(-self.y_range, self.y_range)

        robot_base = RobotBase(
            width=self.config.robot_size[0],
            height=self.config.robot_size[1],
            env_config=self.env_config,
            pos_goal=np.array([goal_x, goal_y]),
            pos_center_start=np.array([robot_x, robot_y]),
            safety_margin=self.config.safety_margin
        )
        logger.info(f"Robot start: {robot_base.position}, goal: {robot_base.pos_goal}")

        # generate the obstacles
        obstacles = []
        number_of_obstacles = random.randint(1, self.config.max_number_of_obstacles)

        for i in range(number_of_obstacles):
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
                    robot=robot_base
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
                    robot=robot_base
                )
                logger.info(f"Generated RectangleObstacle - center: {pos_center}, width: {width}, height: {height}")
            
            obstacles.append(obstacle)
        
        # check if they are in collision
        obstacles = self._check_collision(robot_base, obstacles)
        
        return robot_base, obstacles
    
    def _check_collision(self, robot, obstacles):
        # checks if there is a collision between robot and obstacles
        for i, obstacle in enumerate(obstacles):
            collision = obstacle.check_collision(robot)
            if collision:
                obstacles.pop(i)
        
        # TODO: check if there is a collision between obstacles
        return obstacles
    
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
        while env.running and timesteps <= max_timesteps:
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
            env.step()
            
            # increment timestep
            timesteps += 1
        
        if env.robot_base.check_goal_reached():
            logger.success(f"Goal reached in {timesteps} timesteps.")
        else:
            logger.warning(f"Simulation ended without reaching the goal. Timesteps: {timesteps}")

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

    
    def __call__(self):
        # check if work_dir is available
        if not self.work_dir_available:
            return

        # iterate for all the different elements
        for i in range(self.config.max_number_of_simulations):
            logger.info(f"Start environment {(i+1)}/{self.config.max_number_of_simulations}")
            self._run_env(f"env_{i}.png")
        
        logger.success(f"Simulations done: {self.config.max_number_of_simulations} executed.")


def main():
    directory = './runs'
    # create logger
    logger.add(f"{directory}/simulations.log", rotation="10 MB")

    # cbf_mode 0: PD + CBF
    # cbf_mode 1: CLF + CBF
    config = EnvGeneratorConfig(
        max_number_of_simulations=10,
        max_number_of_obstacles=3,
        environment_size=(20, 20),
        max_obstacle_size={
            'circle': 3.0,
            'rectangle': (3.0, 3.0)
        },
        max_duration_of_simulation=20,
        robot_size=(1, 1),
        safety_margin=0.1,
        cbf_mode=0,
        work_dir=directory
    )

    envs = EnvGenerator(config=config)
    envs()

if __name__ == '__main__':
    main()