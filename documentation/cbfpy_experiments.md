# CBFpy experiments
This file is create to play around with the CBFpy package and get familiar with it. 

### Classes created
In this section a list of all the create classes are listed:
- `RobotBaseEnv`: Environment for the robot base. It inherits from `BaseEnv` from CBFpy. It contains all the functionalities to get the environment working.
- `RobotBase`: class for the robot base. It keeps track of the position and velocity of the base and it has a function to generate the pygame drawing.
- `RectangleObstacle`: class for rectangle obstacles. It inherits from `Obstacle` which is an abstract class for all the different obstacles.
- `VisualizeCBF`: class to visualize how the simulation went. You can choose what you want to visualize. Everything is plotted in one big figure. Currently the following elements can be plotted:
    - Input of nominal controller and CBF
    - Value for the CBF
    - Trajectory of the robot    