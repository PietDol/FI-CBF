# FI-CBF
Git repository for feedback-integrated Control Barrier Functions (FI-CBF).

## Code structure

This section outlines the structure of the simulation code. The system is divided into two main components: the **Environment Manager** and the **Robot**.

- The **Environment Manager** is responsible for generating and managing the simulation environment. It can create randomized environments or load them from file and then run the simulation.
- The **Robot** module contains all the robot-related components: **Perception**, **Planner**, **Control Barrier Function (CBF)**, and **Visualization**. Obstacles are linked to the robot so that they can be accessed by the planner, CBF, and the visualization module.

![Code structure of the simulation](documentation/code_structure.svg)

The simulation pipeline follows these steps:

1. **Environment Configuration**:  
   The simulation begins with the creation of an `EnvGeneratorConfig` object. This configuration can be either created by the user or loaded from a previous run.

2. **Environment Generator Creation**:  
   Using the configuration, an `EnvGenerator` object is instantiated. Upon creation, relevant parameters and environment details are saved to a JSON file.

3. **Simulation Initialization**:  
   The user can either call the object directly (via `__call__`) or invoke the `run_env_from_file()` method.  
   - `run_env_from_file()` loads a previously saved environment and executes the simulation within it.  
   - If the object is called, `EnvGenerator` will run the number of simulations specified in the config file.

4. **Environment Element Generation**:  
   The `_generate_env_elements()` method constructs all simulation components: the robot, its sensors, and the obstacles.  
   When invoked via `__call__`, it randomly generates:  
   - start and goal positions for the robot  
   - obstacles of varying sizes and locations  
   - sensors with varying positions

5. **Running the Simulation**:  
   The `_run_env()` method runs the simulation by calling the robot's logic. It also manages performance tracking across simulations. After the simulation completes, the system logs data and generates plots of:  
   - state evolution over time  
   - nominal vs. CBF control inputs over time
   - CBF values over time
   - robot trajectory
   - costmaps (planner, CBF, perception magnitude, noise)

6. **Batch Execution**:  
   When the `EnvGenerator` object is called, it generates and runs simulations until the configured number is reached.  
   If `run_env_from_file()` is used, a single simulation is executed and the program exits afterward.