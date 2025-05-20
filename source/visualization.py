import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from loguru import logger
import os

class VisualizeData:
    def __init__(self):
        self.u_cbf = []
        self.u_nominal = []
        self.h_true = []
        self.h_estimated = []
        self.safety_margin = []
        self.robot_pos = []
        self.robot_pos_estimated = []
        self.robot_vel = []
        self.planner_costmap = []
        self.cbf_costmap = []
        self.uncertainty_costmap = []
        self.sensor_positions = []
        self.path = []
        self.timestep = []
        self.converted_to_numpy = False

    def to_numpy(self):
        if not self.converted_to_numpy:
            for attr, value in self.__dict__.items():
                if isinstance(value, list):
                    setattr(self, attr, np.array(value))
            
            self.converted_to_numpy = True
    
    def save_data(self, dir):
        # function to save all the data to npy files
        # check if converted to numpy
        if not self.converted_to_numpy:
            self.to_numpy()
        
        # Create directory if it doesn't exist
        os.makedirs(dir, exist_ok=True)
        
        # Save each numpy array to a .npy file
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                file_path = f"{dir}/{attr}.npy"
                np.save(file_path, value)
        
        logger.success(f"Visualization data saved: {dir}")


class VisualizeSimulation:
    def __init__(self, pos_goal, obstacles=[], show_plot=True):
        self.data = VisualizeData()
        self.pos_goal = pos_goal
        self.obstacles = obstacles
        self.show_plot = show_plot
    
    def clear(self):
        # clear the data dictionary
        self.data = VisualizeData()
    
    def plot_state(self, axes):
        # this function converts the axes to plots for the state
        x = self.data.timestep   
        robot_vel = self.data.robot_vel
        robot_pos = self.data.robot_pos
        robot_pos_estimated = self.data.robot_pos_estimated
        dim_pos = robot_pos.shape[1]

        for i in range(dim_pos):
            axes[i].plot(x, robot_pos_estimated[:, i], label='Estimated')
            axes[i].plot(x, robot_pos[:, i], label='True')
            axes[i].set_title(f'Position over time (axes={i})')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('Position [m]')
            axes[i].legend()
            axes[i].grid(True)

        for i in range(robot_vel.shape[1]):
            idx = i + dim_pos
            axes[idx].plot(x, robot_vel[:, i])
            axes[idx].set_title(f'Velocity over time (axes={i})')
            axes[idx].set_xlabel('Time step [-]')
            axes[idx].set_ylabel('Velocity [m/s]')
            axes[idx].grid(True)
        
        return axes

    def plot_control_input(self, axes):
        # this function converts a list of axes to a list of figures with the control inputs over time
        u_nominal = self.data.u_nominal
        u_cbf = self.data.u_cbf
        x = self.data.timestep
        dim_controller = u_nominal.shape[1]

        # Plot the data for controller
        for i in range(dim_controller):
            axes[i].plot(x, u_cbf[:, i], label=f'u cbf {i}')
            axes[i].plot(x, u_nominal[:, i], label=f'u nominal {i}')

            # Customize the plot
            axes[i].set_title('Control input over time')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('Control input')
            axes[i].legend()
            axes[i].grid(True)

        return axes  # Return the modified axis

    def plot_safety_margin(self, ax):
        # function to plot the safety margin over time
        x = self.data.timestep
        safety_margins = self.data.safety_margin
        labels = [f"CBF {i}" for i in range(safety_margins.shape[1])]
        ax.plot(x, safety_margins, label=labels)
        ax.set_title(f'Safety margin over time')
        ax.set_xlabel('Time step [-]')
        ax.set_ylabel('Safety margin')
        ax.legend()
        ax.grid(True)
        return ax

    def plot_cbf(self, axes):
        # converts a list of axes to figures with the value of the cbf over time
        h_true = self.data.h_true
        h_estimated = self.data.h_estimated
        x = self.data.timestep
        num_cbfs = h_true.shape[1]  # Number of CBFs

        # Plot each CBF separately
        for i in range(num_cbfs):
            axes[i].plot(x, h_estimated[:, i], label=f'estimated cbf {i}')
            axes[i].plot(x, h_true[:, i], label=f'true cbf {i}')
            axes[i].set_title(f'CBF {i} over time')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('h')
            axes[i].legend()
            axes[i].grid(True)

        return axes  # Return the modified axes
    
    def plot_distance_costmap(self, ax, planner):
        # get all the data
        costmap = planner.costmap
        path = planner.path_world
        distance_map = self.data.planner_costmap

        obstacle_mask = np.isinf(costmap)
        robot_pos = self.data.robot_pos

        # Display map (avoid inf for colormap)
        if np.any(obstacle_mask):
            max_val = np.max(distance_map[~obstacle_mask])
            distance_map[obstacle_mask] = max_val + 1
            vmax = max_val + 1
            vmin = np.min(distance_map[~obstacle_mask])
        else:
            vmax = np.max(distance_map)
            vmin = np.min(distance_map)

        # Show distance/cost map
        extent = [
            *planner.grid_to_world((0, 0)),                             # lower left (x_min, y_min)
            *planner.grid_to_world((planner.rows, planner.cols))        # upper right (x_max, y_max)
        ]
        extent = [extent[0], extent[2], extent[1], extent[3]]  # reorder for imshow: [x_min, x_max, y_min, y_max]
        img = ax.imshow(distance_map, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, extent=extent)

        # Overlay obstacles as black
        for obstacle in self.obstacles:
            drawing = obstacle.pyplot_drawing(opacity=0.7)
            ax.add_patch(drawing)

        # Plot path
        ax.plot(path[:, 0], path[:, 1], color='cyan', label='Planned traj')

        # Plot start and goal
        ax.plot(robot_pos[0, 0], robot_pos[0, 1], 'ro', label='Start')
        ax.plot(self.pos_goal[0], self.pos_goal[1], 'go', label='Goal')

        ax.set_title("Costmap and A* Path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

        fig = ax.get_figure()  # get the parent figure of this axis
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Distance from start", rotation=270, labelpad=15)

        return ax
    
    def plot_cbf_costmap(self, ax, planner, cbf_costmap=None):
        # input of cbf_costmap is the array of the cbf_costmap
        # the planner is the planner object
        # get all the data
        if cbf_costmap is None:
            cbf_costmap = self.data.cbf_costmap
        else:
            cbf_costmap = np.array(cbf_costmap.costmap)
        
        path = np.array(planner.path_world)
        robot_pos = self.data.robot_pos

        # Basic min/max values for colormap
        vmax = np.max(cbf_costmap)
        vmin = np.min(cbf_costmap)

        # Define extent in world coordinates
        extent = [
            *planner.grid_to_world((0, 0)),
            *planner.grid_to_world((planner.rows, planner.cols))
        ]
        extent = [extent[0], extent[2], extent[1], extent[3]]  # reorder for imshow

        # Plot costmap
        img = ax.imshow(cbf_costmap, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax, extent=extent)

        # plot contour for h=0 values
        X = np.linspace(extent[0], extent[1], cbf_costmap.shape[1])
        Y = np.linspace(extent[2], extent[3], cbf_costmap.shape[0])
        X, Y = np.meshgrid(X, Y)

        contour = ax.contour(X, Y, cbf_costmap, levels=[0], colors='white', linewidths=2)
        ax.clabel(contour, fmt='h=0', colors='white', fontsize=9)

        # plot path and real trajectory
        ax.plot(path[:, 0], path[:, 1], color='cyan', label='Planned traj')
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], color='lime', label='Robot traj')

        # plot goal and start
        ax.plot(robot_pos[0, 0], robot_pos[0, 1], 'ro', label='Start')
        ax.plot(self.pos_goal[0], self.pos_goal[1], 'go', label='Goal')

        ax.set_title("CBF Costmap")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

        # Colorbar
        fig = ax.get_figure()
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("CBF cost", rotation=270, labelpad=15)

        return ax

    def plot_uncertainty_costmap(self, ax, planner, uncertainty_costmap=None):
        # input of cbf_costmap is the array of the cbf_costmap
        # the planner is the planner object
        # get all the data
        if uncertainty_costmap is None:
            uncertainty_costmap = self.data.uncertainty_costmap
        else:
            uncertainty_costmap = np.array(uncertainty_costmap.costmap)
        
        path = np.array(planner.path_world)
        robot_pos = self.data.robot_pos
        sensor_positions = self.data.sensor_positions

        # Basic min/max values for colormap
        vmax = np.max(uncertainty_costmap)
        vmin = np.min(uncertainty_costmap)

        # Define extent in world coordinates
        extent = [
            *planner.grid_to_world((0, 0)),
            *planner.grid_to_world((planner.rows, planner.cols))
        ]
        extent = [extent[0], extent[2], extent[1], extent[3]]  # reorder for imshow

        # Plot costmap
        img = ax.imshow(uncertainty_costmap, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax, extent=extent)

        # plot path and real trajectory
        ax.plot(path[:, 0], path[:, 1], color='cyan', label='Planned traj')
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], color='lime', label='Robot traj')

        # plot goal and start
        ax.plot(robot_pos[0, 0], robot_pos[0, 1], 'ro', label='Start')
        ax.plot(self.pos_goal[0], self.pos_goal[1], 'go', label='Goal')

        # plot sensors 
        ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='k', label='Sensor pos')

        ax.set_title("Uncertainty Costmap")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

        # Colorbar
        fig = ax.get_figure()
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Uncertainty (noise)", rotation=270, labelpad=15)

        return ax

    def plot_robot_trajectory(self, ax, path=None):
        # Convert list to array
        robot_pos = self.data.robot_pos
        robot_pos_estimated = self.data.robot_pos_estimated
        if path is None:
            path = self.data.path
        else:
            path = np.array(path)

        # Plot robot trajectory
        ax.plot(robot_pos_estimated[:, 0], robot_pos_estimated[:, 1], label='Robot traj est', ls='--', alpha=0.6, color='m')
        ax.plot(path[:, 0], path[:, 1], label='Planned traj')
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], label='Robot traj')
        ax.plot(self.pos_goal[0], self.pos_goal[1], color='green', marker='o', label='Goal')
        ax.plot(robot_pos[0, 0], robot_pos[0, 1], color='red', marker='o', label='Start')

        # Add obstacles
        for obstacle in self.obstacles:
            drawing = obstacle.pyplot_drawing()
            ax.add_patch(drawing)

        # Customize the plot
        ax.set_title('Robot trajectory over time')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.legend()
        ax.grid(True)

        return ax 
    
    def create_full_plot(self, planner, filename=None):
        # convert lists to array
        self.data.to_numpy()

        # function save figure if there is a filename
        num_colom_state = self.data.robot_pos.shape[1] + self.data.robot_vel.shape[1]
        num_coloms_control = self.data.u_nominal.shape[1] + 1   # +1 for the safety margin
        num_colom_cbfs = self.data.h_true.shape[1]
        num_costmaps = 4

        # Determine number of rows and columns
        num_rows = 4
        num_cols = max([num_colom_state, num_colom_cbfs, num_coloms_control, num_costmaps])  

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * (num_rows)))

        # Ensure axes is iterable
        if num_rows == 1 and num_cols == 1:
            axes = [[axes]]
        elif num_rows == 1:
            axes = [axes]
        elif num_cols == 1:
            axes = [[ax] for ax in axes]

        # Row 0: state over time
        self.plot_state(axes=axes[0])

        # remove unused subplots
        if num_colom_state < num_cols:
            for i in range(num_colom_state, num_cols):
                fig.delaxes(axes[0][i]) 

        # Row 1: Plot control input (single subplot spanning all columns)
        self.plot_control_input(axes=axes[1])
        self.plot_safety_margin(ax=axes[1][num_coloms_control-1])

        # remove unused subplots
        if num_coloms_control < num_cols:
            for i in range(num_coloms_control, num_cols):
                fig.delaxes(axes[1][i]) 

        # Row 2: Plot multiple CBFs in separate columns
        self.plot_cbf(axes=axes[2])

        # remove unused subplots
        if num_colom_cbfs < num_cols:
            for i in range(num_colom_cbfs, num_cols):
                fig.delaxes(axes[2][i])

        # Row 3: plot costmaps
        self.plot_robot_trajectory(ax=axes[3][0])

        # costmap for planner
        self.plot_distance_costmap(ax=axes[3][1], planner=planner)

        # costmap for cbf
        self.plot_cbf_costmap(ax=axes[3][2], planner=planner)

        # costmap for uncertainty
        self.plot_uncertainty_costmap(ax=axes[3][3], planner=planner)
        
        # remove unused subplots
        if num_cols > 4:
            for i in range(4, num_cols):
                fig.delaxes(axes[3][i])

        plt.tight_layout()

        if isinstance(filename, str):
            plt.savefig(filename)
            logger.success(f"Visualization saved: {filename}")
            plt.close()
        elif isinstance(filename, list):
            for f in filename:
                plt.savefig(f)
                logger.success(f"Visualization saved: {f}")
            plt.close()

        if self.show_plot:
            plt.show()

