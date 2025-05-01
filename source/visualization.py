import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class VisualizeData:
    def __init__(self):
        pass

class VisualizeCBF:
    def __init__(self, pos_goal, obstacles=[], show_plot=True):
        self._empty_dataset = {
            'control_input': {
                'u_cbf': [],
                'u_nominal': []
            },
            'h':[],
            'robot_pos': [],
            'path': [],
            'costmap': [], 
            'distance_map': [],
            'cbf_costmap': []
        }
        self.data = self._empty_dataset
        self.pos_goal = pos_goal
        self.obstacles = obstacles
        self.show_plot = show_plot
    
    def clear(self):
        # clear the data dictionary
        self.data = self._empty_dataset

    def plot_control_input(self, ax=None):
        # Convert lists to array
        u_nominal = np.array(self.data['control_input']['u_nominal'])
        u_cbf = np.array(self.data['control_input']['u_cbf'])
        x = np.arange(u_nominal.shape[0])
        dim_controller = u_nominal.shape[1]

        # Create axis if not provided
        if ax is None:
            fig, ax = plt.subplots()
        else:
            axes = ax

        # Plot the data
        for i in range(dim_controller):
            axes[i].plot(x, u_nominal[:, i], label=f'u nominal {i}')
            axes[i].plot(x, u_cbf[:, i], label=f'u cbf {i}')

            # Customize the plot
            axes[i].set_title('Control input over time')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('Control input')
            axes[i].legend()
            axes[i].grid(True)

        return axes  # Return the modified axis

    def plot_cbf(self, ax=None):
        # Convert list to array
        h = np.array(self.data['h'])
        x = np.arange(h.shape[0])

        num_cbfs = h.shape[1]  # Number of CBFs

        # Create multiple axes in a row if not provided
        if ax is None:
            fig, axes = plt.subplots(1, num_cbfs, figsize=(5 * num_cbfs, 4))
        else:
            axes = ax

        # Plot each CBF separately
        for i in range(num_cbfs):
            axes[i].plot(x, h[:, i], label=f'cbf {i}')
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
        distance_map = planner.distance_map

        obstacle_mask = np.isinf(costmap)
        robot_pos = np.array(self.data['robot_pos'])

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
            *planner.grid_to_world((0, 0)),                     # lower left (x_min, y_min)
            *planner.grid_to_world((planner.rows, planner.cols))      # upper right (x_max, y_max)
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
            cbf_costmap = np.array(self.data['cbf_costmap'])
        else:
            cbf_costmap = np.array(cbf_costmap.costmap)
        
        path = np.array(planner.path_world)
        robot_pos = np.array(self.data['robot_pos'])

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

    def plot_robot_trajectory(self, ax=None, path=None):
        # Convert list to array
        robot_pos = np.array(self.data['robot_pos'])
        if path is None:
            path = np.array(self.data['path'])
        else:
            path = np.array(path)

        # Plot robot trajectory
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
    
    def create_plot(self, plot_types, planner, filename=''):
        # function save figure if there is a filename
        num_control = 'control_input' in plot_types
        num_coloms_control = len(self.data['control_input']['u_nominal'][0]) if 'control_input' in plot_types and len(self.data['control_input']['u_nominal']) > 0 else 0
        num_cbfs = 'h' in plot_types
        num_colom_cbfs = len(self.data['h'][0]) if 'h' in plot_types and len(self.data['h']) > 0 else 0
        num_robot = 'robot_pos' in plot_types
        num_trajectory = 3 if 'robot_pos' in plot_types else 0

        # Determine number of rows and columns
        num_rows = num_control + num_robot + num_cbfs 
        num_cols = max([num_colom_cbfs, num_coloms_control, num_trajectory])  

        # if there are no rows just return
        if num_rows < 1:
            return

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * (num_rows)))

        # Ensure axes is iterable
        if num_rows == 1 and num_cols == 1:
            axes = [[axes]]
        elif num_rows == 1:
            axes = [axes]
        elif num_cols == 1:
            axes = [[ax] for ax in axes]

        ax_idx = 0  # Track which row we're plotting on

        # Plot control input (single subplot spanning all columns)
        if num_control:
            self.plot_control_input(ax=axes[ax_idx])

            # remove unused subplots
            if num_coloms_control < num_cols:
                for i in range(num_coloms_control, num_cols):
                    fig.delaxes(axes[ax_idx][i])
            
            # Move to next row
            ax_idx += 1  

        # Plot multiple CBFs in separate columns
        if num_cbfs > 0:
            self.plot_cbf(ax=axes[ax_idx])

            # remove unused subplots
            if num_colom_cbfs < num_cols:
                for i in range(num_colom_cbfs, num_cols):
                    fig.delaxes(axes[ax_idx][i])

            # move to next row
            ax_idx += 1

        # Plot robot trajectory (last row, spanning all columns)
        if num_robot:

            self.plot_robot_trajectory(ax=axes[ax_idx][0])

            # costmap for planner
            self.plot_distance_costmap(ax=axes[ax_idx][1], planner=planner)

            # costmap for cbf
            self.plot_cbf_costmap(ax=axes[ax_idx][2], planner=planner)
            
            # remove unused subplots
            if num_cols > 3:
                for i in range(3, num_cols):
                    fig.delaxes(axes[ax_idx][i])

        plt.tight_layout()

        if isinstance(filename, str) and filename != '':
            plt.savefig(filename)
            plt.close()

        if self.show_plot:
            plt.show()

