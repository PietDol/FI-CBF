import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class VisualizeCBF:
    def __init__(self, obstacles=[]):
        self.data = {
            'control_input': {
                'u_cbf': [],
                'u_nominal': []
            },
            'h':[],
            'robot_pos': []
        }
        self.obstacles = obstacles

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
        
        # Ensure axes is iterable (even if there's only one CBF)
        if dim_controller == 1:
            axes = [axes]

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

        # Ensure axes is iterable (even if there's only one CBF)
        if num_cbfs == 1:
            axes = [axes]

        # Plot each CBF separately
        for i in range(num_cbfs):
            axes[i].plot(x, h[:, i], label=f'cbf {i}')
            axes[i].set_title(f'CBF {i} over time')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('h')
            axes[i].legend()
            axes[i].grid(True)

        return axes  # Return the modified axes

    def plot_robot_trajectory(self, ax=None):
        # Convert list to array
        robot_pos = np.array(self.data['robot_pos'])

        # Create axis if not provided
        if ax is None:
            fig, ax = plt.subplots()

        # Plot robot trajectory
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], label='Robot trajectory')

        # Add obstacles
        for obstacle in self.obstacles:
            cx, cy = obstacle.pos_center
            bottom_left_x = cx - obstacle.width / 2
            bottom_left_y = cy - obstacle.height / 2

            square = patches.Rectangle((bottom_left_x, bottom_left_y), 
                                    obstacle.width, obstacle.height, 
                                    color='black', fill=True)
            ax.add_patch(square)

        # Customize the plot
        ax.set_title('Robot trajectory over time')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.legend()
        ax.grid(True)

        return ax  # Return the modified axis

    
    def create_plot(self, plot_types):
        num_control = 'control_input' in plot_types
        num_coloms_control = len(self.data['control_input']['u_nominal'][0]) if 'control_input' in plot_types and len(self.data['control_input']['u_nominal']) > 0 else 0
        num_cbfs = 'h' in plot_types
        num_colom_cbfs = len(self.data['h'][0]) if 'h' in plot_types and len(self.data['h']) > 0 else 0
        num_robot = 'robot_pos' in plot_types

        # Determine number of rows and columns
        num_rows = num_control + num_robot + num_cbfs 
        num_cols = max([num_colom_cbfs, num_coloms_control, 1])  

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
            
            # remove unused subplots
            if num_cols > 1:
                for i in range(1, num_cols):
                    fig.delaxes(axes[ax_idx][i])

        plt.tight_layout()
        plt.show()

