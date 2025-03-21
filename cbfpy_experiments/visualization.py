import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class VisualizeCBF:
    def __init__(self):
        self.u_cbf = []
        self.u_nominal = []
        self.h = []
        self.robot_pos = []

    def plot_control_input(self):
        # convert lists to array
        u_nominal = np.array(self.u_nominal)
        u_cbf = np.array(self.u_cbf)

        # Create figure and axes
        x = np.arange(u_nominal.shape[0])
        fig, axes = plt.subplots(1, u_nominal.shape[1])

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Plot the data
        for i in range(u_nominal.shape[1]):
            axes[i].plot(x, u_nominal[:, i], label=f'u nominal {i}')
            axes[i].plot(x, u_cbf[:, i], label=f'u cbf {i}')

            # Customize the plot
            axes[i].set_title('Control input over time')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('Control input')
            axes[i].legend()
            axes[i].grid(True)

        # Show the figure
        plt.show()
    
    def plot_cbf(self):
        # convert list to array
        h = np.array(self.h)

        # Create figure and axes
        x = np.arange(h.shape[0])
        fig, axes = plt.subplots(1, h.shape[1])

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Plot the data
        for i in range(h.shape[1]):
            axes[i].plot(x, h[:, i], label=f'cbf {i}')

            # Customize the plot
            axes[i].set_title('CBF over time')
            axes[i].set_xlabel('Time step [-]')
            axes[i].set_ylabel('h')
            axes[i].legend()
            axes[i].grid(True)

        # Show the figure
        plt.show()
    
    def plot_robot_trajectory(self, ):
        # function to show the trajectory of the robot
        robot_pos = np.array(self.robot_pos)

        # Create figure and axes
        x = np.arange(robot_pos.shape[0])
        fig, ax = plt.subplots(1, 1)

        # create the plot
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], label='Robot trajectory')
        ax.set_title('Robot trajectory over time')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.legend()
        ax.grid(True)

        # Show the figure
        plt.show()

