# this file is created to compare different planner
from a_star import AStarPlanner
from cbf_infused_a_star import CBFInfusedAStar
import matplotlib.pyplot as plt
from loguru import logger


class PlannerComparison:
    def __init__(
        self,
        a_star_planner: AStarPlanner,
        cbf_a_star_planner: CBFInfusedAStar,
        visualizers: dict,
    ):
        self.a_star_planner = a_star_planner
        self.cbf_a_star_planner = cbf_a_star_planner
        self.visualizers = visualizers

    def plot_comparison(self, filename="planner_comparison.png"):
        # Setup figure and axes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Add more space at the top to fit row headers nicely
        fig.subplots_adjust(hspace=0.4, top=0.92)

        # Add row labels using fig.text just *above* the rows
        fig.text(
            0.5, 0.96, "Firs row A* second row CBF infused A*", ha="center", fontsize=16
        )

        # A* planner (row 0)
        if self.a_star_planner.path_world is not None:
            self.visualizers["A*"].plot_robot_trajectory(
                ax=axes[0, 0], path=self.a_star_planner.path_world
            )
            self.visualizers["A*"].plot_distance_costmap(
                ax=axes[0, 1], planner=self.a_star_planner
            )
            self.visualizers["A*"].plot_cbf_costmap(
                ax=axes[0, 2],
                cbf_costmap=self.cbf_a_star_planner.cbf_costmap,
                planner=self.a_star_planner,
            )
        else:
            logger.warning(
                "A* planner has not found a path so cannot be added to the figure."
            )

        # CBF-infused planner (row 1)
        if self.cbf_a_star_planner.path_world is not None:
            self.visualizers["CBF infused A*"].plot_robot_trajectory(
                ax=axes[1, 0], path=self.cbf_a_star_planner.path_world
            )
            self.visualizers["CBF infused A*"].plot_distance_costmap(
                ax=axes[1, 1], planner=self.cbf_a_star_planner
            )
            self.visualizers["CBF infused A*"].plot_cbf_costmap(
                ax=axes[1, 2],
                cbf_costmap=self.cbf_a_star_planner.cbf_costmap,
                planner=self.cbf_a_star_planner,
            )
        else:
            logger.warning(
                "CBF infused A* planner has not found a path so cannot be added to the figure."
            )

        # Save and close
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # reserve space for fig.text at top
        plt.savefig(filename)
        logger.success(f"Planner comparison saved: {filename}")
        plt.close()
