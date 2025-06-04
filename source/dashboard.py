import os
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html

# Path to the directory with .npy files
DATA_DIR = "./runs/baseline_hard_3/simulation_results/loaded_env/simulation_data"  # Replace with actual path

def load_npy(name):
    return np.load(os.path.join(DATA_DIR, f"{name}.npy"))

# Load all relevant data
robot_pos = load_npy("robot_pos")
robot_pos_est = load_npy("robot_pos_estimated")
robot_vel = load_npy("robot_vel")
u_cbf = load_npy("u_cbf")
u_nominal = load_npy("u_nominal")
h_true = load_npy("h_true")
h_est = load_npy("h_estimated")
safety_margin = load_npy("safety_margin")
t_control = load_npy("control_time")
t_estimation = load_npy("state_estimation_time")
cbf_costmap = load_npy("cbf_costmap")
planner_costmap = load_npy("planner_costmap")
perception_costmap = load_npy("perception_magnitude_costmap")
noise_costmap = load_npy("noise_costmap")
sensor_positions = load_npy("sensor_positions")
path = load_npy("path")

# Define heatmap extent
rows, cols = cbf_costmap.shape
x = np.linspace(0, cols, cols)
y = np.linspace(0, rows, rows)
X, Y = np.meshgrid(x, y)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Simulation Results Viewer"),

    html.H2("Robot Position Over Time"),
    dcc.Graph(figure={
        "data": [
            go.Scatter(x=t_estimation, y=robot_pos[:, 0], mode="lines", name="True X"),
            go.Scatter(x=t_estimation, y=robot_pos_est[:, 0], mode="lines", name="Est X"),
            go.Scatter(x=t_estimation, y=robot_pos[:, 1], mode="lines", name="True Y"),
            go.Scatter(x=t_estimation, y=robot_pos_est[:, 1], mode="lines", name="Est Y"),
        ],
        "layout": go.Layout(title="Position Over Time", xaxis_title="Time [s]", yaxis_title="Position [m]")
    }),

    html.H2("Robot Velocity Over Time"),
    dcc.Graph(figure={
        "data": [
            go.Scatter(x=t_estimation, y=robot_vel[:, 0], mode="lines", name="Velocity X"),
            go.Scatter(x=t_estimation, y=robot_vel[:, 1], mode="lines", name="Velocity Y"),
        ],
        "layout": go.Layout(title="Velocity Over Time", xaxis_title="Time [s]", yaxis_title="Velocity [m/s]")
    }),

    html.H2("Control Inputs"),
    dcc.Graph(figure={
        "data": [
            go.Scatter(x=t_control, y=u_cbf[:, 0], mode="lines", name="CBF X"),
            go.Scatter(x=t_control, y=u_nominal[:, 0], mode="lines", name="Nominal X"),
            go.Scatter(x=t_control, y=u_cbf[:, 1], mode="lines", name="CBF Y"),
            go.Scatter(x=t_control, y=u_nominal[:, 1], mode="lines", name="Nominal Y"),
        ],
        "layout": go.Layout(title="Control Inputs", xaxis_title="Time [s]", yaxis_title="Control")
    }),

    html.H2("Safety Margin Over Time"),
    dcc.Graph(figure={
        "data": [go.Scatter(x=t_control, y=safety_margin[:, i], mode="lines", name=f"CBF {i}") for i in range(safety_margin.shape[1])],
        "layout": go.Layout(title="Safety Margin", xaxis_title="Time [s]", yaxis_title="Margin")
    }),

    html.H2("CBF Values"),
    dcc.Graph(figure={
        "data": [
            go.Scatter(x=t_control, y=h_true[:, i], mode="lines", name=f"True CBF {i}") for i in range(h_true.shape[1])
        ] + [
            go.Scatter(x=t_control, y=h_est[:, i], mode="lines", name=f"Estimated CBF {i}", line=dict(dash='dash')) for i in range(h_est.shape[1])
        ],
        "layout": go.Layout(title="CBF Values Over Time", xaxis_title="Time [s]", yaxis_title="CBF h")
    }),

    html.H2("Trajectory with Obstacles"),
    dcc.Graph(figure=go.Figure(
        data=[
            go.Scatter(x=robot_pos[:, 0], y=robot_pos[:, 1], mode="lines+markers", name="True Trajectory"),
            go.Scatter(x=robot_pos_est[:, 0], y=robot_pos_est[:, 1], mode="lines+markers", name="Estimated Trajectory", line=dict(dash="dash")),
            go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", name="Planned Path"),
            go.Scatter(x=sensor_positions[:, 0], y=sensor_positions[:, 1], mode="markers", name="Sensors", marker=dict(color="black", symbol="x")),
        ],
        layout=go.Layout(title="Robot Trajectories", xaxis_title="X [m]", yaxis_title="Y [m]", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    )),

    html.H2("CBF Costmap"),
    dcc.Graph(figure=go.Figure(
        data=go.Heatmap(z=cbf_costmap, x=x, y=y, colorscale="Plasma"),
        layout=go.Layout(title="CBF Costmap", xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    )),

    html.H2("Planner Costmap"),
    dcc.Graph(figure=go.Figure(
        data=go.Heatmap(z=planner_costmap, x=x, y=y, colorscale="Viridis"),
        layout=go.Layout(title="Planner Costmap", xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    )),

    html.H2("Perception Magnitude Costmap"),
    dcc.Graph(figure=go.Figure(
        data=go.Heatmap(z=perception_costmap, x=x, y=y, colorscale="Cividis"),
        layout=go.Layout(title="Perception Magnitude", xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    )),

    html.H2("Sensor Noise Costmap"),
    dcc.Graph(figure=go.Figure(
        data=go.Heatmap(z=noise_costmap, x=x, y=y, colorscale="Cividis"),
        layout=go.Layout(title="Sensor Noise", xaxis_title="X", yaxis_title="Y", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    )),
])

if __name__ == '__main__':
    app.run(debug=True)
