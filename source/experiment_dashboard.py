import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output
from validation import ValidateExperiments


class DashReport:
    """
    Interactive experiment report with Dash.

    - Tab for summary results (summary + full table).
    - One tab per experiment:
        * Select seed
        * Select variables to plot over time
        * Interactive time-series plot across robots
        * Additional per-robot plots for all seeds
    """

    def __init__(
        self, df_summary, df_full, dfs_timeseries, title="Experiment Dashboard"
    ):
        """
        Parameters
        ----------
        df_summary : pd.DataFrame
            Summary table (overall results per experiment/robot).
        df_full : pd.DataFrame
            Full runs table (all seeds, aggregated metrics).
        dfs_timeseries : dict
            Nested dict {exp: {robot: {seed: DataFrame}}}, each DataFrame = time series.
            DataFrame must have columns: ["time", <variables>].
        title : str
            Dashboard title.
        """
        self.df_summary = df_summary
        self.df_full = df_full
        self.dfs_timeseries = dfs_timeseries
        self.title = title

        self.app = Dash(__name__, title=self.title)
        self._build_layout()
        self._register_callbacks()

    def _build_layout(self):
        tabs = []

        # --- Summary tab ---
        exp_tables = []
        for exp in sorted(self.df_full["experiment"].unique()):
            df_exp = self.df_full[self.df_full["experiment"] == exp]
            exp_tables.append(
                html.Div([
                    html.H3(f"Full Runs Table — {exp}"),
                    dash_table.DataTable(
                        id=f"full-table-{exp}",
                        data=df_exp.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in df_exp.columns],
                        sort_action="native",
                        filter_action="native",
                        page_size=25,
                        style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
                        style_cell={"fontFamily": "monospace", "fontSize": 13},
                        style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
                    ),
                ], style={"marginBottom": "30px"})
            )

        tabs.append(
            dcc.Tab(
                label="Summary",
                children=[
                    html.H2("Summary Table (grouped by experiment × robot)"),
                    dash_table.DataTable(
                        id="summary-table",
                        data=self.df_summary.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in self.df_summary.columns],
                        sort_action="native",
                        filter_action="native",
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontFamily": "monospace", "fontSize": 13},
                        style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
                    ),
                    html.H2("Full Runs Tables (per experiment)"),
                    *exp_tables,   # all experiment-specific tables stacked
                ],
            )
        )

        # --- One tab per experiment ---
        for exp in sorted(self.dfs_timeseries.keys()):
            # Collect seeds
            seeds = []
            for robot, seeds_dict in self.dfs_timeseries[exp].items():
                seeds.extend(seeds_dict.keys())
            seeds = sorted(set(seeds))

            # Pick one df to get variable names
            sample_robot = next(iter(self.dfs_timeseries[exp]))
            sample_seed = next(iter(self.dfs_timeseries[exp][sample_robot]))
            sample_df = self.dfs_timeseries[exp][sample_robot][sample_seed]
            variables = [c for c in sample_df.columns if c != "time"]

            # Create robot-specific graph placeholders
            robot_graphs = []
            for robot in sorted(self.dfs_timeseries[exp].keys()):
                robot_graphs.append(
                    html.Div(
                        [
                            html.H4(f"Robot {robot}"),
                            dcc.Graph(
                                id=f"timeseries-{exp}-{robot}", style={"height": "50vh"}
                            ),
                        ]
                    )
                )

            tabs.append(
                dcc.Tab(
                    label=f"Experiment {exp}",
                    children=[
                        html.H2(f"Experiment {exp}"),
                        html.Label("Seed"),
                        dcc.Dropdown(
                            id=f"seed-dropdown-{exp}",
                            options=[{"label": str(s), "value": s} for s in seeds],
                            value=seeds[0],
                            clearable=False,
                        ),
                        html.Label("Variables"),
                        dcc.Dropdown(
                            id=f"var-dropdown-{exp}",
                            options=[{"label": v, "value": v} for v in variables],
                            value=[variables[0]],
                            multi=True,
                        ),
                        html.H3("Combined Time-Series Plot"),
                        dcc.Graph(id=f"timeseries-{exp}", style={"height": "70vh"}),
                        html.H3("Per-Robot Time-Series Plots"),
                        html.Div(robot_graphs),
                    ],
                )
            )

        self.app.layout = html.Div(
            [html.H1(self.title), dcc.Tabs(tabs)], style={"padding": "20px"}
        )

    def _register_callbacks(self):
        # --- Callbacks for each experiment ---
        for exp in sorted(self.dfs_timeseries.keys()):

            @self.app.callback(
                Output(f"timeseries-{exp}", "figure"),
                [
                    Input(f"seed-dropdown-{exp}", "value"),
                    Input(f"var-dropdown-{exp}", "value"),
                ],
            )
            def update_timeseries(seed, variables, exp=exp):
                fig = go.Figure()
                if not variables:
                    return fig
                for robot, seeds_dict in self.dfs_timeseries[exp].items():
                    if seed not in seeds_dict:
                        continue
                    df_seed = seeds_dict[seed]
                    for var in variables:
                        if var in df_seed.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=df_seed["time"],
                                    y=df_seed[var],
                                    mode="lines",
                                    name=f"{robot} - {var}",
                                )
                            )
                fig.update_layout(
                    title=f"{exp} — Seed {seed}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    legend_title="Robot / Variable",
                )
                return fig

            # Per-robot plots: one per robot for this exp
            for robot in sorted(self.dfs_timeseries[exp].keys()):

                @self.app.callback(
                    Output(f"timeseries-{exp}-{robot}", "figure"),
                    [Input(f"var-dropdown-{exp}", "value")],
                )
                def update_robot_timeseries(variables, exp=exp, robot=robot):
                    fig = go.Figure()
                    if not variables:
                        return fig
                    seeds_dict = self.dfs_timeseries[exp][robot]
                    for seed, df_seed in seeds_dict.items():
                        for var in variables:
                            if var in df_seed.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_seed["time"],
                                        y=df_seed[var],
                                        mode="lines",
                                        name=f"Seed {seed} - {var}",
                                    )
                                )
                    fig.update_layout(
                        title=f"{exp} — Robot {robot}",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        legend_title="Seed / Variable",
                    )
                    return fig

    def run(self, port=8050, debug=False):
        self.app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    # all old experiments
    # df_summary = pd.read_csv("./runs/validation/summary.csv")
    # df_full = pd.read_csv("./runs/validation/full_table.csv")

    # gaps experiment -> new relative degree
    df_summary = pd.read_csv("./runs/validation/validation_gaps_exp/summary.csv")
    df_full = pd.read_csv("./runs/validation/validation_gaps_exp/full_table.csv")
    validate_cfg = {
        "exp_folders": [
            # "./runs/fake_exp",
            # "./runs/fabric_exp",
            # "./runs/cluttered_exp",
            "./runs/gaps_exp",
        ],
        "exp_colors": ["k", "g", "r", "b"],
        "val_dir": "./runs/validation/validation_gaps_exp",
    }
    validate_experiments = ValidateExperiments(validate_cfg=validate_cfg)
    # validate_experiments.generate_report()

    # Start interactive dashboard
    report = DashReport(
        df_summary=df_summary,
        df_full=df_full,
        dfs_timeseries=validate_experiments.dfs,
        title="Experiments dashboard",
    )
    report.run(port=8050)
