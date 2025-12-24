import pandas as pd
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
from dash import dcc, html

def create_layout(benchmark_options, run_data):
    initial_highlighted_run_id = run_data.iloc[0]["run_id"] if not run_data.empty else None
    initial_style_data_conditional = (
        [
            {
                "if": {"filter_query": f'{{run_id}} = "{initial_highlighted_run_id}"'},
                "border": "2px solid #4ADE80",
            }
        ]
        if initial_highlighted_run_id
        else []
    )

    # Add "run_id" column for selection tracking
    run_data["run_id"] = run_data.get("run_id", "")

    # Build DataTable columns dynamically from run_data
    def to_column_def(column_name):
        col_def = {
            "name": column_name.replace("_", " ").title(),
            "id": column_name,
        }
        if pd.api.types.is_numeric_dtype(run_data[column_name]):
            col_def["type"] = "numeric"
            if pd.api.types.is_float_dtype(run_data[column_name]):
                col_def["format"] = {"specifier": ".3f"}
        return col_def

    leaderboard_columns = [to_column_def(col) for col in run_data.columns]

    hidden_columns = [col for col in ["run_id"] if col in run_data.columns]

    return dbc.Container(
        [
            html.H1(
                "Magellan - Text Classification Leaderboard",
                className="text-3xl font-bold text-indigo-900 mb-6 mt-4 text-center",
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    html.H5(
                                        "Benchmark Selection",
                                        className="text-xl font-semibold text-indigo-900 mb-4",
                                    ),
                                    dcc.Dropdown(
                                        id="benchmark-dropdown",
                                        options=benchmark_options,
                                        value=benchmark_options[0]["value"],
                                        clearable=False,
                                        className="w-full text-sm rounded-md border-gray-300 focus:ring-cyan-500 focus:border-cyan-500",
                                    ),
                                ],
                                className="p-6 bg-slate-50 rounded-lg shadow-md mb-6",
                            ),
                        ],
                        width=12,
                        lg=4,
                        className="mx-auto",
                    )
                ],
                className="mb-6",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "Leaderboard",
                                className="text-2xl font-semibold text-indigo-900 mb-4",
                            ),
                            DataTable(
                                id="leaderboard-table",
                                columns=leaderboard_columns,
                                data=run_data[
                                    run_data["benchmark"] == benchmark_options[0]["value"]
                                ].to_dict("records"),
                                hidden_columns=hidden_columns,
                                style_table={
                                    "overflowX": "auto",
                                    "maxHeight": "40vh",
                                    "overflowY": "auto",
                                },
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "12px",
                                    "fontSize": "14px",
                                    "fontFamily": "sans-serif",
                                    "color": "#1F2937",
                                },
                                style_header={
                                    "backgroundColor": "#E2E8F0",
                                    "fontWeight": "600",
                                    "fontSize": "14px",
                                    "padding": "12px",
                                    "color": "#1F2937",
                                    "fontFamily": "sans-serif",
                                },
                                style_data={
                                    "borderBottom": "1px solid #E2E8F0",
                                    "transition": "background-color 0.2s",
                                },
                                style_data_conditional=initial_style_data_conditional + [
                                    {
                                        "if": {"state": "selected"},
                                        "border": "2px solid #4ADE80",
                                    },
                                    {
                                        "if": {"state": "active"},
                                        "backgroundColor": "#A7F3D0",
                                        "border": "1px solid #06B6D4",
                                    },
                                ],
                                css=[
                                    {
                                        "selector": ".dash-table-container tr:hover",
                                        "rule": "background-color: #E2E8F0; cursor: pointer;",
                                    },
                                    {
                                        "selector": ".dash-table-container tr.selected",
                                        "rule": "border: 2px solid #4ADE80 !important;",
                                    },
                                ],
                                page_size=6,
                                filter_action="native",
                                sort_action="native",
                                row_selectable="multi",
                                selected_rows=[],
                            ),
                        ],
                        width=12,
                    )
                ],
                className="mb-6",
            ),
            # Comparison Section
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.H3(
                        "Run Comparison",
                        className="text-2xl font-semibold text-indigo-900 mb-4 mt-6",
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Clear Comparison",
                                id="clear-compare-btn",
                                color="secondary",
                            ),
                        ], width=12),
                    ], className="mb-4"),
                    html.Hr(className="my-4"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5(
                                        "Classification Comparison",
                                        className="text-lg font-semibold text-indigo-900 mb-4",
                                    ),
                                ], width=8),
                                dbc.Col([
                                    html.Div([
                                        html.Label("Top N Subtypes", className="me-2"),
                                        dcc.Slider(
                                            id="top-n-subtypes-slider",
                                            min=2,
                                            max=40,
                                            step=1,
                                            value=20,
                                            marks={i: str(i) for i in [2, 5, 10, 20, 30, 40]},
                                            tooltip={"always_visible": False},
                                            className="w-100"
                                        ),
                                    ], className="d-flex align-items-center justify-content-end"),
                                ], width=4),
                                
                            ]),
                            html.Div(id="comparison-transition-matrix"),
                            html.Div(id="comparison-matrix-details"),
                        ])
                    ], className="p-6 bg-slate-50 rounded-lg shadow-md mb-6"),
                ], width=12),
            ]),
            # Separate Filters Card above Type Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(
                                "Filters",
                                className="text-lg font-semibold text-indigo-900 mb-3",
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Checklist(
                                        id="show-unsuccessful-checkbox",
                                        options=[
                                            {"label": "Show only unsuccessful predictions", "value": "fail"}
                                        ],
                                        value=[],
                                        switch=True,
                                        className="mb-3 text-sm text-indigo-900",
                                    ),
                                    width=4,
                                    className="d-flex align-items-center",
                                ),
                                dbc.Col([
                                    html.Div(
                                        "Confidence Range",
                                        className="text-sm text-indigo-900 mb-1",
                                    ),
                                    dcc.RangeSlider(
                                        id="confidence-threshold-slider",
                                        min=0,
                                        max=1,
                                        step=0.05,
                                        value=[0, 1],
                                        marks={0: "0.0", 0.5: "0.5", 1: "1.0"},
                                        tooltip={"always_visible": False},
                                    ),
                                ], width=8),
                            ], align="center"),
                        ])
                    ], className="p-6 bg-slate-50 rounded-lg shadow-md mb-4"),
                ], width=12),
            ]),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="type-histogram-content"),
                            html.Div(id="subtype-confusion-content"),
                            html.Div(id="subtype-f1-content"),
                            html.Div(id="datapoint-content"),
                        ],
                        width=12,
                    )
                ]
            ),
            dcc.Store(id="selected-true-type", data=None),
            dcc.Store(id="histogram-click-data", data=None),
            dcc.Store(id="f1-click-data", data=None),
            dcc.Store(id="highlighted-run-id", data=initial_highlighted_run_id),
            dcc.Store(id="selected-confusion-cell", data=None),
            dcc.Store(id="comparison-runs", data=[]),
            dcc.Store(id="comparison-data-store", data=None),
            dcc.Download(id="download-datapoint-csv"),
        ],
        fluid=False,
        className="p-4 w-4/5 mx-auto",
    )
