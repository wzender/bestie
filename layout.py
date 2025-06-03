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
                                columns=[
                                    {"name": "Model Name", "id": "model_name"},
                                    {"name": "Model Parameters", "id": "model_params"},
                                    {
                                        "name": "Accuracy",
                                        "id": "accuracy",
                                        "type": "numeric",
                                        "format": {"specifier": ".3f"},
                                    },
                                    {
                                        "name": "F1 Score",
                                        "id": "f1_score",
                                        "type": "numeric",
                                        "format": {"specifier": ".3f"},
                                    },
                                ],
                                data=run_data[
                                    run_data["benchmark"] == benchmark_options[0]["value"]
                                ].to_dict("records"),
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
                            ),
                        ],
                        width=12,
                    )
                ],
                className="mb-6",
            ),
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
            dcc.Download(id="download-datapoint-csv"),
        ],
        fluid=False,
        className="p-4 w-4/5 mx-auto",
    )