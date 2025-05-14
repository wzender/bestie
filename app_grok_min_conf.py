import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from uuid import uuid4
from dash import callback_context

# Sample data (replace with your actual data loading logic)
np.random.seed(42)
n_samples = 100
run_data = pd.DataFrame(
    {
        "run_id": [str(uuid4())[:8] for _ in range(n_samples)],
        "sweep_id": [f"sweep_{i%3}" for i in range(n_samples)],
        "model_params": [f"lr:0.{i%5}, layers:{(i%3)+1}" for i in range(n_samples)],
        "accuracy": np.random.uniform(0.7, 0.95, n_samples),
        "f1_score": np.random.uniform(0.65, 0.9, n_samples),
    }
)

# Define types and subtypes
all_types = [chr(65 + i) for i in range(20)]  # ["A", "B", "C", ..., "T"]
all_subtypes = [
    f"{t}{i}" for t in all_types for i in [1, 2]
]  # ["A1", "A2", "B1", "B2", ..., "T1", "T2"]

# Generate detailed data with type-subtype relationship
n_detailed = 2000
detailed_data = pd.DataFrame(
    {
        "run_id": [run_data["run_id"][i % len(run_data)] for i in range(n_detailed)],
        "text": [f"Sample text {i}" for i in range(n_detailed)],
        "true_type": np.random.choice(all_types, n_detailed),
    }
)
detailed_data["true_sub_type"] = detailed_data["true_type"].apply(
    lambda t: np.random.choice([f"{t}1", f"{t}2"])
)
detailed_data["pred_sub_type"] = detailed_data["true_type"].apply(
    lambda t: np.random.choice([f"{t}1", f"{t}2"])
)
detailed_data["correct"] = (
    detailed_data["true_sub_type"] == detailed_data["pred_sub_type"]
)

# Initialize Dash app with Bootstrap Lux theme and suppress callback exceptions
app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True
)

# Layout
app.layout = dbc.Container(
    [
        html.H1("Magellan - Text Classification Leaderboard", className="my-2"),
        # Leaderboard Table (upper third, full width)
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Leaderboard", className="mt-2 mb-2"),
                        dash_table.DataTable(
                            id="leaderboard-table",
                            columns=[
                                {"name": "Run ID", "id": "run_id"},
                                {"name": "Sweep ID", "id": "sweep_id"},
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
                            data=run_data.to_dict("records"),
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "33vh",
                                "overflowY": "auto",
                            },
                            style_cell={
                                "textAlign": "left",
                                "padding": "3px",
                                "fontSize": "12px",
                            },
                            style_header={
                                "backgroundColor": "rgb(230, 230, 230)",
                                "fontWeight": "bold",
                                "fontSize": "12px",
                            },
                            style_filter={"fontSize": "11px"},
                            row_selectable="single",
                            selected_rows=[0],
                            page_size=10,
                            filter_action="native",
                            sort_action="native",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-2",
        ),
        # Run Analysis (lower two-thirds)
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Run Analysis title and Show Mistakes switch in the same row
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H3(
                                        id="run-analysis-title",
                                        className="mt-2 mb-2",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Checklist(
                                        id="show-mistakes-switch",
                                        options=[
                                            {
                                                "label": "Show only mistakes",
                                                "value": 1,
                                            }
                                        ],
                                        value=[],
                                        switch=True,
                                        style={
                                            "fontSize": "0.9rem",
                                            "display": "inline-block",
                                        },
                                        labelStyle={"marginLeft": "5px"},
                                    ),
                                    width="auto",
                                ),
                            ],
                            className="g-2",
                            align="center",
                        ),
                        dcc.Tabs(
                            id="analysis-tabs",
                            value="error-analysis",
                            children=[
                                dcc.Tab(label="Error Analysis", value="error-analysis"),
                                dcc.Tab(label="Full Run Table", value="full-run"),
                                dcc.Tab(
                                    label="Subtype Analysis", value="subtype-analysis"
                                ),
                            ],
                            className="mb-2",
                        ),
                        html.Div(id="tab-content"),
                    ],
                    width=12,
                )
            ]
        ),
        # Store for selected true_type from histogram
        dcc.Store(id="selected-true-type", data=None),
        # Store for histogram click data
        dcc.Store(id="histogram-click-data", data=None),
    ],
    fluid=True,
)


# Callback to control visibility of show-mistakes-switch
@app.callback(
    Output("show-mistakes-switch", "style"), [Input("analysis-tabs", "value")]
)
def toggle_switch_visibility(tab):
    base_style = {"fontSize": "0.9rem", "display": "inline-block"}
    if tab in ["error-analysis", "subtype-analysis"]:
        return base_style
    return {**base_style, "display": "none"}


# Callback to update Run Analysis title
@app.callback(
    Output("run-analysis-title", "children"),
    [Input("leaderboard-table", "selected_rows")],
)
def update_run_analysis_title(selected_rows):
    if not selected_rows:
        return "Run Analysis"
    selected_run_id = run_data.iloc[sorted(selected_rows)[0]]["run_id"]
    return f"{selected_run_id} Run Analysis"


# Callback to capture type-histogram click data
@app.callback(
    Output("histogram-click-data", "data"),
    [
        Input("type-histogram", "clickData"),
        Input("leaderboard-table", "selected_rows"),
        Input("analysis-tabs", "value"),
    ],
    prevent_initial_call=True,
)
def capture_histogram_click_data(click_data, selected_rows, tab):
    ctx = callback_context
    if not ctx.triggered or not selected_rows or tab != "error-analysis":
        return None

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "type-histogram" and click_data and "points" in click_data:
        return click_data

    return None


# Callback to render tab content
@app.callback(
    [Output("tab-content", "children"), Output("selected-true-type", "data")],
    [
        Input("analysis-tabs", "value"),
        Input("leaderboard-table", "selected_rows"),
        Input("histogram-click-data", "data"),
        Input("show-mistakes-switch", "value"),
    ],
    prevent_initial_call=True,
)
def render_tab_content(tab, selected_rows, histogram_click_data, show_mistakes):
    if not selected_rows:
        return html.Div("Select a run from the leaderboard"), None

    selected_run_id = run_data.iloc[sorted(selected_rows)[0]]["run_id"]
    run_data_filtered = detailed_data[detailed_data["run_id"] == selected_run_id]

    # Apply mistakes filter if switch is on
    if show_mistakes and show_mistakes == [1]:
        run_data_filtered = run_data_filtered[run_data_filtered["correct"] == False]

    # Handle histogram click to update selected_true_type, reset on run change
    selected_true_type = None
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    if (
        triggered_id == "histogram-click-data"
        and histogram_click_data
        and "points" in histogram_click_data
    ):
        new_true_type = histogram_click_data["points"][0]["x"]
        if new_true_type in all_types:
            selected_true_type = new_true_type

    # Type Histogram (always show all types, highlight selected)
    type_counts = (
        run_data_filtered["true_type"]
        .value_counts()
        .reindex(all_types, fill_value=0)
        .reset_index()
    )
    type_counts.columns = ["true_type", "count"]
    type_fig = px.bar(
        type_counts, x="true_type", y="count", title="True Type Distribution"
    )
    type_fig.update_traces(
        marker=dict(opacity=0.4),
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.4)),
    )
    if selected_true_type:
        selected_index = (
            all_types.index(selected_true_type)
            if selected_true_type in all_types
            else None
        )
        type_fig.update_traces(
            selectedpoints=[selected_index] if selected_index is not None else []
        )
    type_fig.update_layout(
        clickmode="event+select",
        dragmode=False,
        xaxis={"fixedrange": True, "tickangle": 45},
        yaxis={"fixedrange": True},
        margin={"b": 120},
    )

    # Apply true_type filter for Confusion Matrix and Datapoint Table
    cm_table_data = run_data_filtered
    if selected_true_type and selected_true_type in all_types:
        cm_table_data = run_data_filtered[
            run_data_filtered["true_type"] == selected_true_type
        ]

    if tab == "error-analysis":
        # Initialize content components
        type_histogram_content = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Type Histogram", className="mb-1"),
                            dcc.Graph(
                                id="type-histogram",
                                figure=type_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                            ),
                        ],
                        width=12,
                    )
                ],
                className="mb-2",
            )
        ]

        # Datapoint Table setup
        datapoint_columns = [
            {"name": "Text", "id": "text"},
            {"name": "True Type", "id": "true_type"},
            {"name": "True Subtype", "id": "true_sub_type"},
            {"name": "Pred Subtype", "id": "pred_sub_type"},
            {"name": "Correct", "id": "correct"},
        ]

        # Check if a type is selected for Confusion Matrix
        if not selected_true_type:
            content = type_histogram_content + [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Subtype Confusion Matrix", className="mb-1"),
                                dbc.Alert(
                                    "Please select a type from the Type Histogram to view the Subtype Confusion Matrix",
                                    color="info",
                                    className="text-center",
                                ),
                            ],
                            md=6,
                            className="pe-0",
                        ),
                        dbc.Col(
                            [
                                html.H4("Datapoint Table", className="mb-1"),
                                dash_table.DataTable(
                                    id="datapoint-table",
                                    columns=datapoint_columns,
                                    data=[],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "3px",
                                        "fontSize": "12px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                        "fontSize": "12px",
                                    },
                                    style_filter={"fontSize": "11px"},
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native",
                                ),
                            ],
                            md=6,
                            className="ps-1",
                        ),
                    ]
                )
            ]
            return content, None

        # Subtype Confusion Matrix
        cm = pd.crosstab(
            cm_table_data["true_sub_type"],
            cm_table_data["pred_sub_type"],
            rownames=["True Subtype"],
            colnames=["Predicted Subtype"],
        )
        if cm.empty:
            content = type_histogram_content + [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Subtype Confusion Matrix", className="mb-1"),
                                html.Div("No predictions for selected type's subtypes"),
                            ],
                            md=6,
                            className="pe-0",
                        ),
                        dbc.Col(
                            [
                                html.H4("Datapoint Table", className="mb-1"),
                                dash_table.DataTable(
                                    id="datapoint-table",
                                    columns=datapoint_columns,
                                    data=[],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "3px",
                                        "fontSize": "12px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                        "fontSize": "12px",
                                    },
                                    style_filter={"fontSize": "11px"},
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native",
                                ),
                            ],
                            md=6,
                            className="ps-1",
                        ),
                    ]
                )
            ]
            return content, selected_true_type

        z = cm.values
        row_sums = cm.sum(axis=1)
        z_text = []
        for i, row in enumerate(z):
            row_text = []
            row_sum = row_sums.iloc[i] if i < len(row_sums) else 0
            for count in row:
                percentage = (count / row_sum * 100) if row_sum > 0 else 0.0
                row_text.append(f"{count}<br>({percentage:.1f}%)")
            z_text.append(row_text)
        try:
            cm_fig = ff.create_annotated_heatmap(
                z,
                x=cm.columns.tolist(),
                y=cm.index.tolist(),
                annotation_text=z_text,
                colorscale="Blues",
                showscale=False,
            )
            cm_fig.update_layout(
                title="Subtype Confusion Matrix",
                xaxis_title="Predicted Subtype",
                yaxis_title="True Subtype",
                clickmode="event+select",
                dragmode=False,
                xaxis={"fixedrange": True},
                yaxis={"fixedrange": True},
                width=None,
                height=400,
                font=dict(size=10),
                margin=dict(l=100, r=20, t=100, b=50),
            )
        except Exception as e:
            content = type_histogram_content + [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Subtype Confusion Matrix", className="mb-1"),
                                html.Div(f"Error creating confusion matrix: {str(e)}"),
                            ],
                            md=6,
                            className="pe-0",
                        ),
                        dbc.Col(
                            [
                                html.H4("Datapoint Table", className="mb-1"),
                                dash_table.DataTable(
                                    id="datapoint-table",
                                    columns=datapoint_columns,
                                    data=[],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "3px",
                                        "fontSize": "12px",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                        "fontSize": "12px",
                                    },
                                    style_filter={"fontSize": "11px"},
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native",
                                ),
                            ],
                            md=6,
                            className="ps-1",
                        ),
                    ]
                )
            ]
            return content, selected_true_type

        content = type_histogram_content + [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Subtype Confusion Matrix", className="mb-1"),
                            dcc.Graph(
                                id="confusion-matrix",
                                figure=cm_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                            ),
                        ],
                        md=6,
                        className="pe-0",
                    ),
                    dbc.Col(
                        [
                            html.H4("Datapoint Table", className="mb-1"),
                            dash_table.DataTable(
                                id="datapoint-table",
                                columns=datapoint_columns,
                                data=cm_table_data.to_dict("records"),
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "3px",
                                    "fontSize": "12px",
                                },
                                style_header={
                                    "backgroundColor": "rgb(230, 230, 230)",
                                    "fontWeight": "bold",
                                    "fontSize": "12px",
                                },
                                style_filter={"fontSize": "11px"},
                                page_size=10,
                                filter_action="native",
                                sort_action="native",
                            ),
                        ],
                        md=6,
                        className="ps-1",
                    ),
                ]
            )
        ]
        return content, selected_true_type

    elif tab == "full-run":
        return (
            dash_table.DataTable(
                columns=[
                    {"name": "Text", "id": "text"},
                    {"name": "True Type", "id": "true_type"},
                    {"name": "True Subtype", "id": "true_sub_type"},
                    {"name": "Pred Subtype", "id": "pred_sub_type"},
                    {"name": "Correct", "id": "correct"},
                ],
                data=run_data_filtered.to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "3px", "fontSize": "12px"},
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                    "fontSize": "12px",
                },
                style_filter={"fontSize": "11px"},
                page_size=15,
                filter_action="native",
                sort_action="native",
            ),
            None,
        )

    elif tab == "subtype-analysis":
        # Calculate accuracy for each subtype
        subtype_stats = run_data_filtered.groupby("true_sub_type").agg(
            correct_count=("correct", "sum"), total_count=("true_sub_type", "count")
        )
        subtype_stats["accuracy"] = (
            subtype_stats["correct_count"] / subtype_stats["total_count"]
        )

        # Get top 5 worst subtypes (lowest accuracy)
        top_5_worst_subtypes = subtype_stats.nsmallest(5, "accuracy").index.tolist()

        if not top_5_worst_subtypes:
            return html.Div("No subtype data available for analysis"), None

        # Filter data for worst subtypes
        filtered_df = run_data_filtered[
            run_data_filtered["true_sub_type"].isin(top_5_worst_subtypes)
        ]

        # Create confusion matrix
        cm = pd.crosstab(
            filtered_df["true_sub_type"],
            filtered_df["pred_sub_type"],
            rownames=["True Subtype"],
            colnames=["Predicted Subtype"],
        )
        if cm.empty:
            return html.Div("No subtype data available for analysis"), None

        z = cm.values
        row_sums = cm.sum(axis=1)
        z_text = []
        for i, row in enumerate(z):
            row_text = []
            row_sum = row_sums.iloc[i] if i < len(row_sums) else 0
            for count in row:
                percentage = (count / row_sum * 100) if row_sum > 0 else 0.0
                row_text.append(f"{count}<br>({percentage:.1f}%)")
            z_text.append(row_text)

        try:
            cm_fig = ff.create_annotated_heatmap(
                z,
                x=cm.columns.tolist(),
                y=cm.index.tolist(),
                annotation_text=z_text,
                colorscale="Blues",
                showscale=False,
            )
            cm_fig.update_layout(
                title="Worst Subtypes Confusion Matrix",
                xaxis_title="Predicted Subtype",
                yaxis_title="True Subtype",
                clickmode="event+select",
                dragmode=False,
                xaxis={"fixedrange": True},
                yaxis={"fixedrange": True},
                width=None,
                height=400,
                font=dict(size=10),
                margin=dict(l=100, r=20, t=100, b=50),
            )
        except Exception as e:
            return html.Div(f"Error creating confusion matrix: {str(e)}"), None

        content = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                "Worst Subtypes Confusion Matrix", className="mb-1"
                            ),
                            dcc.Graph(
                                id="worst-subtypes-confusion-matrix",
                                figure=cm_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                            ),
                        ],
                        width=12,
                    )
                ]
            )
        ]
        return content, None


# Callback to update datapoint table based on confusion matrix clicks
@app.callback(
    Output("datapoint-table", "data"),
    [
        Input("confusion-matrix", "clickData"),
        Input("leaderboard-table", "selected_rows"),
        Input("selected-true-type", "data"),
        Input("show-mistakes-switch", "value"),
    ],
    prevent_initial_call=True,
)
def update_datapoint_table(
    confusion_click_data, selected_rows, selected_true_type, show_mistakes
):
    if not selected_rows:
        return []

    selected_run_id = run_data.iloc[sorted(selected_rows)[0]]["run_id"]
    run_data_filtered = detailed_data[detailed_data["run_id"] == selected_run_id]

    # Apply mistakes filter if switch is on
    if show_mistakes and show_mistakes == [1]:
        run_data_filtered = run_data_filtered[run_data_filtered["correct"] == False]

    # Apply true_type filter if selected
    if selected_true_type and selected_true_type in all_types:
        run_data_filtered = run_data_filtered[
            run_data_filtered["true_type"] == selected_true_type
        ]
    else:
        return []

    if confusion_click_data and "points" in confusion_click_data:
        point = confusion_click_data["points"][0]
        true_subtype = point["y"]
        pred_subtype = point["x"]
        filtered = run_data_filtered[
            (run_data_filtered["true_sub_type"] == true_subtype)
            & (run_data_filtered["pred_sub_type"] == pred_subtype)
        ]
        return filtered.to_dict("records")

    return run_data_filtered.to_dict("records")


if __name__ == "__main__":
    app.run(debug=False)
