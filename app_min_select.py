import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
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
        "model_name": [f"llama-{i%3 + 1}" for i in range(n_samples)],
        "benchmark": [f"subtype_00_{i%5:02d}" for i in range(n_samples)],
        "model_params": [f"lr:0.{i%5}, layers:{(i%3)+1}" for i in range(n_samples)],
        "accuracy": np.random.uniform(0.7, 0.95, n_samples),
        "f1_score": np.random.uniform(0.65, 0.9, n_samples),
    }
)

# Test run data
test_run_data = pd.DataFrame(
    {
        "Text": [f"Text{i+1}" for i in range(20)],
        "true_type": ["Perfect"] * 5 + ["WorstMin"] * 5 + ["Medium"] * 10,
        "true_sub_type": ["Perfect1"] * 5
        + ["WorstMin1"] * 3
        + ["WorstMin2"] * 2
        + ["Medium1"] * 5
        + ["Medium2"] * 5,
        "pred_sub_type": ["Perfect1"] * 5
        + ["Else"] * 3
        + ["Else", "WorstMin2"]
        + ["Medium1"] * 4
        + ["Medium2"]
        + ["Medium2"] * 3
        + ["Medium1"] * 2,
        "correct": [True] * 5
        + [False] * 3
        + [False, True]
        + [True] * 4
        + [False]
        + [True] * 3
        + [False] * 2,
    }
)
test_run_id = "test_run"
test_run_data["run_id"] = test_run_id

# Add pred_unk_subtype column to test_run_data
unk_subtypes = [f"UNK{i}" for i in range(1, 11)] + [None]
np.random.seed(43)
test_run_data["pred_unk_subtype"] = np.random.choice(
    unk_subtypes, size=len(test_run_data), p=[0.09] * 10 + [0.1]
)

# Add test run to run_data
test_run_summary = pd.DataFrame(
    {
        "run_id": [test_run_id],
        "sweep_id": ["test_sweep"],
        "model_name": ["llama-1"],
        "benchmark": ["subtype_00_01"],
        "model_params": ["test_params"],
        "accuracy": [test_run_data["correct"].mean()],
        "f1_score": [0.65],
    }
)
run_data = pd.concat([run_data, test_run_summary], ignore_index=True)

# Define types and subtypes
all_types = [chr(65 + i) for i in range(20)] + ["Perfect", "WorstMin", "Medium"]
all_subtypes = [f"{t}{i}" for t in all_types[:20] for i in [1, 2]] + [
    "Perfect1",
    "WorstMin1",
    "WorstMin2",
    "Medium1",
    "Medium2",
]
all_unk_subtypes = [f"UNK{i}" for i in range(1, 11)] + [None]

# Generate detailed data with type-subtype relationship
n_detailed = 2000
detailed_data = pd.DataFrame(
    {
        "run_id": [
            run_data["run_id"][i % (len(run_data) - 1)] for i in range(n_detailed)
        ],
        "text": [f"Sample text {i}" for i in range(n_detailed)],
        "true_type": np.random.choice(all_types[:-3], n_detailed),
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
detailed_data["pred_unk_subtype"] = None

# Append test_run_data to detailed_data
detailed_data = pd.concat([detailed_data, test_run_data], ignore_index=True)

# Get unique benchmarks for dropdown
unique_benchmarks = sorted(run_data["benchmark"].unique())
benchmark_options = [{"label": bm, "value": bm} for bm in unique_benchmarks]

# Initialize highlighted_run_id and style_data_conditional
initial_highlighted_run_id = run_data.iloc[0]["run_id"] if not run_data.empty else None
initial_style_data_conditional = (
    [
        {
            "if": {"filter_query": f'{{run_id}} = "{initial_highlighted_run_id}"'},
            "backgroundColor": "rgba(0, 116, 217, 0.1)",
        }
    ]
    if initial_highlighted_run_id
    else []
)

# Initialize Dash app with Bootstrap Lux theme
app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True
)

# Add custom CSS for styling
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
custom_css = """
.benchmark-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.benchmark-card h5 {
    margin-bottom: 8px;
    font-weight: 500;
}
.dropdown-container {
    max-width: 300px; /* Limit dropdown width */
}
.dcc-dropdown {
    font-size: 0.85rem;
    color: #333;
}
"""
app.css.append_css({"external_url": f"data:text/css,{custom_css}"})

# Layout
app.layout = dbc.Container(
    [
        html.H1("Magellan - Text Classification Leaderboard", className="my-2"),
        # Benchmark Selection Dropdown
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                html.H5("Select Benchmark", className="card-title"),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="benchmark-dropdown",
                                            options=benchmark_options,
                                            value=unique_benchmarks[
                                                0
                                            ],  # Default: first benchmark
                                            clearable=False,  # Prevent clearing selection
                                            className="dcc-dropdown",
                                        ),
                                    ],
                                    className="dropdown-container",
                                ),
                            ],
                            className="benchmark-card mb-3",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        # Leaderboard Table (upper third, full width)
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Leaderboard", className="mt-2 mb-2"),
                        dash_table.DataTable(
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
                                run_data["benchmark"] == unique_benchmarks[0]
                            ].to_dict("records"),
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
                            page_size=6,
                            filter_action="native",
                            sort_action="native",
                            style_data_conditional=initial_style_data_conditional,
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
                        # Subtype Confusion by Type Panel
                        html.Div(id="subtype-confusion-content"),
                    ],
                    width=12,
                )
            ]
        ),
        # Store for selected true_type from histogram
        dcc.Store(id="selected-true-type", data=None),
        # Store for histogram click data
        dcc.Store(id="histogram-click-data", data=None),
        # Store for highlighted run_id
        dcc.Store(id="highlighted-run-id", data=initial_highlighted_run_id),
    ],
    fluid=True,
)


# Combined callback to handle benchmark selection and table row clicks
@app.callback(
    [
        Output("leaderboard-table", "data"),
        Output("highlighted-run-id", "data"),
        Output("leaderboard-table", "style_data_conditional"),
    ],
    [
        Input("benchmark-dropdown", "value"),
        Input("leaderboard-table", "active_cell"),
    ],
    [
        State("leaderboard-table", "derived_viewport_data"),
        State("leaderboard-table", "page_current"),
        State("leaderboard-table", "page_size"),
    ],
)
def update_leaderboard_and_highlight(
    selected_benchmark, active_cell, viewport_data, page_current, page_size
):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "benchmark-dropdown":
        if not selected_benchmark:
            return [], None, []

        # Filter data for the selected benchmark
        df = run_data[run_data["benchmark"] == selected_benchmark]
        data = df.to_dict("records")

        # If no data, return empty results
        if not data:
            return [], None, []

        # Highlight the first row
        highlighted_run_id = data[0]["run_id"]
        style_data_conditional = [
            {
                "if": {"filter_query": f'{{run_id}} = "{highlighted_run_id}"'},
                "backgroundColor": "rgba(0, 116, 217, 0.1)",
            }
        ]
        return data, highlighted_run_id, style_data_conditional

    elif triggered_id == "leaderboard-table" and active_cell and viewport_data:
        row_index = active_cell["row"]
        highlighted_run_id = viewport_data[row_index]["run_id"]
        style_data_conditional = [
            {
                "if": {"filter_query": f'{{run_id}} = "{highlighted_run_id}"'},
                "backgroundColor": "rgba(0, 116, 217, 0.1)",
            }
        ]
        # Keep the current table data
        return dash.no_update, highlighted_run_id, style_data_conditional

    return dash.no_update, dash.no_update, dash.no_update


# Callback to capture type-histogram click data
@app.callback(
    Output("histogram-click-data", "data"),
    [
        Input("type-histogram", "clickData"),
        Input("highlighted-run-id", "data"),
    ],
    prevent_initial_call=True,
)
def capture_histogram_click_data(click_data, highlighted_run_id):
    ctx = callback_context
    if not ctx.triggered or not highlighted_run_id:
        return None

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "type-histogram" and click_data and "points" in click_data:
        return click_data
    return None


# Callback to render Subtype Confusion by Type content
@app.callback(
    [
        Output("subtype-confusion-content", "children"),
        Output("selected-true-type", "data"),
    ],
    [
        Input("highlighted-run-id", "data"),
        Input("histogram-click-data", "data"),
    ],
    prevent_initial_call=True,
)
def render_subtype_confusion_content(highlighted_run_id, histogram_click_data):
    if not highlighted_run_id:
        return html.Div("Highlight a run from the leaderboard"), None

    # Filter data for the selected run
    run_data_filtered = detailed_data[detailed_data["run_id"] == highlighted_run_id]

    # Handle histogram click to update selected_true_type
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

    # Type Histogram
    type_counts = (
        run_data_filtered["true_type"]
        .value_counts()
        .reindex(all_types, fill_value=0)
        .reset_index()
    )
    type_counts.columns = ["true_type", "count"]

    # Check for empty Type Histogram
    if run_data_filtered["true_type"].value_counts().sum() == 0:
        content = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Type Histogram", className="mb-1"),
                            dbc.Alert(
                                "No True Type data available for this run",
                                color="info",
                                className="text-center",
                            ),
                        ],
                        width=12,
                    )
                ]
            )
        ]
        return content, None

    # Create Type Histogram
    type_fig = px.bar(
        type_counts,
        x="true_type",
        y="count",
        title="True Type Distribution",
        text="count",
    )
    type_fig.update_traces(
        marker=dict(opacity=0.4),
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.4)),
        textposition="outside",
        textfont=dict(size=10),
        texttemplate="%{text}",
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
        margin={"b": 120, "t": 60},
    )

    # Type Histogram content
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
        {"name": "Pred Unk Subtype", "id": "pred_unk_subtype"},
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
                        width=12,
                    )
                ]
            )
        ]
        return content, None

    # Filter data for the selected true_type in the run
    cm_table_data = run_data_filtered[
        run_data_filtered["true_type"] == selected_true_type
    ]

    # Get true and predicted subtypes for the selected true_type
    true_subtypes = cm_table_data["true_sub_type"].dropna().unique()
    pred_subtypes = cm_table_data["pred_sub_type"].dropna().unique()

    # Combine true and predicted subtypes to define the NxN matrix
    all_subtypes_in_type = sorted(list(set(true_subtypes).union(set(pred_subtypes))))

    # Check if there are any subtypes to display
    if not all_subtypes_in_type:
        content = type_histogram_content + [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Subtype Confusion Matrix", className="mb-1"),
                            html.Div(
                                "No subtypes available for selected type in this run"
                            ),
                        ],
                        width=12,
                    )
                ]
            )
        ]
        return content, selected_true_type

    # Create NxN confusion matrix for the selected true_type
    cm = pd.crosstab(
        cm_table_data["true_sub_type"],
        cm_table_data["pred_sub_type"],
        rownames=["True Subtype"],
        colnames=["Predicted Subtype"],
        dropna=False,
    ).reindex(index=all_subtypes_in_type, columns=all_subtypes_in_type, fill_value=0)

    if cm.empty:
        content = type_histogram_content + [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Subtype Confusion Matrix", className="mb-1"),
                            html.Div("No predictions for selected type's subtypes"),
                        ],
                        width=12,
                    )
                ]
            )
        ]
        return content, selected_true_type

    # Prepare heatmap data
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
            xaxis={"fixedrange": True, "tickangle": 45},
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
                        width=12,
                    )
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
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
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
                    width=12,
                )
            ]
        ),
    ]
    return content, selected_true_type


# Callback to update datapoint table based on confusion matrix clicks
@app.callback(
    Output("datapoint-table", "data"),
    [
        Input("confusion-matrix", "clickData"),
        Input("highlighted-run-id", "data"),
        Input("selected-true-type", "data"),
    ],
    prevent_initial_call=True,
)
def update_datapoint_table(
    confusion_click_data, highlighted_run_id, selected_true_type
):
    if not highlighted_run_id:
        return []

    run_data_filtered = detailed_data[detailed_data["run_id"] == highlighted_run_id]

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
    onrender_port = 10000
    app.run(debug=False, port=onrender_port, host="0.0.0.0")
