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
        "model_params": [f"lr:0.{i%5}, layers:{(i%3)+1}" for i in range(n_samples)],
        "accuracy": np.random.uniform(0.7, 0.95, n_samples),
        "f1_score": np.random.uniform(0.65, 0.9, n_samples),
    }
)

# Define all possible subtypes and types
all_subtypes = ["Sub1", "Sub2", "Sub3", "Sub4"]
all_types = [f"Type{i}" for i in range(1, 21)]  # 20 different types

detailed_data = pd.DataFrame(
    {
        "run_id": [run_data["run_id"][i % len(run_data)] for i in range(500)],
        "text": [f"Sample text {i}" for i in range(500)],
        "true_type": np.random.choice(all_types, 500),
        "true_sub_type": np.random.choice(all_subtypes, 500),
        "pred_sub_type": np.random.choice(all_subtypes, 500),
    }
)
detailed_data["correct"] = (
    detailed_data["true_sub_type"] == detailed_data["pred_sub_type"]
)

# Initialize Dash app with Bootstrap Lux theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Layout
app.layout = dbc.Container(
    [
        html.H1("Magellan - Text Classification Leaderboard", className="my-2"),
        dbc.Row(
            [
                # Leaderboard Table (1/3 width)
                dbc.Col(
                    [
                        html.H3("Leaderboard", className="mt-2 mb-2"),
                        dash_table.DataTable(
                            id="leaderboard-table",
                            columns=[
                                {"name": "Run ID", "id": "run_id"},
                                {"name": "Sweep ID", "id": "sweep_id"},
                                {"name": "Params", "id": "model_params"},
                                {
                                    "name": "Acc",
                                    "id": "accuracy",
                                    "type": "numeric",
                                    "format": {"specifier": ".3f"},
                                },
                                {
                                    "name": "F1",
                                    "id": "f1_score",
                                    "type": "numeric",
                                    "format": {"specifier": ".3f"},
                                },
                            ],
                            data=run_data.to_dict("records"),
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
                            row_selectable="single",
                            selected_rows=[0],
                            page_size=10,
                        ),
                    ],
                    md=4,
                    className="pe-2",
                ),
                # Run Analysis (2/3 width)
                dbc.Col(
                    [
                        html.H3("Run Analysis", className="mt-2 mb-2"),
                        dcc.Tabs(
                            id="analysis-tabs",
                            value="error-analysis",
                            children=[
                                dcc.Tab(label="Error Analysis", value="error-analysis"),
                                dcc.Tab(label="Full Run Table", value="full-run"),
                            ],
                            className="mb-2",
                        ),
                        html.Div(id="tab-content"),
                    ],
                    md=8,
                    className="ps-2",
                ),
            ]
        ),
        # Store for selected true_type from histogram
        dcc.Store(id="selected-true-type", data=None),
    ],
    fluid=True,
)


# Callback to render tab content
@app.callback(
    [Output("tab-content", "children"), Output("selected-true-type", "data")],
    [
        Input("analysis-tabs", "value"),
        Input("leaderboard-table", "selected_rows"),
        Input("selected-true-type", "data"),
    ],
    [State("type-histogram", "clickData")],
)
def render_tab_content(
    tab, selected_rows, prev_selected_true_type, histogram_click_data
):
    if not selected_rows:
        return html.Div("Select a run from the leaderboard"), None

    selected_run_id = run_data.iloc[selected_rows[0]]["run_id"]
    run_data_filtered = detailed_data[detailed_data["run_id"] == selected_run_id]

    # Handle histogram click to update selected_true_type
    selected_true_type = prev_selected_true_type
    ctx = callback_context
    if (
        ctx.triggered
        and ctx.triggered[0]["prop_id"] == "selected-true-type.data"
        and histogram_click_data
        and "points" in histogram_click_data
    ):
        new_true_type = histogram_click_data["points"][0]["x"]
        if new_true_type in all_types:
            selected_true_type = new_true_type
    # Reset selected_true_type when changing runs or tabs
    if ctx.triggered and ctx.triggered[0]["prop_id"] in [
        "leaderboard-table.selected_rows",
        "analysis-tabs.value",
    ]:
        selected_true_type = None

    # Apply true_type filter if selected
    if selected_true_type and selected_true_type in all_types:
        run_data_filtered = run_data_filtered[
            run_data_filtered["true_type"] == selected_true_type
        ]

    if tab == "error-analysis":
        # Type Histogram
        type_counts = (
            detailed_data[detailed_data["run_id"] == selected_run_id]["true_type"]
            .value_counts()
            .reset_index()
        )
        type_counts.columns = ["true_type", "count"]
        type_fig = px.bar(
            type_counts, x="true_type", y="count", title="True Type Distribution"
        )
        type_fig.update_layout(
            clickmode="event+select",
            dragmode=False,
            xaxis={"fixedrange": True, "tickangle": 45},
            yaxis={"fixedrange": True},
            margin={"b": 120},
        )

        # Subtype Confusion Matrix
        cm = pd.crosstab(
            run_data_filtered["true_sub_type"],
            run_data_filtered["pred_sub_type"],
            rownames=["True Subtype"],
            colnames=["Predicted Subtype"],
        )
        cm = cm.reindex(index=all_subtypes, columns=all_subtypes, fill_value=0)
        labels = all_subtypes
        z = cm.values
        total = np.sum(z)
        z_text = [
            [
                (
                    f"{count}<br>({count/total*100:.1f}%)"
                    if total > 0
                    else f"{count}<br>(0.0%)"
                )
                for count in row
            ]
            for row in z
        ]
        try:
            cm_fig = ff.create_annotated_heatmap(
                z,
                x=labels,
                y=labels,
                annotation_text=z_text,
                colorscale="Blues",
                showscale=True,
            )
            cm_fig.update_layout(
                title="Subtype Confusion Matrix",
                xaxis_title="Predicted Subtype",
                yaxis_title="True Subtype",
                clickmode="event+select",
                dragmode=False,
                xaxis={"fixedrange": True},
                yaxis={"fixedrange": True},
            )
        except Exception as e:
            return html.Div(f"Error creating confusion matrix: {str(e)}"), None

        # Datapoint Table (initially all data for the run, filtered by true_type if selected)
        datapoint_columns = [
            {"name": "Text", "id": "text"},
            {"name": "True Type", "id": "true_type"},
            {"name": "True Subtype", "id": "true_sub_type"},
            {"name": "Pred Subtype", "id": "pred_sub_type"},
            {"name": "Correct", "id": "correct"},
        ]

        content = [
            # Type Histogram (full width)
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
            ),
            # Confusion Matrix and Datapoint Table side by side
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
                        className="pe-1",
                    ),
                    dbc.Col(
                        [
                            html.H4("Datapoint Table", className="mb-1"),
                            dash_table.DataTable(
                                id="datapoint-table",
                                columns=datapoint_columns,
                                data=run_data_filtered.to_dict("records"),
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
                                page_size=10,
                            ),
                        ],
                        md=6,
                        className="ps-1",
                    ),
                ]
            ),
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
                page_size=15,
            ),
            None,
        )


# Callback to update datapoint table based on confusion matrix clicks
@app.callback(
    Output("datapoint-table", "data"),
    [
        Input("confusion-matrix", "clickData"),
        Input("leaderboard-table", "selected_rows"),
        Input("selected-true-type", "data"),
    ],
)
def update_datapoint_table(confusion_click_data, selected_rows, selected_true_type):
    if not selected_rows:
        return []

    selected_run_id = run_data.iloc[selected_rows[0]]["run_id"]
    run_data_filtered = detailed_data[detailed_data["run_id"] == selected_run_id]

    # Apply true_type filter if selected
    if selected_true_type and selected_true_type in all_types:
        run_data_filtered = run_data_filtered[
            run_data_filtered["true_type"] == selected_true_type
        ]

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
    app.run(debug=True)
