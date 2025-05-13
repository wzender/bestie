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
        html.H1("Magellan - Text Classification Leaderboard", className="my-4"),
        # Leaderboard Table
        html.H3("Leaderboard", className="mt-4"),
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
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            row_selectable="single",
            selected_rows=[0],
            page_size=10,
        ),
        # Run Analysis Tabs
        html.H3("Run Analysis", className="mt-4"),
        dcc.Tabs(
            id="analysis-tabs",
            value="error-analysis",
            children=[
                dcc.Tab(label="Error Analysis", value="error-analysis"),
                dcc.Tab(label="Full Run Table", value="full-run"),
            ],
        ),
        html.Div(id="tab-content", className="mt-3"),
    ],
    fluid=True,
)


# Callback to render tab content
@app.callback(
    Output("tab-content", "children"),
    [Input("analysis-tabs", "value"), Input("leaderboard-table", "selected_rows")],
)
def render_tab_content(tab, selected_rows):
    if not selected_rows:
        return html.Div("Select a run from the leaderboard")

    selected_run_id = run_data.iloc[selected_rows[0]]["run_id"]
    run_data_filtered = detailed_data[detailed_data["run_id"] == selected_run_id]

    if tab == "error-analysis":
        # Type Histogram
        type_counts = run_data_filtered["true_type"].value_counts().reset_index()
        type_counts.columns = ["true_type", "count"]
        type_fig = px.bar(
            type_counts, x="true_type", y="count", title="True Type Distribution"
        )
        type_fig.update_layout(
            clickmode="event+select",
            dragmode=False,
            xaxis={"fixedrange": True, "tickangle": 45},
            yaxis={"fixedrange": True},
            margin={"b": 120},  # Extra margin for rotated labels
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
            return html.Div(f"Error creating confusion matrix: {str(e)}")

        # Datapoint Table (initially all data)
        datapoint_columns = [
            {"name": "Text", "id": "text"},
            {"name": "True Type", "id": "true_type"},
            {"name": "True Subtype", "id": "true_sub_type"},
            {"name": "Predicted Subtype", "id": "pred_sub_type"},
            {"name": "Correct", "id": "correct"},
        ]

        return [
            # Type Histogram (full width)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Type Histogram"),
                            dcc.Graph(
                                id="type-histogram",
                                figure=type_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                            ),
                        ],
                        width=12,
                    )
                ],
                className="mb-4",
            ),
            # Confusion Matrix and Datapoint Table side by side
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Subtype Confusion Matrix"),
                            dcc.Graph(
                                id="confusion-matrix",
                                figure=cm_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Datapoint Table"),
                            dash_table.DataTable(
                                id="datapoint-table",
                                columns=datapoint_columns,
                                data=run_data_filtered.to_dict("records"),
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "left", "padding": "5px"},
                                style_header={
                                    "backgroundColor": "rgb(230, 230, 230)",
                                    "fontWeight": "bold",
                                },
                                page_size=10,
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
        ]

    elif tab == "full-run":
        return dash_table.DataTable(
            columns=[
                {"name": "Text", "id": "text"},
                {"name": "True Type", "id": "true_type"},
                {"name": "True Subtype", "id": "true_sub_type"},
                {"name": "Predicted Subtype", "id": "pred_sub_type"},
                {"name": "Correct", "id": "correct"},
            ],
            data=run_data_filtered.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            page_size=15,
        )


# Combined callback to update datapoint table based on confusion matrix or histogram clicks
@app.callback(
    Output("datapoint-table", "data"),
    [
        Input("confusion-matrix", "clickData"),
        Input("type-histogram", "clickData"),
        Input("leaderboard-table", "selected_rows"),
    ],
)
def update_datapoint_table(confusion_click_data, histogram_click_data, selected_rows):
    if not selected_rows:
        return []

    selected_run_id = run_data.iloc[selected_rows[0]]["run_id"]
    run_data_filtered = detailed_data[detailed_data["run_id"] == selected_run_id]

    ctx = callback_context
    if not ctx.triggered:
        return run_data_filtered.to_dict("records")

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (
        triggered_id == "confusion-matrix"
        and confusion_click_data
        and "points" in confusion_click_data
    ):
        point = confusion_click_data["points"][0]
        true_subtype = point["y"]
        pred_subtype = point["x"]
        filtered = run_data_filtered[
            (run_data_filtered["true_sub_type"] == true_subtype)
            & (run_data_filtered["pred_sub_type"] == pred_subtype)
        ]
        return filtered.to_dict("records")

    elif (
        triggered_id == "type-histogram"
        and histogram_click_data
        and "points" in histogram_click_data
    ):
        point = histogram_click_data["points"][0]
        true_type = point["x"]
        filtered = run_data_filtered[run_data_filtered["true_type"] == true_type]
        return filtered.to_dict("records")

    return run_data_filtered.to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)
