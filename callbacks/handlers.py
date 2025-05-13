# callbacks/handlers.py
from dash import html, dcc, Input, Output, State, dash_table, ctx, no_update
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

# Simulated run data (in real use, load dynamically)
run_data = {
    "run_01": pd.DataFrame(
        {
            "text": ["doc1", "doc2", "doc3"],
            "pred": ["A", "B", "C"],
            "true": ["A", "C", "C"],
            "type": ["T1", "T2", "T3"],
            "subtype": ["A", "B", "C"],
        }
    ),
    "run_02": pd.DataFrame(
        {
            "text": ["doc4", "doc5", "doc6"],
            "pred": ["B", "C", "A"],
            "true": ["A", "C", "A"],
            "type": ["T1", "T2", "T3"],
            "subtype": ["B", "C", "A"],
        }
    ),
}

leaderboard_df = pd.DataFrame(
    [
        {
            "Run ID": "run_01",
            "Sweep ID": "sweep_01",
            "Model Name": "bert-base",
            "Duration (s)": 123.4,
            "# Items": 3,
            "Benchmark Version": "v1.0",
            "Accuracy": 0.91,
            "F1": 0.89,
            "Sweep Time": "2025-05-13 10:00",
        },
        {
            "Run ID": "run_02",
            "Sweep ID": "sweep_01",
            "Model Name": "roberta-large",
            "Duration (s)": 145.7,
            "# Items": 3,
            "Benchmark Version": "v1.1",
            "Accuracy": 0.85,
            "F1": 0.83,
            "Sweep Time": "2025-05-13 10:05",
        },
    ]
)


def register_callbacks(app):

    @app.callback(
        Output("run-analysis", "children"), Input("leaderboard-table", "selected_rows")
    )
    def show_run_analysis(selected_rows):
        if not selected_rows:
            return html.Div("Select a run from the leaderboard.")

        run_id = leaderboard_df.iloc[selected_rows[0]]["Run ID"]
        df = run_data[run_id]

        base_table = dash_table.DataTable(
            id="base-table",
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict("records"),
            page_size=10,
            filter_action="native",
            sort_action="native",
            style_table={"overflowX": "auto"},
        )

        return html.Div(
            [
                html.H2(f"Run: {run_id}"),
                html.Button("Show Analysis", id="show-analysis-btn", n_clicks=0),
                html.Button(
                    "Back to Table",
                    id="show-table-btn",
                    n_clicks=0,
                    style={"marginLeft": "10px"},
                ),
                html.Div(id="analysis-content"),
                html.H3("Full Dataset"),
                base_table,
            ]
        )

    @app.callback(
        Output("analysis-content", "children"),
        Input("show-analysis-btn", "n_clicks"),
        Input("show-table-btn", "n_clicks"),
        State("leaderboard-table", "selected_rows"),
    )
    def render_analysis(n_analysis, n_table, selected_rows):
        if not ctx.triggered or not selected_rows:
            return no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "show-table-btn":
            return html.Div()

        run_id = leaderboard_df.iloc[selected_rows[0]]["Run ID"]
        df = run_data[run_id]

        cm = pd.crosstab(df["true"], df["pred"], rownames=["True"], colnames=["Pred"])
        percentages = cm.div(cm.sum(axis=1), axis=0).fillna(0)
        annotations = [
            [f"{val}<br>{percent:.1%}" for val, percent in zip(row_vals, row_pcts)]
            for row_vals, row_pcts in zip(cm.values, percentages.values)
        ]

        fig = ff.create_annotated_heatmap(
            z=cm.values,
            x=list(cm.columns),
            y=list(cm.index),
            annotation_text=annotations,
            colorscale="Viridis",
            showscale=True,
            xgap=3,
            ygap=3,
            hoverinfo="z",
            font_colors=["white"],
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            clickmode="event+select",
        )

        type_hist = px.histogram(df, x="type", title="Type Distribution")
        subtype_hist = px.histogram(df, x="subtype", title="Subtype Distribution")

        return html.Div(
            [
                dcc.Graph(id="confusion-matrix", figure=fig),
                dcc.Graph(figure=type_hist),
                dcc.Graph(figure=subtype_hist),
                html.H3("Filtered Datapoint Table"),
                dash_table.DataTable(
                    id="filtered-table",
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict("records"),
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),
            ]
        )

    @app.callback(
        Output("selected-cm-cell", "data"),
        Input("confusion-matrix", "clickData"),
        prevent_initial_call=True,
    )
    def store_cm_cell(clickData):
        if not clickData:
            return None
        true_val = clickData["points"][0]["y"]
        pred_val = clickData["points"][0]["x"]
        return {"true": true_val, "pred": pred_val}

    @app.callback(
        Output("filtered-table", "data"),
        Input("selected-cm-cell", "data"),
        Input("leaderboard-table", "selected_rows"),
    )
    def filter_table(selected_cell, selected_rows):
        if not selected_rows:
            return []

        run_id = leaderboard_df.iloc[selected_rows[0]]["Run ID"]
        df = run_data[run_id]

        if selected_cell:
            df = df[
                (df["true"] == selected_cell["true"])
                & (df["pred"] == selected_cell["pred"])
            ]

        return df.to_dict("records")
