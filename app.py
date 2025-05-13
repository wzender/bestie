# bestie_app.py (Enhanced)
import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Bestie Leaderboard"

# Sample placeholder data (replace with your real data source)
leaderboard_df = pd.DataFrame(
    [
        {
            "Sweep Time": "2025-05-13 10:00",
            "Run ID": "run_01",
            "Sweep ID": "sweep_01",
            "Model Name": "bert-base",
            "Duration (s)": 123.4,
            "# Items": 3,
            "Benchmark Version": "v1.0",
            "Accuracy": 0.91,
            "F1": 0.89,
        },
        {
            "Sweep Time": "2025-05-13 10:05",
            "Run ID": "run_02",
            "Sweep ID": "sweep_01",
            "Model Name": "roberta-large",
            "Duration (s)": 145.7,
            "# Items": 3,
            "Benchmark Version": "v1.1",
            "Accuracy": 0.85,
            "F1": 0.83,
        },
    ]
)

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

app.layout = html.Div(
    [
        html.H1("🧠 Bestie — Text Classification Leaderboard"),
        html.H2("Leaderboard Table"),
        dash_table.DataTable(
            id="leaderboard-table",
            columns=[{"name": i, "id": i} for i in leaderboard_df.columns],
            data=leaderboard_df.to_dict("records"),
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            style_table={"overflowX": "auto"},
        ),
        dcc.Store(id="selected-cm-cell"),
        html.Div(id="run-analysis"),
    ]
)


@app.callback(
    Output("run-analysis", "children"), Input("leaderboard-table", "selected_rows")
)
def show_run_analysis(selected_rows):
    if not selected_rows:
        return html.Div("Select a run from the leaderboard.")

    run_id = leaderboard_df.iloc[selected_rows[0]]["Run ID"]
    df = run_data[run_id]

    cm = pd.crosstab(df["true"], df["pred"], rownames=["True"], colnames=["Pred"])
    annotations = cm.applymap(lambda v: f"{v}").values

    import plotly.figure_factory as ff

    fig = ff.create_annotated_heatmap(
        z=cm.values,
        x=list(cm.columns),
        y=list(cm.index),
        annotation_text=annotations,
        colorscale="Viridis",
        showscale=True,
    )

    fig.update_layout(title="Confusion Matrix", clickmode="event+select")

    type_hist = px.histogram(df, x="type", title="Type Distribution")
    subtype_hist = px.histogram(df, x="subtype", title="Subtype Distribution")

    return html.Div(
        [
            html.H2(f"Run Analysis: {run_id}"),
            dcc.Graph(id="confusion-matrix", figure=fig),
            dcc.Graph(figure=type_hist),
            dcc.Graph(figure=subtype_hist),
            html.H3("Datapoint Table"),
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=10000)
