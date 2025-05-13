import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, dash_table
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Sample run data - replace with dynamic loading in production
run_data = {
    "run_01": pd.DataFrame(
        {
            "text": [f"doc{i+1}" for i in range(9)],
            "true_type": ["A", "A", "B", "B", "B", "C", "A", "B", "C"],
            "true_sub_type": ["A1", "A1", "B1", "B2", "B2", "C1", "A2", "B3", "C1"],
            "pred_sub_type": ["A1", "A2", "B1", "B2", "B3", "C1", "A1", "B2", "C1"],
        }
    )
}

leaderboard = pd.DataFrame(
    {
        "Run ID": ["run_01"],
        "Sweep ID": ["sweep_001"],
        "Learning Rate": [0.001],
        "Batch Size": [32],
        "Accuracy": [0.77],
        "F1-score": [0.75],
    }
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Magellan Leaderboard"

# Helper to calculate confusion matrix


def generate_confusion_matrix(df):
    subtypes = sorted(list(set(df["true_sub_type"]).union(set(df["pred_sub_type"]))))
    conf_mat = pd.crosstab(
        df["true_sub_type"],
        df["pred_sub_type"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )
    conf_mat = conf_mat.reindex(index=subtypes, columns=subtypes, fill_value=0)
    conf_values = conf_mat.values
    total = conf_values.sum(axis=1, keepdims=True)
    percent = (conf_values / total.clip(min=1) * 100).round(1).astype(str) + "%"
    annotations = [
        [f"{v}<br>{p}" for v, p in zip(row_v, row_p)]
        for row_v, row_p in zip(conf_values, percent)
    ]
    fig = ff.create_annotated_heatmap(
        conf_values,
        x=subtypes,
        y=subtypes,
        annotation_text=annotations,
        colorscale="Blues",
    )
    fig.update_layout(title="Subtype Confusion Matrix")
    return fig


# App Layout
app.layout = dbc.Container(
    [
        html.H2("🧠 Magellan — Lightweight Leaderboard for Text Classification"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Leaderboard"),
                        dash_table.DataTable(
                            id="leaderboard-table",
                            columns=[{"name": i, "id": i} for i in leaderboard.columns],
                            data=leaderboard.to_dict("records"),
                            row_selectable="single",
                            selected_rows=[0],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left"},
                        ),
                    ]
                )
            ]
        ),
        html.Hr(),
        dcc.Tabs(
            id="tabs",
            value="error",
            children=[
                dcc.Tab(label="Error Analysis", value="error"),
                dcc.Tab(label="Full Run Table", value="full"),
            ],
        ),
        html.Div(id="run-content"),
    ],
    fluid=True,
)


@app.callback(
    Output("run-content", "children"),
    Input("tabs", "value"),
    Input("leaderboard-table", "selected_rows"),
)
def update_tab_content(tab, selected_rows):
    run_id = leaderboard.iloc[selected_rows[0]]["Run ID"]
    df = run_data[run_id].copy()
    df["correct"] = df["true_sub_type"] == df["pred_sub_type"]

    if tab == "error":
        type_fig = px.histogram(
            df, x="true_type", title="Type Histogram", color="true_type"
        )
        cm_fig = generate_confusion_matrix(df)
        return dbc.Container(
            [
                dcc.Graph(figure=type_fig),
                dcc.Graph(id="cm-graph", figure=cm_fig),
                html.H5("Filtered Data Points"),
                dash_table.DataTable(
                    id="confusion-click-data",
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=[],
                    page_size=5,
                ),
            ]
        )

    elif tab == "full":
        return dbc.Container(
            [
                dash_table.DataTable(
                    columns=[
                        {"name": "text", "id": "text"},
                        {"name": "true_type", "id": "true_type"},
                        {"name": "true_sub_type", "id": "true_sub_type"},
                        {"name": "pred_sub_type", "id": "pred_sub_type"},
                        {"name": "correct", "id": "correct"},
                    ],
                    data=df.to_dict("records"),
                    style_table={"overflowX": "auto"},
                    page_size=10,
                )
            ]
        )


@app.callback(
    Output("confusion-click-data", "data"),
    Input("cm-graph", "clickData"),
    Input("leaderboard-table", "selected_rows"),
)
def update_table_on_click(clickData, selected_rows):
    if clickData is None:
        return []
    run_id = leaderboard.iloc[selected_rows[0]]["Run ID"]
    df = run_data[run_id]
    point = clickData["points"][0]
    true_val = point["y"]
    pred_val = point["x"]
    return df[
        (df["true_sub_type"] == true_val) & (df["pred_sub_type"] == pred_val)
    ].to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)
