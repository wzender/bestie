# components/layout.py
from dash import html, dcc, dash_table
import pandas as pd

# Placeholder data for the leaderboard
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

layout = html.Div(
    [
        html.H1("🧠 Bestie — Text Classification Leaderboard"),
        html.H2("Leaderboard Table"),
        dash_table.DataTable(
            id="leaderboard-table",
            columns=[{"name": i, "id": i} for i in leaderboard_df.columns],
            data=leaderboard_df.to_dict("records"),
            row_selectable="single",
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
        ),
        dcc.Store(id="selected-cm-cell"),
        html.Div(id="run-analysis"),
    ]
)
