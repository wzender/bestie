from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go
import numpy as np

from filter_transit import filtered_transition_heatmap

# ---- Example data (replace with your real matrix + labels) ----
n = 100
np.random.seed(0)
M = (np.random.rand(n, n) * 10).round(2)
# Add diagonal dominance then we will strip it anyway
for i in range(n):
    M[i, i] += 30

run1_labels = [f"type_{i:03d}" for i in range(n)]
run2_labels = [f"type_{i:03d}" for i in range(n)]
# ---- Example data end ----

app = Dash(__name__)
app.layout = html.Div([
    html.H3("Top Off-Diagonal Transitions"),
    dcc.Slider(
        id="topn",
        min=2, max=40, step=1, value=20,
        marks={i: str(i) for i in [2, 5, 10, 20, 30, 40]}
    ),
    dcc.Graph(id="heatmap", style={"height": "80vh"})
])

@app.callback(
    Output("heatmap", "figure"),
    Input("topn", "value")
)
def update_heatmap(topn):
    fig: go.Figure = filtered_transition_heatmap(
        matrix=M,
        run1_labels=run1_labels,
        run2_labels=run2_labels,
        top_n=int(topn),
        colorscale="Viridis",
        title=f"Top {int(topn)} Off-Diagonal Transitions"
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)
