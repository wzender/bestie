import numpy as np
import pandas as pd
import plotly.graph_objects as go

def filtered_transition_heatmap(
    matrix: np.ndarray | pd.DataFrame,
    run1_labels: list[str],
    run2_labels: list[str],
    top_n: int = 20,
    colorscale: str = "Viridis",
    zmin: float | None = None,
    zmax: float | None = None,
    title: str = "Top Off-Diagonal Transitions"
) -> go.Figure:
    """
    Build a Plotly heatmap showing only the largest off-diagonal transitions
    and only rows (run1 types) that actually changed to something else in run2.

    Parameters
    ----------
    matrix : array-like (n x n)
        Transition matrix where rows = run1 types, cols = run2 types.
    run1_labels : list[str]
        Labels for rows (run1).
    run2_labels : list[str]
        Labels for cols (run2).
    top_n : int
        Keep only the top-N off-diagonal cells by value (default 20).
    colorscale : str
        Plotly colorscale name.
    zmin, zmax : float | None
        Optional fixed color range.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
        Plotly figure with a compact heatmap.
    """
    # Wrap as DataFrame for convenience
    if not isinstance(matrix, pd.DataFrame):
        df = pd.DataFrame(matrix, index=run1_labels, columns=run2_labels)
    else:
        df = matrix.copy()
        df.index = run1_labels
        df.columns = run2_labels

    # Drop diagonal influence
    df_no_diag = df.copy()
    np.fill_diagonal(df_no_diag.values, 0)

    # Keep only run1 labels that actually changed (any off-diagonal > 0)
    changing_rows = df_no_diag.sum(axis=1)
    changing_rows = changing_rows[changing_rows > 0].index.tolist()
    if len(changing_rows) == 0:
        # Nothing changed; return empty-looking figure
        fig = go.Figure()
        fig.update_layout(
            title="No off-diagonal transitions found",
            xaxis_title="run2",
            yaxis_title="run1",
            template="plotly_white"
        )
        return fig

    df_changing = df_no_diag.loc[changing_rows, :]

    # Melt to long form and take top-N off-diagonal transitions
    long = (
        df_changing
        .reset_index(names="run1")
        .melt(id_vars="run1", var_name="run2", value_name="val")
    )
    long = long[long["val"] > 0]
    long_sorted = long.sort_values("val", ascending=False)
    long_top = long_sorted.head(max(0, top_n))

    # Subset labels that actually appear in the top-N
    rows_used = long_top["run1"].unique().tolist()
    cols_used = long_top["run2"].unique().tolist()

    # If top_n too small and hides all rows, fall back to any changing rows/cols
    if len(rows_used) == 0 or len(cols_used) == 0:
        rows_used = changing_rows
        cols_used = df_changing.columns[df_changing.sum(axis=0) > 0].tolist()

    # Optional: order rows/cols by total off-diagonal mass in the selected subset
    row_weights = df_changing[cols_used].sum(axis=1).sort_values(ascending=False)
    col_weights = df_changing.loc[rows_used, cols_used].sum(axis=0).sort_values(ascending=False)
    rows_order = [r for r in row_weights.index if r in rows_used]
    cols_order = [c for c in col_weights.index if c in cols_used]

    sub = df_changing.loc[rows_order, cols_order]

    # Build heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=sub.values,
            x=sub.columns.tolist(),
            y=sub.index.tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Value")
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="run2 (destination type)",
        yaxis_title="run1 (source type)",
        template="plotly_white",
        margin=dict(l=80, r=20, t=60, b=60),
    )

    # Optional: annotate the top cells for readability
    annos = []
    sub_long = (
        sub.reset_index(names="run1")
           .melt(id_vars="run1", var_name="run2", value_name="val")
    )
    # Only annotate the original top-N pairs
    top_pairs = set(zip(long_top["run1"], long_top["run2"]))
    for _, row in sub_long.iterrows():
        if row["val"] > 0 and (row["run1"], row["run2"]) in top_pairs:
            annos.append(
                dict(
                    x=row["run2"], y=row["run1"],
                    text=f"{row['val']:.0f}" if row["val"] >= 1 else f"{row['val']:.2f}",
                    showarrow=False, font=dict(size=10)
                )
            )
    fig.update_layout(annotations=annos)

    return fig
