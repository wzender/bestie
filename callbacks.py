from dash import Output, Input, State, callback_context, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dash_table import DataTable
from utils import get_font_color
from sklearn.metrics import f1_score

F1_CIRCLE_SIZE = 35

# def f1_to_rgb(f1, f1_min=0.0, f1_max=1.0):
#     # Normalize f1 to [0, 1]
#     norm = (f1 - f1_min) / (f1_max - f1_min) if f1_max > f1_min else 0
#     norm = max(0.0, min(1.0, norm))  # Clamp

#     # Red to green gradient
#     r = int(255 * (1 - norm))      # 255 → 0 as F1 increases
#     g = int(200 * norm + 55)       # 55 → 255
#     b = int(150 - 100 * norm)      # 150 → 50
#     return f"rgb({r},{g},{b})"

# Updated color function for light green to light yellow to light red (traffic light style)
def f1_to_rgb(f1, f1_min=0.0, f1_max=1.0):
    # Normalize f1 to [0, 1]
    norm = (f1 - f1_min) / (f1_max - f1_min) if f1_max > f1_min else 0
    norm = max(0.0, min(1.0, norm))  # Clamp

    # Light green (high F1) to light yellow (mid F1) to light red (low F1)
    if norm >= 0.5:
        # Transition from light yellow (norm=0.5) to light green (norm=1.0)
        r = int(255 - 135 * (norm - 0.5) / 0.5)  # 255 → 120
        g = int(255)                              # Fixed high green for brightness
        b = int(120 - 0 * (norm - 0.5) / 0.5)    # Fixed at 120
    else:
        # Transition from light red (norm=0.0) to light yellow (norm=0.5)
        r = int(255)                              # Fixed high red
        g = int(120 + 135 * norm / 0.5)          # 120 → 255
        b = int(120)                              # Fixed at 120
    return f"rgb({r},{g},{b})"


def register_callbacks(app, run_data, detailed_data, test_run_id):
    all_types = [chr(65 + i) for i in range(20)] + ["Perfect", "WorstMin", "Medium"]

    # @app.callback(
    #     Output("selected-true-type", "data"),
    #     Input("highlighted-run-id", "data"),
    # )
    # def reset_selected_type(highlighted_run_id):
    #     return None

    # Clientside callback for CSV download
    app.clientside_callback(
        """
        function(n_clicks, data) {
            if (!n_clicks || !data || data.length === 0) return null;
            const columns = ['text', 'true_type', 'true_subtype', 'pred_subtype', 'pred_unk_subtype', 'correct'];
            let csv = columns.join(',') + '\\n';
            data.forEach(row => {
                const values = columns.map(col => {
                    const value = row[col];
                    if (value === null || value === undefined) return '';
                    const str = String(value).replace(/"/g, '""');
                    return `"${str}"`;
                });
                csv += values.join(',') + '\\n';
            });
            const filename = 'datapoint_table.csv';
            return {
                content: csv,
                filename: filename,
                type: 'text/csv',
                base64: false
            };
        }
        """,
        Output("download-datapoint-csv", "data"),
        Input("export-csv-btn", "n_clicks"),
        State("datapoint-table", "data"),
        prevent_initial_call=True,
    )

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
            return no_update, no_update, no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "benchmark-dropdown":
            if not selected_benchmark:
                return [], None, []
            df = run_data[run_data["benchmark"] == selected_benchmark]
            data = df.to_dict("records")
            if not data:
                return [], None, []
            highlighted_run_id = data[0]["run_id"]
            style_data_conditional = [
                {
                    "if": {"filter_query": f'{{run_id}} = "{highlighted_run_id}"'},
                    "border": "2px solid #4ADE80",
                },
                {
                    "if": {"state": "selected"},
                    "border": "2px solid #4ADE80",
                },
                {
                    "if": {"state": "active"},
                    "backgroundColor": "#A7F3D0",
                    "border": "1px solid #06B6D4",
                },
            ]
            return data, highlighted_run_id, style_data_conditional
        elif triggered_id == "leaderboard-table" and active_cell and viewport_data:
            row_index = active_cell["row"]
            highlighted_run_id = viewport_data[row_index]["run_id"]
            style_data_conditional = [
                {
                    "if": {"filter_query": f'{{run_id}} = "{highlighted_run_id}"'},
                    "border": "2px solid #4ADE80",
                },
                {
                    "if": {"state": "selected"},
                    "border": "2px solid #4ADE80",
                },
                {
                    "if": {"state": "active"},
                    "backgroundColor": "#A7F3D0",
                    "border": "1px solid #06B6D4",
                },
            ]
            return no_update, highlighted_run_id, style_data_conditional
        return no_update, no_update, no_update

    @app.callback(
        [
            Output("histogram-click-data", "data"),
            Output("selected-true-type", "data"),
        ],
        [
            Input("type-histogram", "clickData"),
            Input("highlighted-run-id", "data"),
        ],
        [
            State("selected-true-type", "data"),
        ],
        prevent_initial_call=True,
    )
    def capture_histogram_click_data(
        type_click_data, highlighted_run_id, current_selected_true_type
    ):
        ctx = callback_context
        if not ctx.triggered or not highlighted_run_id:
            return None, None
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if (
            triggered_id == "type-histogram"
            and type_click_data
            and "points" in type_click_data
        ):
            clicked_type = type_click_data["points"][0]["x"]
            if clicked_type in all_types:
                if clicked_type == current_selected_true_type:
                    return type_click_data, None
                else:
                    return type_click_data, clicked_type
        return type_click_data, current_selected_true_type

    @app.callback(
        [
            Output("type-histogram-content", "children"),
            Output("subtype-confusion-content", "children"),
            Output("subtype-f1-content", "children"),
            Output("datapoint-content", "children"),
        ],
        [
            Input("highlighted-run-id", "data"),
            Input("selected-true-type", "data"),
            Input("selected-confusion-cell", "data"),
        ],
        prevent_initial_call=True,
    )
    def render_content(highlighted_run_id, selected_true_type, selected_confusion_cell):
        if not highlighted_run_id:
            return [
                html.Div(),
                html.Div(
                    "Highlight a run from the leaderboard",
                    className="text-center text-gray-600 text-lg",
                ),
                html.Div(),
                html.Div(),
            ]
        run_data_filtered = detailed_data[detailed_data["run_id"] == highlighted_run_id]
        type_counts = (
            run_data_filtered["true_type"]
            .value_counts()
            .reindex(all_types, fill_value=0)
            .reset_index()
        )
        type_counts.columns = ["true_type", "count"]
        # Calculate F1 scores per type
        f1_scores = []
        for t in all_types:
            type_data = run_data_filtered[run_data_filtered["true_type"] == t]
            if not type_data.empty:
                f1 = f1_score(
                    type_data["true_subtype"],
                    type_data["pred_subtype"],
                    labels=type_data["true_subtype"].unique(),
                    average="weighted",
                    zero_division=0,
                )
            else:
                f1 = 0
            f1_scores.append(f1)
        f1_df = pd.DataFrame({"true_type": all_types, "f1_score": f1_scores})
        # Sort types by F1 score (descending)
        f1_df = f1_df.sort_values(by="f1_score", ascending=False)
        sorted_types = f1_df["true_type"].tolist()
        type_counts = (
            type_counts.set_index("true_type").reindex(sorted_types).reset_index()
        )
        f1_df = f1_df.set_index("true_type").reindex(sorted_types).reset_index()
        if run_data_filtered["true_type"].value_counts().sum() == 0:
            type_histogram_content = [
                dbc.Card(
                    [
                        html.H3(
                            "Type Metrics",
                            className="text-2xl font-semibold text-indigo-900 mb-4",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Alert(
                                            "No True Type data available for this run",
                                            color="info",
                                            className="text-center rounded-md bg-cyan-100 text-indigo-900",
                                        ),
                                    ],
                                    width=12,
                                )
                            ]
                        ),
                    ],
                    className="p-4 bg-slate-50 rounded-lg shadow-md mb-6",
                )
            ]
            return type_histogram_content, html.Div(), [], html.Div()
        # Type histogram with F1 score circles
        type_fig = go.Figure()
        type_fig.add_trace(
            go.Bar(
                x=type_counts["true_type"],
                y=type_counts["count"],
                marker=dict(
                    color="#06B6D4",
                    opacity=0.1,
                ),
                text=type_counts["count"],
                textposition="outside",
                textfont=dict(size=12, family="sans-serif", color="#1F2937"),
                hovertemplate="Type: %{x}<br>Count: %{y}<extra></extra>",
                selected=dict(marker=dict(opacity=0.9)),
                unselected=dict(marker=dict(opacity=0.1)),
            )
        )

        max_count = type_counts["count"].max()
        # Add F1 score circles
        f1_y_position = max_count * 1.15 + 5  # Horizontally align circles above bars
        # Light red to light green color scale
        num_types = len(sorted_types)

        # f1_colors = create_f1_colors(num_types)
        type_f1_colors = [f1_to_rgb(f1) for f1 in f1_df["f1_score"]]

        type_fig.add_trace(
            go.Scatter(
                x=type_counts["true_type"],
                y=[f1_y_position] * len(type_counts),
                mode="markers+text",
                marker=dict(
                    size=F1_CIRCLE_SIZE,
                    color=type_f1_colors,
                    line=dict(width=1, color="#1F2937"),
                ),
                text=[f"{int(f1 * 100)}%" for f1 in f1_df["f1_score"]],
                textposition="middle center",
                textfont=dict(size=10, family="sans-serif", color="#1F2937"),
                hovertemplate="Type: %{x}<br>F1 Score: %{text}<extra></extra>",
                showlegend=False,
            )
        )
        if selected_true_type:
            selected_index = (
                sorted_types.index(selected_true_type)
                if selected_true_type in sorted_types
                else None
            )
            type_fig.update_traces(
                selectedpoints=[selected_index] if selected_index is not None else [],
                selector=dict(type="bar"),
            )
            type_fig.update_traces(
                selectedpoints=[selected_index] if selected_index is not None else [],
                selector=dict(type="scatter"),
            )
        type_fig.update_layout(
            clickmode="event+select",
            dragmode=False,
            xaxis=dict(
                fixedrange=True,
                tickangle=45,
                title="",
                tickfont=dict(color="#1F2937"),
            ),
            yaxis=dict(
                fixedrange=True,
                title="Count",
                tickfont=dict(color="#1F2937"),
                range=[0, f1_y_position * 1.1],
            ),
            height=400,
            margin=dict(l=40, r=20, t=20, b=80),
            font=dict(family="sans-serif", size=12, color="#1F2937"),
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#F8FAFC",
            template="plotly_white",
            showlegend=False,
        )
        type_histogram_content = [
            dbc.Card(
                [
                    html.Div(
                        [
                            html.H3(
                                "Type Metrics (F1 score and count)",
                                className="text-2xl font-semibold text-indigo-900 mb-4",
                            ),
                            # dcc.Graph(
                            #     id="sample-f1-circle",
                            #     figure=sample_f1_fig,
                            #     config={"displayModeBar": False, "scrollZoom": False},
                            #     className="p-0",
                            # ),
                        ]
                    ),
                    dcc.Graph(
                        id="type-histogram",
                        figure=type_fig,
                        config={"displayModeBar": False, "scrollZoom": False},
                        className="p-0",
                    ),
                ],
                className="p-4 bg-slate-50 rounded-lg shadow-md mb-6",
            )
        ]
        subtype_f1_content = []
        confusion_content = []
        if selected_true_type and selected_true_type in all_types:
            cm_table_data = run_data_filtered[
                run_data_filtered["true_type"] == selected_true_type
            ]
            true_subtypes = cm_table_data["true_subtype"].dropna().unique()
            pred_subtypes = cm_table_data["pred_subtype"].dropna().unique()
            all_subtypes_in_type = sorted(
                list(set(true_subtypes).union(set(pred_subtypes))), reverse=True
            )
            # Calculate F1 scores per subtype
            subtype_f1_scores = []
            for subtype in all_subtypes_in_type:
                subtype_data = cm_table_data[cm_table_data["true_subtype"] == subtype]
                if not subtype_data.empty:
                    f1 = f1_score(
                        subtype_data["true_subtype"],
                        subtype_data["pred_subtype"],
                        labels=[subtype],
                        average="weighted",
                        zero_division=0,
                    )
                else:
                    f1 = 0
                subtype_f1_scores.append(f1)
            subtype_df = pd.DataFrame(
                {"subtype": all_subtypes_in_type, "f1_score": subtype_f1_scores}
            )
            # Sort subtypes by F1 score (descending)
            subtype_df = subtype_df.sort_values(by="f1_score", ascending=True)
            sorted_subtypes = subtype_df["subtype"].tolist()
            if not all_subtypes_in_type:
                subtype_f1_content = [] # subtype_f1_content
                confusion_content = [
                    dbc.Alert(
                        f"No subtypes available for selected type {selected_true_type} in this run",
                        # color="info",
                        className="text-center rounded-md bg-cyan-100 text-indigo-900"
                    ),
                ]
            else:
                cm = pd.crosstab(
                    cm_table_data["true_subtype"],
                    cm_table_data["pred_subtype"],
                    rownames=["True Subtype"],
                    colnames=["Predicted Subtype"],
                    dropna=False,
                ).reindex(index=sorted_subtypes, columns=sorted_subtypes, fill_value=0)
                if cm.empty:
                    subtype_f1_content = [] # confusion_content
                    confusion_content = [

                        dbc.Alert(
                        f"No predictions for {selected_true_type} subtypes",
                        # color="info",
                        className="text-center rounded-md bg-cyan-100 text-indigo-900"
                    ),
                    ]
                else:
                    # Assuming subtype_df, sorted_subtypes, f1_colors, and F1_CIRCLE_SIZE are defined
                    num_subtypes = len(sorted_subtypes)
                    gap_size = 0.8  # Adjust this value to control the gap between circles

                    subtype_f1_colors = [f1_to_rgb(f1) for f1 in subtype_df["f1_score"]]

                    # Create figure
                    subtype_f1_fig = go.Figure()

                    # Add a scatter trace for each subtype to control individual positioning
                    for i, (subtype, f1) in enumerate(zip(sorted_subtypes, subtype_df["f1_score"])):
                        y_position = i * gap_size  # Numerical y-position with gap
                        subtype_f1_fig.add_trace(
                            go.Scatter(
                                x=[-0.5],  # Fixed x-position for each circle
                                y=[y_position],  # Unique y-position for each circle
                                mode="markers+text",
                                marker=dict(
                                    size=F1_CIRCLE_SIZE,
                                    color=subtype_f1_colors[i],  # Use corresponding color
                                    line=dict(width=1, color="#1F2937"),
                                ),
                                text=[f"{int(f1 * 100)}%"],  # F1 score as text
                                textposition="middle center",
                                textfont=dict(
                                    size=10, family="sans-serif", color="#1F2937"
                                ),
                                hovertemplate=f"Subtype: {subtype}<br>F1 Score: {int(f1 * 100)}%<extra></extra>",
                                showlegend=False,
                            )
                        )

                    # Update layout to customize y-axis and figure appearance
                    subtype_f1_fig.update_layout(
                        xaxis=dict(visible=False, range=[-1, 0]),
                        yaxis=dict(
                            visible=False,  # Show y-axis for labels
                            tickvals=[i * gap_size for i in range(num_subtypes)],  # Numerical positions
                            ticktext=sorted_subtypes,  # Subtype names as labels
                            range=[-0.5, (num_subtypes - 1) * gap_size + 0.5],  # Adjust range for all circles
                            showgrid=False,
                        ),
                        height=40 * num_subtypes + 120,  # Adjust height based on number of subtypes
                        margin=dict(l=40, r=0, t=40, b=70),
                        plot_bgcolor="#F8FAFC",
                        paper_bgcolor="#F8FAFC",
                        showlegend=False,
                    )
                    subtype_f1_content = [
                        dcc.Graph(
                            id="subtype-f1-circles",
                            figure=subtype_f1_fig,
                            config={"displayModeBar": False, "scrollZoom": False},
                            className="p-0",
                        ),
                    ]
                    # Confusion matrix
                    z = cm.values
                    z_min, z_max = z.min(), z.max()
                    if z_max == z_min:
                        z_normalized = np.zeros_like(z, dtype=float)
                    else:
                        z_normalized = (z - z_min) / (z_max - z_min)
                    vivid_colors = ["#111827", "#6B7280", "#06B6D4"]
                    vivid_scale = [[0.0, "#111827"], [0.5, "#6B7280"], [1.0, "#06B6D4"]]
                    colors = np.empty_like(z, dtype=object)
                    for i in range(z.shape[0]):
                        for j in range(z.shape[1]):
                            norm_val = z_normalized[i, j]
                            for k in range(len(vivid_scale) - 1):
                                if norm_val <= vivid_scale[k + 1][0]:
                                    frac = (
                                        (norm_val - vivid_scale[k][0])
                                        / (vivid_scale[k + 1][0] - vivid_scale[k][0])
                                        if vivid_scale[k + 1][0] != vivid_scale[k][0]
                                        else 0
                                    )
                                    rgb_start = [
                                        int(x, 16)
                                        for x in [
                                            vivid_scale[k][1].lstrip("#")[i : i + 2]
                                            for i in range(0, 6, 2)
                                        ]
                                    ]
                                    rgb_end = [
                                        int(x, 16)
                                        for x in [
                                            vivid_scale[k + 1][1].lstrip("#")[i : i + 2]
                                            for i in range(0, 6, 2)
                                        ]
                                    ]
                                    rgb = [
                                        int(
                                            rgb_start[c]
                                            + frac * (rgb_end[c] - rgb_start[c])
                                        )
                                        for c in range(3)
                                    ]
                                    colors[i, j] = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                                    break
                                else:
                                    colors[i, j] = vivid_scale[-1][1]
                    font_colors = [
                        [get_font_color(colors[i, j]) for j in range(z.shape[1])]
                        for i in range(z.shape[0])
                    ]
                    # Confusion Matrix Plot
                    cm_fig = go.Figure(
                        data=[
                            go.Heatmap(
                                z=cm.values,
                                x=cm.columns.tolist(),
                                y=cm.index.tolist(),  # Uses sorted_subtypes via reindex
                                colorscale=vivid_scale,
                                showscale=False,
                                text=cm.values,
                                texttemplate="%{text}",
                                textfont=dict(size=12, family="sans-serif"),
                                hoverinfo="z+x+y",
                            )
                        ]
                    )
                    text_trace = go.Scatter(
                        x=[
                            cm.columns[j]
                            for i in range(z.shape[0])
                            for j in range(z.shape[1])
                        ],
                        y=[
                            cm.index[i]
                            for i in range(z.shape[0])
                            for j in range(z.shape[1])
                        ],
                        text=[
                            str(z[i, j])
                            for i in range(z.shape[0])
                            for j in range(z.shape[1])
                        ],
                        mode="text",
                        textfont=dict(
                            size=12,
                            family="sans-serif",
                            color=[
                                font_colors[i][j]
                                for i in range(z.shape[0])
                                for j in range(z.shape[1])
                            ],
                        ),
                        showlegend=False,
                        hoverinfo="none",
                    )
                    cm_fig.add_trace(text_trace)
                    if selected_confusion_cell:
                        true_subtype = selected_confusion_cell.get("true_subtype")
                        pred_subtype = selected_confusion_cell.get("pred_subtype")
                        if true_subtype in cm.index and pred_subtype in cm.columns:
                            i = cm.index.tolist().index(true_subtype)
                            j = cm.columns.tolist().index(pred_subtype)
                            cm_fig.add_shape(
                                type="rect",
                                x0=j - 0.5,
                                x1=j + 0.5,
                                y0=i - 0.5,
                                y1=i + 0.5,
                                xref="x",
                                yref="y",
                                line=dict(color="#4ADE80", width=3),
                                fillcolor="rgba(0,0,0,0)",
                            )
                    cell_size = 40
                    num_subtypes = len(all_subtypes_in_type)
                    graph_width = cell_size * num_subtypes + 80
                    graph_height = cell_size * num_subtypes + 120
                    cm_fig.update_layout(
                        xaxis=dict(
                            tickangle=45, tickfont=dict(size=12, color="#1F2937")
                        ),
                        yaxis=dict(
                            tickfont=dict(size=12, color="#1F2937"),
                            categoryorder="array",
                            categoryarray=sorted_subtypes,  # Match F1 circles
                        ),
                        width=graph_width,
                        height=40 * num_subtypes + 120,  # Consistent height
                        margin=dict(l=20, r=20, t=40, b=80),
                        plot_bgcolor="#F8FAFC",
                        paper_bgcolor="#F8FAFC",
                    )
                    confusion_content = [
                        dcc.Graph(
                            id="confusion-matrix",
                            figure=cm_fig,
                            config={
                                "displayModeBar": False,
                                "scrollZoom": False,
                            },
                            className="p-0",
                        ),
                    ]
        else:
            confusion_content = [
                dbc.Alert(
                    "Please select a type from the Type Histogram to view the Subtype Metrics",
                    color="info",
                    className="text-center rounded-md bg-cyan-100 text-indigo-900",
                ),
            ]
            subtype_f1_content = []
        # Combine confusion_content and subtype_f1_content in a single dbc.Card with layout
        subtype_metrics_content = [
            html.Div(
                [
                    html.H3(
                        "Subtype Metrics (F1 score and count)",
                        className="text-2xl font-semibold text-indigo-900 mb-4",
                    ),
                    html.Div(
                        [
                            html.Div(
                                subtype_f1_content,
                                style={"width": "7%", "paddingRight": "8px"},
                            ),
                            html.Div(
                                confusion_content,
                                style={"width": "93%"},
                            ),
                        ],
                        className="flex flex-row items-start w-full",  # align left-to-right
                    ),
                ],
                className="p-4 bg-slate-50 rounded-lg shadow-md mb-6",
            )
        ]
        datapoint_columns = [
            {"name": "Text", "id": "text"},
            {"name": "True Type", "id": "true_type"},
            {"name": "True Subtype", "id": "true_subtype"},
            {"name": "Pred Subtype", "id": "pred_subtype"},
            {"name": "Pred Unk Subtype", "id": "pred_unk_subtype"},
            {"name": "Correct", "id": "correct"},
        ]
        if selected_true_type and selected_true_type in all_types:
            table_data = run_data_filtered[
                run_data_filtered["true_type"] == selected_true_type
            ]
        else:
            table_data = run_data_filtered
        if selected_confusion_cell:
            true_subtype = selected_confusion_cell.get("true_subtype")
            pred_subtype = selected_confusion_cell.get("pred_subtype")
            table_data = run_data_filtered[
                (run_data_filtered["true_subtype"] == true_subtype)
                & (run_data_filtered["pred_subtype"] == pred_subtype)
            ]
        current_count = len(table_data)
        total_count = len(run_data_filtered)
        notification = []
        if selected_confusion_cell:
            true_subtype = selected_confusion_cell.get("true_subtype")
            pred_subtype = selected_confusion_cell.get("pred_subtype")
            notification = [
                dbc.Alert(
                    f"Selected: True Subtype = {true_subtype}, Predicted Subtype = {pred_subtype}",
                    color="primary",
                    className="mb-4 text-sm rounded-md bg-cyan-100 text-indigo-900",
                    dismissable=True,
                )
            ]
        datapoint_content = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                "Datapoint Table",
                                className="text-xl font-semibold text-indigo-900 mb-4",
                            ),
                            *notification,
                            html.Div(
                                [
                                    dbc.Button(
                                        "Export to CSV",
                                        id="export-csv-btn",
                                        color="primary",
                                        className="mb-4 bg-indigo-600 hover:bg-green-400 text-white font-medium py-2 px-4 rounded-md transition duration-200",
                                        disabled=len(table_data) == 0,
                                    ),
                                ],
                            ),
                            DataTable(
                                id="datapoint-table",
                                columns=datapoint_columns,
                                data=table_data.to_dict("records"),
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "12px",
                                    "fontSize": "14px",
                                    "fontFamily": "sans-serif",
                                    "color": "#1F2937",
                                },
                                style_header={
                                    "backgroundColor": "#E2E8F0",
                                    "fontWeight": "600",
                                    "fontSize": "14px",
                                    "padding": "12px",
                                    "color": "#1F2937",
                                    "fontFamily": "sans-serif",
                                },
                                style_data={
                                    "borderBottom": "1px solid #E2E8F0",
                                    "transition": "background-color 0.2s",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "#4ADE80",
                                        "border": "1px solid #06B6D4",
                                    },
                                    {
                                        "if": {"state": "active"},
                                        "backgroundColor": "#A7F3D0",
                                        "border": "1px solid #06B6D4",
                                    },
                                ],
                                css=[
                                    {
                                        "selector": ".dash-table-container tr:hover",
                                        "rule": "background-color: #E2E8F0; cursor: pointer;",
                                    }
                                ],
                                page_size=10,
                                filter_action="native",
                                sort_action="native",
                            ),
                            html.Div(
                                f"Showing {current_count} of {total_count} items",
                                className="mt-4 text-sm text-gray-600 bg-slate-200 py-2 px-4 rounded-full inline-block",
                            ),
                        ],
                        width=12,
                    )
                ]
            )
        ]
        return type_histogram_content, subtype_metrics_content, [], datapoint_content

    @app.callback(
        [
            Output("datapoint-table", "data"),
            Output("selected-confusion-cell", "data"),
        ],
        [
            Input("confusion-matrix", "clickData"),
            Input("highlighted-run-id", "data"),
            Input("selected-true-type", "data"),
        ],
        [
            State("selected-confusion-cell", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_datapoint_table(
        confusion_click_data,
        highlighted_run_id,
        selected_true_type,
        current_selected_cell,
    ):
        if not highlighted_run_id:
            return [], None
        run_data_filtered = detailed_data[detailed_data["run_id"] == highlighted_run_id]
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if (
            triggered_id == "confusion-matrix"
            and confusion_click_data
            and "points" in confusion_click_data
        ):
            point = confusion_click_data["points"][0]
            true_subtype = point["y"]
            pred_subtype = point["x"]
            if (
                current_selected_cell
                and current_selected_cell.get("true_subtype") == true_subtype
                and current_selected_cell.get("pred_subtype") == pred_subtype
            ):
                if selected_true_type and selected_true_type in all_types:
                    filtered = run_data_filtered[
                        run_data_filtered["true_type"] == selected_true_type
                    ]
                    return filtered.to_dict("records"), None
                return run_data_filtered.to_dict("records"), None
            filtered = run_data_filtered[
                (run_data_filtered["true_subtype"] == true_subtype)
                & (run_data_filtered["pred_subtype"] == pred_subtype)
            ]
            return filtered.to_dict("records"), {
                "true_subtype": true_subtype,
                "pred_subtype": pred_subtype,
            }
        if selected_true_type and selected_true_type in all_types:
            filtered = run_data_filtered[
                run_data_filtered["true_type"] == selected_true_type
            ]
            return filtered.to_dict("records"), None
        return run_data_filtered.to_dict("records"), None


    # New comparison callback: update comparison-runs based on selected_rows in leaderboard-table
    @app.callback(
        Output("comparison-runs", "data"),
        [
            Input("leaderboard-table", "selected_rows"),
            Input("clear-compare-btn", "n_clicks"),
        ],
        [
            State("leaderboard-table", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_comparison_runs(selected_rows, clear_clicks, leaderboard_data):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "clear-compare-btn":
            return []
        if not selected_rows or not leaderboard_data:
            return []
        # Get run_ids for selected rows, limited to first 3
        selected_run_ids = []
        for i in selected_rows[:3]:  # Only process first 3 selected rows
            if i < len(leaderboard_data) and "run_id" in leaderboard_data[i]:
                selected_run_ids.append(leaderboard_data[i]["run_id"])
        return selected_run_ids


    @app.callback(
        Output("leaderboard-table", "selected_rows"),
        Input("clear-compare-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_leaderboard_selections(n_clicks):
        # When the Clear Comparison button is clicked, also clear table row selections
        return []

    @app.callback(
        [
            Output("comparison-datapoint-display", "children"),
            Output("comparison-data-store", "data"),
        ],
        Input("comparison-runs", "data"),
    )
    def display_comparison_datapoints(comparison_runs):
        if len(comparison_runs) < 2:
            return html.Div(), None

        # Get detailed data for all comparison runs
        comparison_detailed = detailed_data[detailed_data["run_id"].isin(comparison_runs)]

        if comparison_detailed.empty:
            return dbc.Alert("No datapoints available for comparison", color="warning")

        # Get sample of datapoints to compare (limit to 20 for performance)
        # Group by text to ensure we're comparing same samples
        grouped = comparison_detailed.groupby("text", as_index=False).filter(
            lambda x: len(x) == len(comparison_runs)
        )

        if grouped.empty:
            return dbc.Alert("No common datapoints between selected runs", color="info")

        sample_texts = grouped["text"].unique()[:20]
        comparison_sample = grouped[grouped["text"].isin(sample_texts)]

        # Build comparison table
        rows = []
        for row_num, text in enumerate(sample_texts, start=1):
            text_data = comparison_sample[comparison_sample["text"] == text]
            if text_data.empty:
                continue

            row_dict = {"index": row_num, "text": text[:50] + "..." if len(text) > 50 else text}

            # Add each run's predictions
            for idx, run_id in enumerate(comparison_runs):
                run_text_data = text_data[text_data["run_id"] == run_id]
                if not run_text_data.empty:
                    row = run_text_data.iloc[0]
                    true_subtype = row.get("true_subtype", "N/A")
                    pred_subtype = row.get("pred_subtype", "N/A")
                    correct = row.get("correct", False)

                    # Format: True: X → Pred: Y (✓/✗)
                    status_icon = "✓" if correct else "✗"
                    row_dict[f"run_{idx+1}_pred"] = f"{pred_subtype} ({status_icon})"
                    row_dict[f"run_{idx+1}_true"] = true_subtype

            rows.append(row_dict)

        # Create table columns dynamically
        columns = [{"name": "#", "id": "index"}]
        columns.append({"name": "Text Sample", "id": "text"})
        for idx, run_id in enumerate(comparison_runs, start=1):
            # Get the run number for this run_id
            run_info = run_data[run_data["run_id"] == run_id]
            run_number = run_info.iloc[0]["run_number"] if not run_info.empty else idx
            columns.append({"name": f"Run {run_number} - True", "id": f"run_{idx}_true"})
            columns.append({"name": f"Run {run_number} - Prediction", "id": f"run_{idx}_pred"})

        # Style cells for correct/incorrect predictions
        style_data_conditional = []
        for idx in range(len(comparison_runs)):
            # Highlight correct predictions in green, incorrect in red
            for row_idx, row in enumerate(rows):
                if f"run_{idx+1}_pred" in row:
                    pred_text = row[f"run_{idx+1}_pred"]
                    if "✓" in pred_text:
                        style_data_conditional.append({
                            "if": {
                                "column_id": f"run_{idx+1}_pred",
                                "row_index": row_idx
                            },
                            "backgroundColor": "#DCFCE7",
                            "color": "#166534"
                        })
                    else:
                        style_data_conditional.append({
                            "if": {
                                "column_id": f"run_{idx+1}_pred",
                                "row_index": row_idx
                            },
                            "backgroundColor": "#FEE2E2",
                            "color": "#991B1B"
                        })

        table = DataTable(
            columns=columns,
            data=rows,
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "fontSize": "12px",
                "fontFamily": "monospace",
                "color": "#1F2937",
                "whiteSpace": "normal",
                "maxWidth": "200px",
            },
            style_header={
                "backgroundColor": "#E2E8F0",
                "fontWeight": "600",
                "fontSize": "12px",
                "padding": "10px",
                "color": "#1F2937",
                "fontFamily": "sans-serif",
                "textAlign": "center",
            },
            style_data={
                "borderBottom": "1px solid #E2E8F0",
            },
            style_data_conditional=style_data_conditional,
            page_size=10,
        )

        card = dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.P(
                            f"Showing {len(rows)} datapoints classified by all {len(comparison_runs)} runs",
                            className="text-sm text-gray-600 mb-3"
                        ),
                        table,
                    ]
                )
            ]
        )

        # Store the rows data for export
        return card, rows

    # Export comparison datapoints to CSV
    app.clientside_callback(
        """
        function(n_clicks, data) {
            if (!n_clicks || !data || data.length === 0) return null;
            const columns = Object.keys(data[0]);
            let csv = columns.join(',') + '\\n';
            data.forEach(row => {
                const values = columns.map(col => {
                    const value = row[col];
                    if (value === null || value === undefined) return '';
                    const str = String(value).replace(/"/g, '""');
                    return `"${str}"`;
                });
                csv += values.join(',') + '\\n';
            });
            const filename = 'comparison_datapoints.csv';
            return {
                content: csv,
                filename: filename,
                type: 'text/csv',
                base64: false
            };
        }
        """,
        Output("download-comparison-csv", "data"),
        Input("export-comparison-csv-btn", "n_clicks"),
        State("comparison-data-store", "data"),
        prevent_initial_call=True,
    )


