import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback_context, no_update

from data import generate_mock_data


# Prepare a small deck of datapoints to review
run_data, detailed_data, test_run_id, _ = generate_mock_data()
deck = detailed_data[detailed_data["run_id"] == test_run_id][
    ["text", "pred_subtype", "true_subtype"]
].head(50)
deck["suggestion"] = ""
deck_records = deck.to_dict("records")


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container(
    [
        html.H1(
            "Approve Me – Swipe the Predictions",
            className="text-center my-4 text-indigo-900",
        ),
        html.P(
            "Think Tinder for labels: swipe right to approve the prediction, swipe up to bless the truth, or suggest a new match.",
            className="text-center text-muted mb-4",
        ),
        dcc.Store(id="review-items", data=deck_records),
        dcc.Store(id="current-index", data=0),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id="card-shell",
                            children=[
                                html.Div(
                                    id="card-text",
                                    className="p-3 rounded bg-light text-dark mb-3",
                                    style={
                                        "minHeight": "160px",
                                        "fontSize": "16px",
                                        "border": "1px solid #e2e8f0",
                                        "boxShadow": "0 10px 20px rgba(0,0,0,0.08)",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Small("Predicted", className="text-muted"),
                                                html.Div(
                                                    id="card-pred",
                                                    className="fw-bold py-1 px-2 rounded-pill bg-danger text-white",
                                                ),
                                            ],
                                            className="me-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Small("Truth", className="text-muted"),
                                                html.Div(
                                                    id="card-true",
                                                    className="fw-bold py-1 px-2 rounded-pill bg-success text-white",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="d-flex align-items-center mb-3",
                                ),
                            ],
                            className="p-4 rounded-4 bg-white shadow-sm",
                        ),
                        html.Div(
                            [
                                dbc.Button(
                                    "❤️ Approve Prediction",
                                    id="approve-pred-btn",
                                    color="danger",
                                    className="me-2 my-2",
                                ),
                                dbc.Button(
                                    "👍 Approve Truth",
                                    id="approve-true-btn",
                                    color="success",
                                    className="me-2 my-2",
                                ),
                                dbc.InputGroup(
                                    [
                                        dbc.Input(
                                            id="suggest-pred-input",
                                            placeholder="Suggest a better subtype…",
                                        ),
                                        dbc.Button(
                                            "✨ Submit",
                                            id="submit-suggestion-btn",
                                            color="warning",
                                        ),
                                    ],
                                    className="my-2",
                                ),
                            ],
                            className="d-flex flex-column flex-md-row align-items-stretch align-items-md-center",
                        ),
                        html.Div(id="feedback", className="mt-2 text-secondary"),
                        html.Div(id="progress-text", className="mt-1 text-muted"),
                        html.Hr(),
                        html.H5("Deck Overview", className="mt-3"),
                        dbc.Table(
                            id="review-table",
                            bordered=True,
                            striped=True,
                            hover=True,
                            responsive=True,
                            className="text-sm",
                        ),
                    ],
                    width=12,
                    lg=8,
                    className="mx-auto",
                )
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    [
        Output("card-text", "children"),
        Output("card-pred", "children"),
        Output("card-true", "children"),
        Output("progress-text", "children"),
    ],
    [
        Input("review-items", "data"),
        Input("current-index", "data"),
    ],
)
def update_card(items, idx):
    if not items:
        return "No datapoints available.", "", "", ""
    idx = idx or 0
    idx = min(idx, len(items) - 1)
    item = items[idx]
    text = item.get("text", "No text")
    pred = item.get("pred_subtype", "N/A")
    truth = item.get("true_subtype", "N/A")
    progress = f"Card {idx + 1} of {len(items)}"
    return text, pred, truth, progress


@app.callback(
    [
        Output("current-index", "data"),
        Output("feedback", "children"),
        Output("review-items", "data"),
    ],
    [
        Input("approve-pred-btn", "n_clicks"),
        Input("approve-true-btn", "n_clicks"),
        Input("submit-suggestion-btn", "n_clicks"),
    ],
    [
        State("current-index", "data"),
        State("review-items", "data"),
        State("suggest-pred-input", "value"),
    ],
    prevent_initial_call=True,
)
def handle_action(pred_clicks, true_clicks, suggest_clicks, idx, items, suggestion):
    if not items:
        return 0, "Deck is empty.", items
    idx = idx or 0
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    feedback = ""
    items = items.copy()
    if triggered == "approve-pred-btn":
        feedback = "🔥 Swipe right! Prediction approved."
    elif triggered == "approve-true-btn":
        feedback = "☝️ Swipe up! Truth stands tall."
    elif triggered == "submit-suggestion-btn":
        if suggestion:
            feedback = f"💡 New suggestion noted: {suggestion}"
            if 0 <= idx < len(items):
                items[idx]["suggestion"] = suggestion
        else:
            return no_update, "Please type a suggestion before submitting.", no_update
    next_idx = idx + 1 if idx + 1 < len(items) else 0
    return next_idx, feedback, items


@app.callback(
    Output("review-table", "children"),
    Input("review-items", "data"),
)
def render_table(items):
    if not items:
        return html.Div("No datapoints.")
    header = html.Thead(
        html.Tr(
            [
                html.Th("#"),
                html.Th("Text"),
                html.Th("Predicted"),
                html.Th("Truth"),
                html.Th("Suggestion"),
            ]
        )
    )
    body_rows = []
    for i, row in enumerate(items, start=1):
        body_rows.append(
            html.Tr(
                [
                    html.Td(i),
                    html.Td(row.get("text", "")[:80] + ("..." if len(row.get("text", "")) > 80 else "")),
                    html.Td(row.get("pred_subtype", "")),
                    html.Td(row.get("true_subtype", "")),
                    html.Td(row.get("suggestion", "")),
                ]
            )
        )
    body = html.Tbody(body_rows)
    return [header, body]


if __name__ == "__main__":
    app.run(debug=False, port=8051)
