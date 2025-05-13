# app.py
from dash import Dash
from components.layout import layout
from callbacks.handlers import register_callbacks

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Magellan Leaderboard"

app.layout = layout
register_callbacks(app)

if __name__ == "__main__":
    render_com_port = 10000
    app.run(host="0.0.0.0", debug=True, port=render_com_port)
