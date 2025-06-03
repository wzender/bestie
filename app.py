import dash
from layout import create_layout
from callbacks import register_callbacks
from data import generate_mock_data, benchmark_options
import os

# Initialize Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder="assets"
)

# Custom index string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="/assets/bootstrap-flatly.css">
        <link rel="stylesheet" href="/assets/tailwind.css">
        <style>
            .alert .btn-close {
                background: transparent;
                color: #1F2937 !important;
                opacity: 1;
            }
        </style>
    </head>
    <body class="bg-slate-50">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Generate data
run_data, detailed_data, test_run_id, unique_benchmarks = generate_mock_data()

# Set layout
app.layout = create_layout(benchmark_options, run_data)

# Register callbacks
register_callbacks(app, run_data, detailed_data, test_run_id)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, port=port, host="0.0.0.0")