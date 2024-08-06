import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

# Function to load data based on noiseType, noiseWhen, noiseAddType, and noiseLevel
def load_noise_data(noiseType, noiseWhen, noiseAddType, noiseLevel, data_dir='/Users/elise/Documents/Recherches/Projects/Causality/Lorenz_Interactive_Viz/Lorenz'):
    if noiseLevel == 0:
        file_name = 'noNoise.csv'
    else:
        file_name = f"{noiseType}_{noiseWhen}_{noiseAddType}_{round(noiseLevel, 2)}.csv"
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)

# Function to create time delay embeddings
def create_delay_embedding(series, delay=1, dimension=3):
    n = len(series)
    if n < (dimension - 1) * delay + 1:
        raise ValueError("Time series is too short for the given delay and dimension.")
    embedding = np.zeros((n - (dimension - 1) * delay, dimension))
    for i in range(dimension):
        embedding[:, i] = series[i * delay: n - (dimension - 1 - i) * delay]
    return embedding

# Function to create 3D scatter plot traces for an embedding
def create_3d_scatter_trace(embedding, name):
    return go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        name=name,
        marker=dict(size=2, opacity=0.8)
    )

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Lorenz System Manifold Visualization with Delay Embeddings"),
        html.Div([
            html.Div([
                html.Label('Noise Type:'),
                dcc.Dropdown(
                    id='noise-type-dropdown',
                    options=[{'label': nt, 'value': nt} for nt in ['gNoise', 'lpNoise']],
                    value='gNoise'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.Label('Noise When:'),
                dcc.Dropdown(
                    id='noise-when-dropdown',
                    options=[{'label': nw, 'value': nw} for nw in ['in', 'post']],
                    value='in'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div([
            html.Div([
                html.Label('Noise Addition Type:'),
                dcc.Dropdown(
                    id='noise-add-type-dropdown',
                    options=[{'label': nat, 'value': nat} for nat in ['add', 'mult', 'both']],
                    value='add'
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.Label('Delay Value:'),
                dcc.Dropdown(
                    id='delay-dropdown',
                    options=[{'label': str(d), 'value': d} for d in range(1, 6)],
                    value=1
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div([
            html.Label('Noise Level:'),
            dcc.Slider(
                id='noise-level-slider',
                min=0.05,
                # max=0.75,
                max=1.05,
                step=0.05,
                value=0.05,
                marks={round(float(i) * 0.05, 2): f'{round(float(i) * 0.05, 2)}' for i in range(int(1.05/0.05+1))}
            )
        ], style={'width': '98%', 'padding': '10px'})
    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    html.Div([
        dcc.Graph(id='manifold-graph', style={'height': '90vh'})
    ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'})
])

# Callback to update the graph based on selections
@app.callback(
    Output('manifold-graph', 'figure'),
    [Input('noise-type-dropdown', 'value'),
     Input('noise-when-dropdown', 'value'),
     Input('noise-add-type-dropdown', 'value'),
     Input('noise-level-slider', 'value'),
     Input('delay-dropdown', 'value')],
    [State('manifold-graph', 'relayoutData')]
)
def update_graph(noise_type, noise_when, noise_add_type, noise_level, delay, relayoutData):
    noise_level = round(float(noise_level), 2)
    initial_data = load_noise_data(noise_type, noise_when, noise_add_type, noise_level)

    # Create delay embeddings for X, Y, Z with selected delay
    embeddings = {
        'Ground Truth': initial_data[['X', 'Y', 'Z']].values,
        'X Delay Embedding': create_delay_embedding(initial_data['X'], delay, 3),
        'Y Delay Embedding': create_delay_embedding(initial_data['Y'], delay, 3),
        'Z Delay Embedding': create_delay_embedding(initial_data['Z'], delay, 3)
    }

    # Create traces for all embeddings
    traces = [create_3d_scatter_trace(embedding, name) for name, embedding in embeddings.items()]

    # Create the figure with four subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Ground Truth', 'X Delay Embedding', 'Y Delay Embedding', 'Z Delay Embedding')
    )

    # Add traces to the subplots
    for i, trace in enumerate(traces):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(trace, row=row, col=col)

    # Extract the current camera view for each subplot
    camera = None
    if relayoutData:
        camera_keys = ['scene.camera', 'scene2.camera', 'scene3.camera', 'scene4.camera']
        for key in camera_keys:
            if key in relayoutData:
                camera = relayoutData[key]
                break

    # Update the layout to scale axes appropriately
    fig.update_layout(
        title_text=f"Lorenz System Manifolds with Noise {noise_type}, {noise_when}, {noise_add_type} at Level {noise_level} and Delay {delay}",
        scene=dict(
            camera=camera,
            aspectmode="cube",
            xaxis=dict(title="X", range=[-30, 30]),
            yaxis=dict(title="Y", range=[-30, 30]),
            zaxis=dict(title="Z", range=[-1, 60]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        scene2=dict(
            camera=camera,
            aspectmode="cube",
            xaxis=dict(title="X(t)", range=[-30, 30]),
            yaxis=dict(title=f"X(t-{delay})", range=[-30, 30]),
            zaxis=dict(title=f"X(t-2*{delay})", range=[-30, 30]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        scene3=dict(
            camera=camera,
            aspectmode="cube",
            xaxis=dict(title="Y", range=[-30, 30]),
            yaxis=dict(title=f"Y-{delay}", range=[-30, 30]),
            zaxis=dict(title=f"Y-2*{delay}", range=[-30, 30]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        scene4=dict(
            camera=camera,
            aspectmode="cube",
            xaxis=dict(title="Z", range=[-1, 60]),
            yaxis=dict(title=f"Z-{delay}", range=[-1, 60]),
            zaxis=dict(title=f"Z-2*{delay}", range=[-1, 60]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        height=900
    )

    return fig

if __name__ == '__main__':
    # Run the app
    app.run_server(debug=True, use_reloader=False)
