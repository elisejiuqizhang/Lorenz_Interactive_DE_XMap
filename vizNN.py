import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

# Function to load data based on noiseType, noiseWhen, noiseAddType, and noiseLevel
def load_noise_data(noiseType, noiseWhen, noiseAddType, noiseLevel, data_dir='/Users/elise/Documents/Recherches/Projects/Causality/Lorenz_Interactive_Viz/Lorenz'):
    if noiseLevel == 0:
        file_name = 'noNoise.csv'
    else:
        file_name = f"{noiseType}_{noiseWhen}_{noiseAddType}_{round(noiseLevel, 2)}.csv"
    file_path = os.path.join(data_dir, file_name)
    data = pd.read_csv(file_path)
    # Fill NaN values with the mean of the column
    data = data.fillna(data.mean())
    return data

# Function to downsample data
def downsample_data(data, downsample_type, downsample_factor):
    if downsample_type is None or downsample_type.lower() == 'none':
        return data

    n = len(data)
    downsampled_data = np.zeros((n // downsample_factor, data.shape[1]))

    if downsample_type.lower() in ['a', 'av', 'average']:
        for i in range(0, n, downsample_factor):
            segment = data[i:i + downsample_factor]
            if len(segment) == downsample_factor:
                downsampled_data[i // downsample_factor] = np.mean(segment, axis=0)
    elif downsample_type.lower() in ['d', 'de', 'decimation', 'remove']:
        downsampled_data = data[::downsample_factor]
    elif downsample_type.lower() in ['s', 'sub', 'subsample', 'half', 'half-subsample']:
        for i in range(0, n, downsample_factor):
            segment = data[i:i + downsample_factor]
            if len(segment) == downsample_factor:
                rdm_start = np.random.randint(0, len(segment) // 2)
                downsampled_data[i // downsample_factor] = np.mean(segment[rdm_start:rdm_start + downsample_factor // 2], axis=0)
    else:
        raise ValueError('downsampleType not recognized')

    return downsampled_data

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
def create_3d_scatter_trace(embedding, name, time_indices):
    return go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        name=name,
        marker=dict(size=2, opacity=0.8),
        customdata=time_indices,  # Store time indices as custom data
        hovertemplate='Time Index: %{customdata}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}'
    )

# Function to find the k-nearest neighbors of a point in an embedding
def find_nearest_neighbors(embedding, point_index, k):
    tree = cKDTree(embedding)
    distances, indices = tree.query(embedding[point_index], k=k)
    return indices

# Initialize Dash app
app = dash.Dash(__name__)

# Default camera settings for initial zoom
default_camera = dict(
    eye=dict(x=0.45, y=0.3, z=0.6)
)

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Lorenz System Manifold Visualization with Delay Embeddings and Nearest Neighbors"),
        html.Div([
            html.Div([
                html.Label('Noise Type:'),
                dcc.Dropdown(
                    id='noise-type-dropdown',
                    options=[{'label': nt, 'value': nt} for nt in ['gNoise', 'lpNoise']],
                    value='gNoise'
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '6px'}),
            html.Div([
                html.Label('Noise When:'),
                dcc.Dropdown(
                    id='noise-when-dropdown',
                    options=[{'label': nw, 'value': nw} for nw in ['in', 'post']],
                    value='in'
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '6px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div([
            html.Div([
                html.Label('Noise Addition Type:'),
                dcc.Dropdown(
                    id='noise-add-type-dropdown',
                    options=[{'label': nat, 'value': nat} for nat in ['add', 'mult', 'both']],
                    value='add'
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '6px'}),
            html.Div([
                html.Label('Delay Value:'),
                dcc.Dropdown(
                    id='delay-dropdown',
                    options=[{'label': str(d), 'value': d} for d in range(1, 6)],
                    value=1
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '6px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div([
            html.Div([
                html.Label('Downsampling Type:'),
                dcc.Dropdown(
                    id='downsampling-type-dropdown',
                    options=[{'label': 'none', 'value': 'none'},
                             {'label': 'average', 'value': 'average'},
                             {'label': 'remove', 'value': 'remove'},
                             {'label': 'half-subsample', 'value': 'half-subsample'}],
                    value='none'
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '6px'}),
            html.Div([
                html.Label('Downsampling Factor:'),
                dcc.Dropdown(
                    id='downsampling-factor-dropdown',
                    options=[{'label': str(f), 'value': f} for f in [5, 8, 10, 12, 15]],
                    value=5
                )
            ], style={'width': '70%', 'display': 'inline-block', 'padding': '6px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div([
            html.Label('Noise Level:'),
            dcc.Slider(
                id='noise-level-slider',
                min=0,
                max=0.75,
                step=0.05,
                value=0,
                marks={round(i * 0.05, 2): f'{round(i * 0.05, 2)}' for i in range(16)}
            )
        ], style={'width': '110%', 'padding': '6px'}),
        html.Div([
            html.Label('Number of Nearest Neighbors:'),
            dcc.Dropdown(
                id='neighbors-dropdown',
                options=[{'label': str(k), 'value': k} for k in range(1, 21)],
                value=5
            )
        ], style={'width': '60%', 'padding': '6px'})
    ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '6px'}),
    html.Div([
        dcc.Graph(id='manifold-graph', style={'height': '80vh'})
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '6px'}),
    html.Div([
        html.P("Author: Elise Zhang"),
        html.P("© 2024 Elise Zhang. Licensed under the MIT License.")
    ], style={'textAlign': 'center', 'padding': '10px'})
])

# Callback to update the graph based on selections
@app.callback(
    Output('manifold-graph', 'figure'),
    [Input('noise-type-dropdown', 'value'),
     Input('noise-when-dropdown', 'value'),
     Input('noise-add-type-dropdown', 'value'),
     Input('noise-level-slider', 'value'),
     Input('delay-dropdown', 'value'),
     Input('downsampling-type-dropdown', 'value'),
     Input('downsampling-factor-dropdown', 'value'),
     Input('neighbors-dropdown', 'value'),
     Input('manifold-graph', 'clickData')],
    [State('manifold-graph', 'relayoutData')]
)
def update_graph(noise_type, noise_when, noise_add_type, noise_level, delay, downsample_type, downsample_factor, k_neighbors, click_data, relayoutData):
    noise_level = round(noise_level, 2)
    initial_data = load_noise_data(noise_type, noise_when, noise_add_type, noise_level)

    # Downsample the data
    downsampled_data = downsample_data(initial_data.values, downsample_type, downsample_factor)
    downsampled_df = pd.DataFrame(downsampled_data, columns=initial_data.columns)

    # Create delay embeddings for X, Y, Z with selected delay
    embeddings = {
        'Ground Truth': downsampled_df[['X', 'Y', 'Z']].values,
        'X Delay Embedding': create_delay_embedding(downsampled_df['X'].values, delay, 3),
        'Y Delay Embedding': create_delay_embedding(downsampled_df['Y'].values, delay, 3),
        'Z Delay Embedding': create_delay_embedding(downsampled_df['Z'].values, delay, 3)
    }

    time_indices = np.arange(len(embeddings['Ground Truth']))

    # Create traces for all embeddings
    traces = [create_3d_scatter_trace(embedding, name, time_indices) for name, embedding in embeddings.items()]

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

    # Extract the current camera view from the first subplot
    camera = default_camera
    if relayoutData and 'scene.camera' in relayoutData:
        camera = relayoutData['scene.camera']

    # Highlight nearest neighbors if a point is selected
    if click_data and click_data['points']:
        selected_point = click_data['points'][0]
        if 'customdata' in selected_point:
            selected_index = selected_point['customdata']
            selected_trace = selected_point['curveNumber']

            if selected_trace < 4:  # Ensure the clicked trace is within the embeddings
                selected_embedding_name = list(embeddings.keys())[selected_trace]
                selected_embedding = embeddings[selected_embedding_name]

                if selected_index < len(selected_embedding):
                    neighbor_indices = find_nearest_neighbors(selected_embedding, selected_index, k_neighbors)

                    # Add highlighted points to the corresponding embedding trace
                    for j, (trace_name, trace_embedding) in enumerate(embeddings.items()):
                        neighbor_points = trace_embedding[neighbor_indices]
                        highlight_trace = go.Scatter3d(
                            x=neighbor_points[:, 0],
                            y=neighbor_points[:, 1],
                            z=neighbor_points[:, 2],
                            mode='markers',
                            name=f'Neighbors in {trace_name}',
                            marker=dict(size=5, color='red', opacity=0.8),
                            customdata=neighbor_indices,  # Store indices as custom data
                            hovertemplate='Time Index: %{customdata}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}'
                        )
                        row = j // 2 + 1
                        col = j % 2 + 1
                        fig.add_trace(highlight_trace, row=row, col=col)

    # Apply the camera settings to each subplot
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
            yaxis=dict(title=f"Y(t-{delay})", range=[-30, 30]),
            zaxis=dict(title=f"Y(t-2*{delay})", range=[-30, 30]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        scene4=dict(
            camera=camera,
            aspectmode="cube",
            xaxis=dict(title="Z", range=[-1, 60]),
            yaxis=dict(title=f"Z(t-{delay})", range=[-1, 60]),
            zaxis=dict(title=f"Z(t-2*{delay})", range=[-1, 60]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        height=900
    )

    return fig

if __name__ == '__main__':
    # Run the app
    app.run_server(debug=True, use_reloader=False)
