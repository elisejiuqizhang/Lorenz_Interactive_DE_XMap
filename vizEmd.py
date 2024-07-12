import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import plotly.io as pio

# Function to load data based on noiseType, noiseWhen, noiseAddType, and noiseLevel
def load_noise_data(noiseType, noiseWhen, noiseAddType, noiseLevel, data_dir='/Users/elise/Documents/Recherches/Projects/Causality/Lorenz_Interactive_Viz/Lorenz'):
    if noiseLevel == 0:
        file_name = 'noNoise.csv'
    else:
        file_name = f"{noiseType}_{noiseWhen}_{noiseAddType}_{round(noiseLevel, 2)}.csv"
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)

# List of noise types, noise whens, noise add types, and noise levels
noise_types = ['gNoise', 'lpNoise']
noise_whens = ['in', 'post']
noise_add_types = ['add', 'mult', 'both']
noise_levels = [round(i*0.05, 2) for i in range(1, 16)]  # from 0.05 to 0.75

# Initial data loading for default view
initial_noise_type = 'gNoise'
initial_noise_when = 'in'
initial_noise_add_type = 'add'
initial_noise_level = 0.05

# Create initial plot
initial_data = load_noise_data(initial_noise_type, initial_noise_when, initial_noise_add_type, initial_noise_level)
trace = go.Scatter3d(
    x=initial_data['X'],
    y=initial_data['Y'],
    z=initial_data['Z'],
    mode='markers',
    marker=dict(size=2, opacity=0.8)
)

layout = go.Layout(
    title=f'Manifold with {initial_noise_type} {initial_noise_when} {initial_noise_add_type} at Noise Level {initial_noise_level}',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

fig = go.Figure(data=[trace], layout=layout)

# Add dropdowns and sliders for parameter selection
fig.update_layout(
    updatemenus=[
        dict(
            buttons=[
                dict(
                    args=[{"x": [load_noise_data('gNoise', 'in', 'add', initial_noise_level)['X']],
                           "y": [load_noise_data('gNoise', 'in', 'add', initial_noise_level)['Y']],
                           "z": [load_noise_data('gNoise', 'in', 'add', initial_noise_level)['Z']]}],
                    label="gNoise in add",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('gNoise', 'post', 'add', initial_noise_level)['X']],
                           "y": [load_noise_data('gNoise', 'post', 'add', initial_noise_level)['Y']],
                           "z": [load_noise_data('gNoise', 'post', 'add', initial_noise_level)['Z']]}],
                    label="gNoise post add",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('lpNoise', 'in', 'add', initial_noise_level)['X']],
                           "y": [load_noise_data('lpNoise', 'in', 'add', initial_noise_level)['Y']],
                           "z": [load_noise_data('lpNoise', 'in', 'add', initial_noise_level)['Z']]}],
                    label="lpNoise in add",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('lpNoise', 'post', 'add', initial_noise_level)['X']],
                           "y": [load_noise_data('lpNoise', 'post', 'add', initial_noise_level)['Y']],
                           "z": [load_noise_data('lpNoise', 'post', 'add', initial_noise_level)['Z']]}],
                    label="lpNoise post add",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('gNoise', 'in', 'mult', initial_noise_level)['X']],
                           "y": [load_noise_data('gNoise', 'in', 'mult', initial_noise_level)['Y']],
                           "z": [load_noise_data('gNoise', 'in', 'mult', initial_noise_level)['Z']]}],
                    label="gNoise in mult",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('gNoise', 'post', 'mult', initial_noise_level)['X']],
                           "y": [load_noise_data('gNoise', 'post', 'mult', initial_noise_level)['Y']],
                           "z": [load_noise_data('gNoise', 'post', 'mult', initial_noise_level)['Z']]}],
                    label="gNoise post mult",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('lpNoise', 'in', 'mult', initial_noise_level)['X']],
                           "y": [load_noise_data('lpNoise', 'in', 'mult', initial_noise_level)['Y']],
                           "z": [load_noise_data('lpNoise', 'in', 'mult', initial_noise_level)['Z']]}],
                    label="lpNoise in mult",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('lpNoise', 'post', 'mult', initial_noise_level)['X']],
                           "y": [load_noise_data('lpNoise', 'post', 'mult', initial_noise_level)['Y']],
                           "z": [load_noise_data('lpNoise', 'post', 'mult', initial_noise_level)['Z']]}],
                    label="lpNoise post mult",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('gNoise', 'in', 'both', initial_noise_level)['X']],
                           "y": [load_noise_data('gNoise', 'in', 'both', initial_noise_level)['Y']],
                           "z": [load_noise_data('gNoise', 'in', 'both', initial_noise_level)['Z']]}],
                    label="gNoise in both",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('gNoise', 'post', 'both', initial_noise_level)['X']],
                           "y": [load_noise_data('gNoise', 'post', 'both', initial_noise_level)['Y']],
                           "z": [load_noise_data('gNoise', 'post', 'both', initial_noise_level)['Z']]}],
                    label="gNoise post both",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('lpNoise', 'in', 'both', initial_noise_level)['X']],
                           "y": [load_noise_data('lpNoise', 'in', 'both', initial_noise_level)['Y']],
                           "z": [load_noise_data('lpNoise', 'in', 'both', initial_noise_level)['Z']]}],
                    label="lpNoise in both",
                    method="restyle"
                ),
                dict(
                    args=[{"x": [load_noise_data('lpNoise', 'post', 'both', initial_noise_level)['X']],
                           "y": [load_noise_data('lpNoise', 'post', 'both', initial_noise_level)['Y']],
                           "z": [load_noise_data('lpNoise', 'post', 'both', initial_noise_level)['Z']]}],
                    label="lpNoise post both",
                    method="restyle"
                ),
            ],
            direction="down",
            showactive=True,
            x=0.17,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )
    ]
)

fig.update_layout(
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": "Noise Level: "},
            pad={"t": 50},
            steps=[
                dict(
                    label=str(round(i*0.05, 2)),
                    method="update",
                    args=[{"x": [load_noise_data(initial_noise_type, initial_noise_when, initial_noise_add_type, round(i*0.05, 2))['X']],
                           "y": [load_noise_data(initial_noise_type, initial_noise_when, initial_noise_add_type, round(i*0.05, 2))['Y']],
                           "z": [load_noise_data(initial_noise_type, initial_noise_when, initial_noise_add_type, round(i*0.05, 2))['Z']]}]
                ) for i in range(1, 16)
            ]
        )
    ]
)

# Save as an HTML file
output_path = 'lorenz_system_manifold_plot.html'
pio.write_html(fig, file=output_path, auto_open=False)

