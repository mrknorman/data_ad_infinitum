from pathlib import Path

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Category20  # A palette with 20 distinct colors

import numpy as np
import gravyflow as gf

def transform_string(s):
    # Remove the 'perceptron_' prefix and split by underscore
    parts = s.replace('perceptron_', '').split('_')

    # Remove empty strings and convert to integers
    parts = [int(x) for x in parts if x]

    # Append 1 if there are less than 2 parts
    parts.append(1)

    return ', '.join(map(str, parts))

def plot_metrics(metrics_dict):
    if len(metrics_dict) > 20:
        raise ValueError("Number of series exceeds the number of available colors in the palette")

    plot_widths = [1200] * 4  # Default width for each plot

    # Create a figure for each metric
    figures = []

    for i, key in enumerate(['binary_accuracy', 'loss', 'val_binary_accuracy', 'val_loss']):
        # Use specified width for each plot, or default if not enough widths are specified
        width = plot_widths[i] if i < len(plot_widths) else plot_widths[-1]

        p = figure(title=key, x_axis_label='Epochs', y_axis_label=key, width=width)
        
        # Plot each series in the list
        for idx, (name, metrics) in enumerate(metrics_dict.items()):
            if key in metrics:
                color = Category20[20][idx % 20]
                p.line(
                    np.arange(len(metrics[key])), 
                    metrics[key], 
                    legend_label=name, 
                    line_color=color,
                )

        figures.append(p)

    # Layout the plots in a grid
    grid = gridplot(figures, ncols=2)

    # Output file
    output_file("metrics.html")
    save(grid)

histories = {}

directory_path = Path('./models/chapter_04_perceptrons_single/')
for entry in directory_path.iterdir():
    if entry.name.startswith("perceptron"):
        name = transform_string(entry.name)

        histories[name] = gf.load_history(entry)

plot_metrics(histories)