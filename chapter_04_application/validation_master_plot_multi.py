
import glob
import os
from pathlib import Path
import numpy as np
from bokeh.palettes import Category20  # A palette with 20 distinct colors

import gravyflow as gf

def transform_string(s):
    # Remove the 'perceptron_' prefix and split by underscore
    parts = s.replace('multi_perceptron_', '').split('_')

    # Remove empty strings and convert to integers
    parts = [int(x) for x in parts if x]

    # Append 1 if there are less than 2 parts
    parts.append(1)

    return ', '.join(map(str, parts))

def get_perceptron_names():
    pattern = './models/chapter_04_perceptrons_multi/multi_perceptron*/validation_data.h5'
    directories = glob.glob(pattern)
    names = {}

    for dir in directories:
        parts = os.path.normpath(dir).split(os.sep)
        # Assuming the name is always in the second to last position
        name = parts[-2].replace('multi_perceptron_', '')

        names[transform_string(name)] = os.path.normpath(dir)

    return names

if __name__ == "__main__":

    validation_file_paths = get_perceptron_names()

    validators = []
    for key, value in validation_file_paths.items():
        validator = gf.Validator.load(value)
        validator.name = key
        validators.append(validator)

    validators[0].plot(
        Path("./models/chapter_04_perceptrons_multi/master_validation_plot_multi.html"), 
        comparison_validators = validators[1:],
        colors=Category20[20],
        width=1200
    )