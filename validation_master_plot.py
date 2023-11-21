
import glob
import os
from pathlib import Path
from bokeh.palettes import Category20  # A palette with 20 distinct colors

import gravyflow as gf

def transform_string(s):
    # Remove the 'perceptron_' prefix and split by underscore
    parts = s.replace('perceptron_', '').split('_')

    # Remove empty strings and convert to integers
    parts = [int(x) for x in parts if x]

    # Append 1 if there are less than 2 parts
    parts.append(1)

    return ', '.join(map(str, parts))

def get_perceptron_names():
    pattern = './models/chapter_04_perceptrons_single/perceptron*/validation_data.h5'
    directories = glob.glob(pattern)
    names = {}

    for dir in directories:
        parts = os.path.normpath(dir).split(os.sep)
        # Assuming the name is always in the second to last position
        name = parts[-2].replace('perceptron', '')

        names[transform_string(name)] = os.path.normpath(dir)

    return names

if __name__ == "__main__":

    validation_file_paths = get_perceptron_names()

    validators = []
    for key, value in validation_file_paths.items():
        validators.append(gf.Validator.load(value))

    validators[0].plot(
        Path("./models/chapter_04_perceptrons_single/master_validation_plot.html"), 
        comparison_validators = validators[1:],
        colors=Category20[20],
        width=1200
    )