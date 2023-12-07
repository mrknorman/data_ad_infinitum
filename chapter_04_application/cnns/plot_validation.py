
import glob
import os
import sys
from pathlib import Path
import numpy as np
from bokeh.palettes import Bright  # A palette with 20 distinct colors

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import gravyflow as gf

def snake_case_to_capitalised_first_with_spaces(text):
    """
    Convert a string from snake_case to Capitalised First With Spaces format.
    """

    text.replace('val', 'validate') if 'val' in text else text
    # Split the string at underscores
    words = text.split('_')
    
    # Capitalise the first letter of each word and join them with spaces
    return ' '.join(word.capitalize() for word in words)

def transform_string(s):
    # Remove the 'perceptron_' prefix and split by underscore
    name = s.replace('cnn_', '')

    return snake_case_to_capitalised_first_with_spaces(name)

def get_cnn_names():
    pattern = f'{current_dir}/models/cnn*/validation_data.h5'
    directories = glob.glob(pattern)
    names = {}

    for dir in directories:
        parts = os.path.normpath(dir).split(os.sep)
        # Assuming the name is always in the second to last position
        name = parts[-2].replace('cnn_', '')

        names[transform_string(name)] = os.path.normpath(dir)

    return names

if __name__ == "__main__":

    validation_file_paths = get_cnn_names()

    validators = []
    for key, value in validation_file_paths.items():
        validator = gf.Validator.load(value)
        validator.name = key
        validators.append(validator)

    validators[0].plot(
        Path(f"{current_dir}/models/validation_plot.html"), 
        comparison_validators = validators[1:],
        colors=Bright[7],
        width=1200
    )