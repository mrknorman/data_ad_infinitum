# Built-In imports:
import logging
from pathlib import Path
from itertools import islice

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Bright
from tqdm import tqdm
import tensorflow as tf

from gravyflow.psd import calculate_psd
from gravyflow.maths import set_random_seeds
from gravyflow.setup import (find_available_GPUs, setup_cuda, 
                               ensure_directory_exists)
from gravyflow.plotting import generate_psd_plot, generate_strain_plot

def plot_supernovae(
    output_diretory_path : Path = Path("./figures")
    ):
    
    data_path : Path = Path("./s11WW.h.dat")
    
    data = np.loadtxt(data_path)
    
    time = data[:, 0]
    plus = data[:, 1]
    sample_rate_hertz = time[-1] / len(time)
    duration_seconds = time[-1]
    
    layout = [
        [
        generate_strain_plot(
            {"supernovae signal" : plus},
            sample_rate_hertz,
            duration_seconds,
            has_legend = False,
            height = 500
        )
        ]
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "04_supernovae_example.html")
    
    grid = gridplot(layout)
    
    save(grid)    
    

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 10000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 4000
    
    set_random_seeds(100)
    
    # Setup CUDA
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = setup_cuda(
        "1", 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test injection generation:
    with strategy.scope():
        plot_supernovae()

