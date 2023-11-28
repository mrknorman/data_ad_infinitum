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

def generate_sine_wave(
        frequency_hertz: float, 
        duration_seconds: float, 
        sample_rate_hertz: float
    ) -> np.ndarray:

    """
    Generate a sine wave of a given frequency, duration, and sample rate.

    Parameters:

    - frequency_hz (float): 
        Frequency of the sine wave in Hz.
    - duration_seconds (float): 
        Duration of the sine wave in seconds.
    - sample_rate_hertz (float): 
        Sample rate in Hz.
        
    Returns:
    - np.ndarray: Array containing the sine wave values.

    """

    t = np.linspace(
        0, 
        duration_seconds, 
        int(sample_rate_hertz * duration_seconds), 
        endpoint=False
    )

    y = 0.5 * np.sin(2 * np.pi * frequency_hertz * t)

    return y

def plot_example_psds(
    output_diretory_path : Path = Path("./figures")
    ):
    
    sample_rate_hertz : float = 256
    time_t_seconds : float = 0.7
    total_duration_seconds : float = 2.0
    num_samples : int = int(sample_rate_hertz*total_duration_seconds)
    extra_duration_seconds : float = total_duration_seconds - time_t_seconds
    
    sine_wave_20hz = generate_sine_wave(
        20, 
        time_t_seconds, 
        sample_rate_hertz
    )
    
    # Compute its PSD
    sine_wave_20hz: tf.Tensor = tf.constant(sine_wave_20hz, dtype=tf.float32)
    frequencies_tf_hertz, psd_tf = calculate_psd(
        sine_wave_20hz, nperseg=128, sample_rate_hertz=sample_rate_hertz
    )
    frequencies_tf_hertz, psd_tf = frequencies_tf_hertz.numpy(), psd_tf.numpy()
    
    sine_wave_40hz = generate_sine_wave(
        40, 
        extra_duration_seconds, 
        sample_rate_hertz
    )
    sine_wave_40hz: tf.Tensor = tf.constant(sine_wave_40hz, dtype=tf.float32)
    
    combined_wave = tf.concat([sine_wave_20hz, sine_wave_40hz], axis=0)
    
    
    combined_wave: tf.Tensor = tf.constant(combined_wave, dtype=tf.float32)
    combined_frequencies_tf_hertz, combined_psd_tf = calculate_psd(
        combined_wave, nperseg=128, sample_rate_hertz=sample_rate_hertz
    )
    combined_frequencies_tf_hertz, combined_psd_tf = \
        combined_frequencies_tf_hertz.numpy(), combined_psd_tf.numpy()
    
    zero_combined_signal : tf.Tensor = tf.concat(
        [sine_wave_20hz,
         tf.zeros([int(sample_rate_hertz*extra_duration_seconds)])
        ], 
        axis = 0
    )
    
    layout = [
        [
        generate_strain_plot(
            {"20 and 40 hertz signal" : combined_wave.numpy(), 
             "40 hertz signal": zero_combined_signal.numpy()},
            sample_rate_hertz,
            total_duration_seconds,
            has_legend = False
        ),
         generate_psd_plot(
            {"20 and 40 hertz signal" : combined_psd_tf, 
             "40 hertz signal" : psd_tf, 
            },
            frequencies_tf_hertz,
            has_legend = False
        )
        ]
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "04_psd_example.html")
    
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
        plot_example_psds()