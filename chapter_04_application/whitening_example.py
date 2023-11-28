# Built-In imports:
import logging
from pathlib import Path
from itertools import islice

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Bright
from bokeh.models import BoxAnnotation
from bokeh.colors import RGB
from tqdm import tqdm
import tensorflow as tf

from gravyflow.psd import calculate_psd
from gravyflow.maths import set_random_seeds
from gravyflow.setup import (find_available_GPUs, setup_cuda, 
                                     ensure_directory_exists)
from gravyflow.plotting import generate_psd_plot, generate_strain_plot
from gravyflow.noise import NoiseObtainer, NoiseType
from gravyflow.acquisition import (IFODataObtainer, SegmentOrder, 
                                           ObservingRun, DataQuality, DataLabel, 
                                           IFO)
from gravyflow.dataset import get_ifo_dataset, ReturnVariables

def plot_whitening_examples(
    output_diretory_path : Path = Path("./figures")
    ):
    
    # Test Parameters:
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : IFODataObtainer = \
        IFODataObtainer(
            ObservingRun.O3, 
            DataQuality.BEST, 
            [
                DataLabel.NOISE, 
                DataLabel.GLITCHES
            ],
            SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False,
            logging_level=logging.INFO
        )
    
    noise_obtainer : NoiseObtainer = \
        NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=NoiseType.REAL,
            ifos=IFO.L1,
        )

    dataset_args : Dict = {
        # Random Seed:
        "seed" : 1000,
        # Temporal components:
        "sample_rate_hertz" : sample_rate_hertz,   
        "onsource_duration_seconds" : onsource_duration_seconds,
        "offsource_duration_seconds" : offsource_duration_seconds,
        "crop_duration_seconds" : crop_duration_seconds,
        # Output configuration:
        "num_examples_per_batch" : num_examples_per_batch,
        "input_variables" : [
            ReturnVariables.ONSOURCE,
            ReturnVariables.WHITENED_ONSOURCE,
            ReturnVariables.GPS_TIME
        ],
        "noise_obtainer" : noise_obtainer
    }
    
    # Setupd data generator with updated noise arguments:
    dataset : tf.data.Dataset = get_ifo_dataset(
        **dataset_args
    )

    # Get one batch from dataset and convert onsource data to numpy array
    # for plotting:
    input_dict, _ = next(iter(dataset))
    onsource = tf.cast(
        input_dict[ReturnVariables.ONSOURCE.name][0],
        dtype = tf.float32
    )
    whitened_onsource = tf.cast(
        input_dict[ReturnVariables.WHITENED_ONSOURCE.name][0],
        dtype = tf.float32
    )
    
    frequencies_hertz, onsource_psd = calculate_psd(
        onsource, nperseg=256, sample_rate_hertz=sample_rate_hertz
    )
    frequencies_hertz, whitened_psd = calculate_psd(
        whitened_onsource, nperseg=256, sample_rate_hertz=sample_rate_hertz
    )
    onsource_psd, whitened_psd = onsource_psd.numpy(), whitened_psd.numpy()
    onsource, whitened_onsource = onsource.numpy(), whitened_onsource.numpy()
    frequencies_hertz = frequencies_hertz.numpy()
    
    layout = [
        [
            generate_psd_plot(
                {"onsource_psd" : onsource_psd},
                frequencies_hertz,
                has_legend = False
            ),
            generate_psd_plot(
                {"whitened_onsource_psd" : whitened_psd},
                frequencies_hertz,
                has_legend = False,
                colors = Bright[7][1:]
            )
        ],
        [
            generate_strain_plot(
                {"onsource" : onsource},
                sample_rate_hertz,
                onsource_duration_seconds,
                has_legend = False
            ),
            generate_strain_plot(
                {"whitened_onsource" : whitened_onsource},
                sample_rate_hertz,
                onsource_duration_seconds,
                has_legend = False,
                colors = Bright[7][1:]
            )
        ]
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "04_whitnening_example.html")
    
    grid = gridplot(layout)
    
    save(grid)
    
def plot_onsource_offsource_comparison(
    output_diretory_path : Path = Path("./figures")
    ):
    
    # Test Parameters:
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 18.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : IFODataObtainer = \
        IFODataObtainer(
            ObservingRun.O3, 
            DataQuality.BEST, 
            [
                DataLabel.NOISE, 
                DataLabel.GLITCHES
            ],
            SegmentOrder.RANDOM,
            force_acquisition=True,
            cache_segments=False,
            logging_level=logging.INFO
        )
    
    noise_obtainer : NoiseObtainer = \
        NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=NoiseType.REAL,
            ifos=IFO.L1,
        )

    dataset_args : Dict = {
        # Random Seed:
        "seed" : 1000,
        # Temporal components:
        "sample_rate_hertz" : sample_rate_hertz,   
        "onsource_duration_seconds" : onsource_duration_seconds,
        "offsource_duration_seconds" : offsource_duration_seconds,
        "crop_duration_seconds" : crop_duration_seconds,
        # Output configuration:
        "num_examples_per_batch" : num_examples_per_batch,
        "input_variables" : [
            ReturnVariables.ONSOURCE,
            ReturnVariables.WHITENED_ONSOURCE,
            ReturnVariables.GPS_TIME
        ],
        "noise_obtainer" : noise_obtainer
    }
    
    # Setupd data generator with updated noise arguments:
    dataset : tf.data.Dataset = get_ifo_dataset(
        **dataset_args
    )

    # Get one batch from dataset and convert onsource data to numpy array
    # for plotting:
    input_dict, _ = next(iter(dataset))
    onsource = tf.cast(
        input_dict[ReturnVariables.ONSOURCE.name][0],
        dtype = tf.float32
    )
    
    onsource_plot = generate_strain_plot(
        {"onsource" : onsource},
        sample_rate_hertz,
        onsource_duration_seconds,
        has_legend = False,
        width = 1000,
        colors = [RGB(0, 0, 0)]
    )
    
    # Create BoxAnnotations for different segments
    off_source = BoxAnnotation(left=0, right=16, fill_alpha=0.5, fill_color=Bright[7][0])
    on_source = BoxAnnotation(left=16.5, right=17.5, fill_alpha=0.5, fill_color=Bright[7][2])
    buffer_1 = BoxAnnotation(left=16.0, right=16.5, fill_alpha=0.5, fill_color=Bright[7][1])
    buffer_2 = BoxAnnotation(left=17.5, right=18.0, fill_alpha=0.5, fill_color=Bright[7][1])

    # Add annotations to the plot
    onsource_plot.add_layout(off_source)
    onsource_plot.add_layout(on_source)
    onsource_plot.add_layout(buffer_1)
    onsource_plot.add_layout(buffer_2)
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "04_onsource_offsource_comparisons.html")
        
    save(onsource_plot)

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
        plot_onsource_offsource_comparison()
        plot_whitening_examples()