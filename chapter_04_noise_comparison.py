# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
from copy import deepcopy

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Bright
from tqdm import tqdm

# Local imports:
from py_ml_tools.setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from py_ml_tools.noise import NoiseObtainer, NoiseType
from py_ml_tools.acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                          DataQuality, DataLabel, IFO)
from py_ml_tools.plotting import (generate_strain_plot, generate_spectrogram, 
                                  create_info_panel)
from py_ml_tools.dataset import get_ifo_dataset, ReturnVariables

def generate_noise_comparisons(
    output_diretory_path : Path = Path("./figures")
    ):

    # Test Parameters:
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21

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
        ]
    }
    
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
    
    noise_data : Dict = {}
    
    noise_types = {
        "White Noise" :  NoiseType.WHITE,
        "Coloured Noise" : NoiseType.COLORED,
        "Pseudo-Real Noise" : NoiseType.PSEUDO_REAL,
        "Real Noise" : NoiseType.REAL
    }
    
    layout = []
    for color, (type_name, noise_type) in zip(Bright[7], noise_types.items()):
        # Copy default arguments:
        noise_args : Dict = deepcopy(dataset_args)

        # Initilise noise generator wrapper:
        noise_obtainer : NoiseObtainer = \
            NoiseObtainer(
                ifo_data_obtainer=ifo_data_obtainer,
                noise_type=noise_type,
                ifos=IFO.L1,
            )
        
        # Update argument dictionary which will act as parameters to initilise
        # dataset:
        noise_args.update(
            {"noise_obtainer" : noise_obtainer}
        )
        
        # Setupd data generator with updated noise arguments:
        dataset : tf.data.Dataset = get_ifo_dataset(
            **noise_args
        )
        
        # Get one batch from dataset and convert onsource data to numpy array
        # for plotting:
        input_dict, _ = next(iter(dataset))
        onsource = input_dict[ReturnVariables.ONSOURCE.name].numpy()[0]   
        whitened_onsource = input_dict[ReturnVariables.WHITENED_ONSOURCE.name].numpy()[0]   
        
        gps_time = input_dict[ReturnVariables.GPS_TIME.name].numpy()[0]    
        
        if (gps_time == -1):
            gps_time = "N/A"
            
        if noise_type == "White Noise":
            scale_factor = 1.0
        else:
            scale_factor = 1.0E21
            
        # Add strain to list of strain plots to be assmbled into to grid plot:
        layout.append(
            [
            create_info_panel({
                "Type": f"{type_name}.",
                "Whitened": "False.",
                "Scale Factor": f"{scale_factor}.",
                "Gps Time": f"{gps_time}."
            }), 
            generate_strain_plot(
                {
                    type_name: onsource,
                },
                sample_rate_hertz,
                onsource_duration_seconds,
                scale_factor=scale_factor,
                colors = [color],
                has_legend = False
            ),
            generate_strain_plot(
                {
                    f"Whitened {type_name}": whitened_onsource,
                },
                sample_rate_hertz,
                onsource_duration_seconds,
                scale_factor=scale_factor,
                colors = [color],
                has_legend = False
            ),
            create_info_panel({
                "Type": f"Whitened {type_name}.",
                "Whitened": "True.",
                "Scale Factor": f"{scale_factor}.",
                "GPS Time": f"{gps_time}."
            })
            ] 
        )
        
    # Ensure output directory exists:
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard:
    output_file(output_diretory_path / "04_noise_type_comparison.html")

    # Arrange the plots in a grid:
    grid = gridplot(layout)
    
    # Save the plots as html file:
    save(grid)


if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
        
    # Test IFO noise generator:
    with strategy.scope():
                
        generate_noise_comparisons()    
    