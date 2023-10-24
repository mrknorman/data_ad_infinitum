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
from tqdm import tqdm

# Local imports:
from py_ml_tools.maths import Distribution, DistributionType
from py_ml_tools.setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from py_ml_tools.injection import (cuPhenomDGenerator, InjectionGenerator, 
                         WaveformParameters, WaveformGenerator, ScalingMethod, 
                         ScalingTypes, IncoherentGenerator)
from py_ml_tools.acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                          DataQuality, DataLabel, IFO)
from py_ml_tools.noise import NoiseObtainer, NoiseType
from py_ml_tools.plotting import generate_strain_plot, generate_correlation_plot
from py_ml_tools.dataset import get_ifo_dataset, get_ifo_data, ReturnVariables

def plot_pearson_correlation(
    num_tests : int = 32,
    output_diretory_path : Path = Path("./figures/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_tests
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    ifos = [IFO.L1, IFO.H1]
    
    # Define injection directory path:
    injection_directory_path : Path = \
        Path("./py_ml_tools/tests/example_injection_parameters")
    
    # Intilise Scaling Method:
    scaling_method = \
        ScalingMethod(
            Distribution(min_=8.0,max_=15.0,type_=DistributionType.UNIFORM),
            ScalingTypes.SNR
        )
    
    # Load injection config:
    phenom_d_generator : cuPhenomDGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "phenom_d_parameters.json", 
            sample_rate_hertz, 
            onsource_duration_seconds,
            scaling_method=scaling_method,
            network=ifos
        )
    
    wnb_generator : WNBGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "wnb_parameters.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    incoherent_generator = IncoherentGenerator(
        [phenom_d_generator, wnb_generator]
    )
    
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
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: NoiseObtainer = \
        NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=NoiseType.REAL,
            ifos=ifos
        )
    
    dataset : tf.data.Dataset = get_ifo_dataset(
        # Random Seed:
        seed= 1000,
        # Temporal components:
        sample_rate_hertz=sample_rate_hertz,   
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        # Noise: 
        noise_obtainer=noise_obtainer,
        # Injections:
        injection_generators=incoherent_generator, 
        # Output configuration:
        num_examples_per_batch=num_examples_per_batch,
        input_variables = [
            ReturnVariables.WHITENED_ONSOURCE, 
            ReturnVariables.INJECTION_MASKS, 
            ReturnVariables.INJECTIONS,
            ReturnVariables.WHITENED_INJECTIONS,
            ReturnVariables.ROLLING_PEARSON_ONSOURCE
        ],
    )
    
    input_dict, _ = next(iter(dataset))
        
    onsource = input_dict[ReturnVariables.WHITENED_ONSOURCE.name].numpy()
    injections = input_dict[ReturnVariables.INJECTIONS.name].numpy()
    correlation = input_dict[ReturnVariables.ROLLING_PEARSON_ONSOURCE.name].numpy()
    whitened_injections = input_dict[
        ReturnVariables.WHITENED_INJECTIONS.name
    ].numpy()
    masks = input_dict[ReturnVariables.INJECTION_MASKS.name].numpy()
    
    layout = [
        generate_strain_plot(
            {
                "Whitened Onsouce + Injection": onsource_,
                "Whitened Injection" : whitened_injection,
                "Injection": injection
            },
            sample_rate_hertz,
            onsource_duration_seconds,
            scale_factor=scale_factor
        ) + 
        [generate_correlation_plot(
            correlation_,
            sample_rate_hertz
        )]
        for onsource_, whitened_injection, injection, correlation_ in zip(
            onsource,
            whitened_injections[0],
            injections[0],
            correlation
        )
    ]
        
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "04_pearson_plots.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
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
    
    # Test IFO noise dataset:
    with strategy.scope():
        plot_pearson_correlation()
