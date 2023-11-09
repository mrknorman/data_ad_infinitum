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
from bokeh.layouts import column
from tqdm import tqdm

# Local imports:
import gravyflow as gf

def generate_plots(
    dataset, 
    sample_rate_hertz, 
    onsource_duration_seconds, 
    scale_factor,
    has_injection,
    injection_type,
    coherent
    ):
    
    layout = []
    input_dict, _ = next(iter(dataset))
    
    onsource = input_dict[gf.ReturnVariables.WHITENED_ONSOURCE.name].numpy()
    onsource_ = onsource[0]
    
    correlation = input_dict[gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE.name].numpy()
    correlation_ = correlation[0]

    if has_injection:
        injections = input_dict[gf.ReturnVariables.INJECTIONS.name].numpy()
        whitened_injections = input_dict[gf.ReturnVariables.WHITENED_INJECTIONS.name].numpy()

        # Using the first values from each array
        whitened_injection = whitened_injections[0][0]
        injection = injections[0][0]
        
        to_plot = {
            "Whitened Onsouce + Injection": onsource_,
            "Whitened Injection": whitened_injection,
            "Injection": injection
        }
    else:
        to_plot = {
            "Whitened Onsouce": onsource_
        }
    
    # Extract two strain plots
    strain_plots = gf.generate_strain_plot(
        to_plot, 
        sample_rate_hertz, 
        onsource_duration_seconds, 
        height=400, 
        has_legend=False,
        scale_factor=scale_factor
    )

    # Extract correlation plot
    correlation_plot = gf.generate_correlation_plot(
        correlation_, 
        sample_rate_hertz, 
        height=400, 
        has_legend=False
    )
    
    info_panel =  gf.create_info_panel({
        "Injection Type": f"{injection_type}.",
        "Coherent": f"{coherent}."
    }, height = 400)
    
    # Append to the layout
    layout.append([info_panel, strain_plots, correlation_plot])
    
    return layout

def plot_pearson_correlation(
    output_diretory_path : Path = Path("./figures/")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    ifos = [gf.IFO.L1, gf.IFO.H1]
    
    # Define injection directory path:
    injection_directory_path : Path = \
        Path("./injection_parameters/")
    
    # Intilise Scaling Method:
    scaling_method = \
        gf.ScalingMethod(
            gf.Distribution(min_=50.0,max_=50.0,type_=gf.DistributionType.UNIFORM),
            gf.ScalingTypes.SNR
        )
    
    # Load injection config:
    phenom_d_generator : gf.cuPhenomDGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "baseline_phenom_d.json", 
            sample_rate_hertz, 
            onsource_duration_seconds,
            scaling_method=scaling_method,
            network=ifos
        )
    
    wnb_generator : WNBGenerator = \
        gf.WaveformGenerator.load(
            injection_directory_path / "baseline_wnb.json", 
            sample_rate_hertz, 
            onsource_duration_seconds,
            scaling_method=scaling_method,
            network=ifos
        )
    
    incoherent_wnb_generator = gf.IncoherentGenerator(
        [wnb_generator, wnb_generator]
    )
    
    incoherent_phenom_d_generator = gf.IncoherentGenerator(
        [phenom_d_generator, phenom_d_generator]
    )
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = \
        gf.IFODataObtainer(
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = \
        gf.NoiseObtainer(
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=ifos
        )
    
    dataset_params = {
        "seed": 1000,
        "sample_rate_hertz": sample_rate_hertz,
        "onsource_duration_seconds": onsource_duration_seconds,
        "offsource_duration_seconds": offsource_duration_seconds,
        "crop_duration_seconds": crop_duration_seconds,
        "noise_obtainer": noise_obtainer,
        "num_examples_per_batch": num_examples_per_batch,
        "input_variables": [
            gf.ReturnVariables.WHITENED_ONSOURCE, 
            gf.ReturnVariables.INJECTION_MASKS, 
            gf.ReturnVariables.INJECTIONS,
            gf.ReturnVariables.WHITENED_INJECTIONS,
            gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE
        ]
    }
    
    injection_generators = [
        None,
        wnb_generator, 
        incoherent_wnb_generator,
        phenom_d_generator,
        incoherent_phenom_d_generator
    ]
    
    injection_types = [
        "No Injection",
        "White Noise Burst",
        "White Noise Burst",
        "IMRPhenomD",
        "IMRPhenomD"
    ]
    
    coherent = [
        "False",
        "True",
        "False",
        "True",
        "False"
    ]
    
    layout = []
    for injection_generator, type_, coherent_ in zip(injection_generators, injection_types, coherent):
        params = deepcopy(dataset_params)
        params["injection_generators"] = injection_generator
        generator : tf.data.Dataset = gf.Dataset(**params)
        
        layout += generate_plots(
            generator, 
            sample_rate_hertz, 
            onsource_duration_seconds, 
            scale_factor,
            injection_generator != None,
            type_,
            coherent_
        )
        
    # Ensure output directory exists
    gf.ensure_directory_exists(output_diretory_path)
    
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
    gpus = gf.find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gf.setup_cuda(
        gpus, 
        max_memory_limit = memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test IFO noise dataset:
    with strategy.scope():
        plot_pearson_correlation()
