# Built-In imports:
import logging
from pathlib import Path
from itertools import islice
from copy import deepcopy

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Bright
from tqdm import tqdm

# Local imports:
from py_ml_tools.maths import Distribution, DistributionType
from py_ml_tools.setup import (find_available_GPUs, setup_cuda, 
                               ensure_directory_exists)
from py_ml_tools.injection import (cuPhenomDGenerator, InjectionGenerator, 
                                   WaveformParameters, WaveformGenerator, 
                                   ScalingMethod, ScalingTypes)
from py_ml_tools.acquisition import (IFODataObtainer, SegmentOrder, ObservingRun, 
                                     DataQuality, DataLabel, IFO)
from py_ml_tools.noise import NoiseObtainer, NoiseType
from py_ml_tools.dataset import get_ifo_dataset, get_ifo_data, ReturnVariables
from py_ml_tools.plotting import (generate_strain_plot, create_info_panel, 
                                  generate_spectrogram)

def plot_scaling_comparison(
    num_examples : int = 4,
    output_diretory_path : Path = Path("./figures")
    ):
    
    # User Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_examples
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    # Define injection directory path:
    injection_directory_path : Path = \
        Path("./injection_parameters")
    
    # Intilise Scaling Methods:
    snr_scaling_method = \
        ScalingMethod(
            np.array([4.0,8.0,12.0,16.0]),
            ScalingTypes.SNR
        )
    
    phenom_d_generator : cuPhenomDGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "constant_phenom_d.json", 
            sample_rate_hertz, 
            onsource_duration_seconds,
            scaling_method=snr_scaling_method
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
            ifos=IFO.L1
        )
    
    input_variables = [
        ReturnVariables.WHITENED_ONSOURCE, 
        ReturnVariables.INJECTIONS,
        ReturnVariables.WHITENED_INJECTIONS,
        WaveformParameters.MASS_1_MSUN, 
        WaveformParameters.MASS_2_MSUN,
        WaveformParameters.SPIN_1_IN,
        WaveformParameters.SPIN_2_IN,
        ScalingTypes.SNR,
        ScalingTypes.HRSS,
        ScalingTypes.HPEAK
    ]
    
    dataset_args : Dict = {
        # Random Seed:
        "seed" : 1000,
        # Temporal components:
        "sample_rate_hertz" : sample_rate_hertz,   
        "onsource_duration_seconds" : onsource_duration_seconds,
        "offsource_duration_seconds" : offsource_duration_seconds,
        "crop_duration_seconds" : crop_duration_seconds,
        # Noise: 
        "noise_obtainer" : noise_obtainer,
        "scale_factor" : scale_factor,
        # Injections:
        "injection_generators" : phenom_d_generator, 
        # Output configuration:
        "num_examples_per_batch" : num_examples_per_batch,
        "input_variables" : input_variables
    }
    
    phenom_d_args : Dict = deepcopy(dataset_args)
    dataset : tf.data.Dataset = get_ifo_dataset(
        **phenom_d_args
    )
    
    phenom_d_data, _ = next(iter(dataset))
        
    phenom_d_onsource = phenom_d_data[ReturnVariables.WHITENED_ONSOURCE.name].numpy()
    phenom_d_injections = phenom_d_data[ReturnVariables.INJECTIONS.name].numpy()[0]
    phenom_d_whitened_injections = phenom_d_data[ReturnVariables.WHITENED_INJECTIONS.name].numpy()[0]
    
    phenom_d_mass_1_msun = phenom_d_data[WaveformParameters.MASS_1_MSUN.name].numpy()
    phenom_d_mass_2_msun = phenom_d_data[WaveformParameters.MASS_2_MSUN.name].numpy()
    
    phenom_d_snr = phenom_d_data[ScalingTypes.SNR.name].numpy()[0]
    phenom_d_hrss = phenom_d_data[ScalingTypes.HRSS.name].numpy()[0]
    phenom_d_hpeak = phenom_d_data[ScalingTypes.HPEAK.name].numpy()[0]
    
    wnb_args : Dict = deepcopy(dataset_args)
    
    # Intilise Scaling Methods:
    hrss_scaling_method = \
        ScalingMethod(
            np.arange(phenom_d_hrss[0], phenom_d_hrss[0]*5, phenom_d_hrss[0]),
            ScalingTypes.HRSS
        )
    
    wnb_generator : WNBGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "constant_wnb.json", 
            sample_rate_hertz, 
            onsource_duration_seconds,
            scaling_method=hrss_scaling_method

        )
    
    wnb_args["injection_generators"] = wnb_generator
    wnb_args["input_variables"] = [
        ReturnVariables.WHITENED_ONSOURCE, 
        ReturnVariables.INJECTIONS,
        ReturnVariables.WHITENED_INJECTIONS,
        WaveformParameters.DURATION_SECONDS, 
        WaveformParameters.MIN_FREQUENCY_HERTZ,
        WaveformParameters.MAX_FREQUENCY_HERTZ,
        ScalingTypes.SNR,
        ScalingTypes.HRSS,
        ScalingTypes.HPEAK
    ]
    
    dataset : tf.data.Dataset = get_ifo_dataset(
        **wnb_args
    )
    
    wnb_data, _ = next(iter(dataset))
        
    wnb_onsource = wnb_data[ReturnVariables.WHITENED_ONSOURCE.name].numpy()
    wnb_injections = wnb_data[ReturnVariables.INJECTIONS.name].numpy()[0]
    wnb_whitened_injections = wnb_data[ReturnVariables.WHITENED_INJECTIONS.name].numpy()[0]
    
    wnb_duration_seconds = wnb_data[WaveformParameters.DURATION_SECONDS.name].numpy()
    wnb_min_frequency_hertz = wnb_data[WaveformParameters.MIN_FREQUENCY_HERTZ.name].numpy()
    wnb_max_frequency_hertz = wnb_data[WaveformParameters.MAX_FREQUENCY_HERTZ.name].numpy()
    
    wnb_snr = wnb_data[ScalingTypes.SNR.name].numpy()[0]
    wnb_hrss = wnb_data[ScalingTypes.HRSS.name].numpy()[0]
    wnb_hpeak = wnb_data[ScalingTypes.HPEAK.name].numpy()[0]

    layout = [
        [
        create_info_panel({
            "Type": f"PhenomD Waveform",
            "HPEAK": f"{phenom_d_hpeak_/scale_factor:.2e}.",
            "HRSS": f"{phenom_d_hrss_/scale_factor:.2e}.",
            "Optimal SNR": f"{phenom_d_snr_:.2f}."
        }),  
        generate_strain_plot(
            {
                "Whitened Onsouce + Injection": phenom_d_onsource_,
                "Whitened Injection" : phenom_d_whitened_injection_,
                "Injection": phenom_d_injection_
            },
            sample_rate_hertz,
            onsource_duration_seconds,
            scale_factor=scale_factor,
            has_legend=False
        ), 
        generate_strain_plot(
            {
                "Whitened Onsouce + Injection": wnb_onsource_,
                "Whitened Injection" : wnb_whitened_injection_,
                "Injection": wnb_injection_
            },
            sample_rate_hertz,
            onsource_duration_seconds,
            scale_factor=scale_factor,
            has_legend=False
        ),
        create_info_panel({
            "Type": f"WNB",
            "HPEAK": f"{wnb_hpeak_/scale_factor:.2e}.",
            "HRSS": f"{wnb_hrss_/scale_factor:.2e}.",
            "Optimal SNR": f"{wnb_snr_:.2f}.",
        })
        ]
        for wnb_onsource_, phenom_d_onsource_, \
            wnb_whitened_injection_, phenom_d_whitened_injection_, \
            wnb_injection_, phenom_d_injection_, \
            phenom_d_snr_, phenom_d_hrss_, phenom_d_hpeak_, \
            wnb_snr_, wnb_hrss_, wnb_hpeak_ \
            in zip(
            wnb_onsource, phenom_d_onsource, \
            wnb_whitened_injections, phenom_d_whitened_injections, \
            wnb_injections, phenom_d_injections,
            phenom_d_snr, phenom_d_hrss, phenom_d_hpeak,
            wnb_snr, wnb_hrss, wnb_hpeak
        )
    ]
        
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "04_scaling_comparison.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
    save(grid)

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 10000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 4000
    
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
        plot_scaling_comparison()