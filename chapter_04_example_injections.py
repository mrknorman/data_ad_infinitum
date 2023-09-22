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

# Local imports:
from py_ml_tools.maths import Distribution, DistributionType, set_random_seeds
from py_ml_tools.setup import find_available_GPUs, setup_cuda, ensure_directory_exists
from py_ml_tools.injection import (cuPhenomDGenerator, WNBGenerator, InjectionGenerator, 
                         WaveformParameters, WaveformGenerator)
from py_ml_tools.plotting import generate_strain_plot, create_info_panel, generate_spectrogram

def plot_injection_examples(
    num_examples : int = 4,
    output_diretory_path : Path = Path("./figures")
    ):
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = num_examples
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 1.0
    crop_duration_seconds : float = 0.0
    scale_factor : float = 1.0E21
    
    set_random_seeds(101)
    
     # Define injection directory path:
    injection_directory_path : Path = \
        Path("./injection_parameters")
    
    phenom_d_generator : cuPhenomDGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "baseline_phenom_d.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    phenom_d_generator.front_padding_duration_seconds = 0.9
    
    wnb_generator : WNBGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "baseline_wnb.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    wnb_generator.front_padding_duration_seconds = 0.9
    
    return_parameters = [
        # CBC parameters:
        WaveformParameters.MASS_1_MSUN,
        WaveformParameters.MASS_2_MSUN,
        WaveformParameters.SPIN_1_IN,
        WaveformParameters.SPIN_2_IN,
    
        # WNB paramters:
        WaveformParameters.DURATION_SECONDS, 
        WaveformParameters.MIN_FREQUENCY_HERTZ,
        WaveformParameters.MAX_FREQUENCY_HERTZ,
    ]
    
    injection_generator : InjectionGenerator = \
        InjectionGenerator(
            [phenom_d_generator, wnb_generator],
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return = return_parameters
        )        
    
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
        
    generator : Iterator = injection_generator.generate
    
    injections, mask, parameters = next(generator())
        
    phenom_d = [
        generate_strain_plot(
            {"PhenomD Waveform": injection},
            sample_rate_hertz,
            total_onsource_duration_seconds,
            colors = Bright[7],
            scale_factor=scale_factor
        )
        for injection in injections[0]
    ]
    
    wnb = [
        generate_strain_plot(
            {"White Noise Burst": injection},
            sample_rate_hertz,
            total_onsource_duration_seconds,
            colors = Bright[7][1:]
        )
        for injection in injections[1]
    ]
    
    def print_spins(spin):
        return  f"x: {spin[0]:.2f}, y: {spin[1]:.2f}, z: {spin[2]:.2f}."
    
    # For PhenomD injections
    phenom_d_info = [
        create_info_panel({
            "Companion 1 Mass": f"{mass_1:.2f} msun.",
            "Companion 2 Mass": f"{mass_2:.2f} msun.",
            "Companion 1 Spin": f"{print_spins(spin_1)}",
            "Companion 2 Spin": f"{print_spins(spin_2)}",
        })
        for mass_1, mass_2, spin_1, spin_2 in zip(
            parameters[WaveformParameters.MASS_1_MSUN][0].numpy(),
            parameters[WaveformParameters.MASS_2_MSUN][0].numpy(),
            parameters[WaveformParameters.SPIN_1_IN][0].numpy().reshape(-1, 3),
            parameters[WaveformParameters.SPIN_2_IN][0].numpy().reshape(-1, 3)
        )
    ]

    # For WNB injections
    wnb_info = [
        create_info_panel({
            "Duration": f"{duration:.2f} seconds.",
            "Minimum Frequency": f"{min_freq:.2f} hertz.",
            "Maximum Frequency": f"{max_freq:.2f} hertz."
        })
        for duration, min_freq, max_freq in zip(
            parameters[WaveformParameters.DURATION_SECONDS][1].numpy(),
            parameters[WaveformParameters.MIN_FREQUENCY_HERTZ][1].numpy(),
            parameters[WaveformParameters.MAX_FREQUENCY_HERTZ][1].numpy(),
        )
    ]
        
    layout = [list(item) for item in zip(phenom_d_info, phenom_d, wnb, wnb_info)]
    
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "04_example_injections.html")

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
        plot_injection_examples()