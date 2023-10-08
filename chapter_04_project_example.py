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
from py_ml_tools.setup import (find_available_GPUs, setup_cuda, 
                               ensure_directory_exists)
from py_ml_tools.detector import Network, IFO
from py_ml_tools.injection import (cuPhenomDGenerator, WNBGenerator, 
                                   InjectionGenerator, WaveformParameters,
                                   WaveformGenerator)
from py_ml_tools.plotting import (generate_strain_plot, create_info_panel, 
                                  generate_spectrogram)

def plot_projection_examples(
    num_tests : int = 2048,
    output_diretory_path : Path = Path("./figures")
    ):
    
    set_random_seeds(101)
    
    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = 32
    sample_rate_hertz : float = 2050.0
    onsource_duration_seconds : float = 0.3
    crop_duration_seconds : float = 0.05
    scale_factor : float = 1.0E21
        
    # Define injection directory path:
    injection_directory_path : Path = \
        Path("./injection_parameters")
    
    phenom_d_generator : cuPhenomDGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "baseline_phenom_d.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    phenom_d_generator.injection_chance = 1.0
    phenom_d_generator.front_padding_duration_seconds = 0.24
    phenom_d_generator.back_padding_duration_seconds = 0.05
    
    wnb_generator : WNBGenerator = \
        WaveformGenerator.load(
            injection_directory_path / "baseline_wnb.json", 
            sample_rate_hertz, 
            onsource_duration_seconds
        )
    
    wnb_generator.injection_chance = 1.0
    wnb_generator.front_padding_duration_seconds = 0.10
    wnb_generator.back_padding_duration_seconds = 0.15
    wnb_generator.duration_seconds = Distribution(
        value=0.1, type_=DistributionType.CONSTANT)
    
    injection_generator : InjectionGenerator = \
        InjectionGenerator(
            [phenom_d_generator, wnb_generator],
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return = \
                [WaveformParameters.MASS_1_MSUN, WaveformParameters.MASS_2_MSUN]
        )
    
    total_onsource_duration_seconds : float = \
        onsource_duration_seconds + (crop_duration_seconds * 2.0)
        
    generator : Iterator = injection_generator.generate
    
    injections, mask, parameters = next(generator())
    
    network = Network([IFO.L1, IFO.H1, IFO.V1])
    
    phenom_d_projected_injections = \
        network.project_wave(injections[0], sample_rate_hertz)
    wnb_projected_injections = \
        network.project_wave(injections[1], sample_rate_hertz)
    
    duration_seconds = 0.2
    start_crop_num_samples : int = int(crop_duration_seconds*sample_rate_hertz)
    end_crop_num_samples : int = int(
        (crop_duration_seconds + duration_seconds)*sample_rate_hertz
    )
            
    phenom_d_injection = phenom_d_projected_injections.numpy()[0][:, start_crop_num_samples:end_crop_num_samples]
    wnb_injection = wnb_projected_injections.numpy()[0][:, start_crop_num_samples:end_crop_num_samples]
    
    layout = [
        [generate_strain_plot(
            {"Injection Test": phenom_d_projection},
            sample_rate_hertz,
            duration_seconds,
            scale_factor=scale_factor,
            colors = Bright[7][index:],
            has_legend = False
        ), 
        generate_strain_plot(
            {"Injection Test": wnb_projection},
            sample_rate_hertz,
            duration_seconds,
            scale_factor=1.0,
            colors = Bright[7][index:],
            has_legend = False
        )
        ]
        for index, (phenom_d_projection, wnb_projection) in enumerate(zip(phenom_d_injection, wnb_injection))
    ]
    
    # Update the y-range of each plot in this column
    for plots_column in layout:
        plots_column[0].y_range.end = 0.04
        plots_column[0].y_range.start = -0.04
        plots_column[1].y_range.end = 0.06
        plots_column[1].y_range.start = -0.06

    
    # Ensure output directory exists
    ensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "04_projection_plots.html")

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
        plot_projection_examples()