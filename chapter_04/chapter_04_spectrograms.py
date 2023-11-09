from pathlib import Path
import logging

import gravyflow as gf

# Library imports:
import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.palettes import Bright
from tqdm import tqdm

def plot_spectrograms(
    output_diretory_path : Path = Path("./figures/")
    ):
    
    with gf.env():
    
        # Test Parameters:
        num_examples_per_generation_batch : int = 2048
        num_examples_per_batch : int = 2
        sample_rate_hertz : float = 2048.0
        onsource_duration_seconds : float = 1.0
        offsource_duration_seconds : float = 16.0
        crop_duration_seconds : float = 0.5
        scale_factor : float = 1.0E21
        height = 300

         # Define injection directory path:
        injection_directory_path : Path = \
            Path("./injection_parameters")

        # Intilise Scaling Method:
        scaling_method = \
            gf.ScalingMethod(
                gf.Distribution(
                    min_=15.0,
                    max_=30.0
                    ,type_=gf.DistributionType.UNIFORM
                ),
                gf.ScalingTypes.SNR
            )

        # Load injection config:
        phenom_d_generator : gf.cuPhenomDGenerator = \
            gf.WaveformGenerator.load(
                injection_directory_path / "baseline_phenom_d.json", 
                sample_rate_hertz, 
                onsource_duration_seconds,
                scaling_method=scaling_method
            )
        
        # Load WNB config:
        wnb_generator : gf.WNBGenerator = \
            gf.WaveformGenerator.load(
                injection_directory_path / "baseline_wnb.json", 
                sample_rate_hertz, 
                onsource_duration_seconds,
                scaling_method=scaling_method

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
                ifos=gf.IFO.L1
            )

        phenom_d_dataset : tf.data.Dataset = gf.Dataset(
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
            injection_generators=phenom_d_generator, 
            # Output configuration:
            num_examples_per_batch=num_examples_per_batch,
            input_variables = [
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS, 
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.ReturnVariables.SPECTROGRAM_ONSOURCE,
                gf.WaveformParameters.MASS_1_MSUN, 
                gf.WaveformParameters.MASS_2_MSUN,
            ],
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
                ifos=gf.IFO.L1
            )

        wnb_dataset : tf.data.Dataset = gf.Dataset(
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
            injection_generators=wnb_generator, 
            # Output configuration:
            num_examples_per_batch=num_examples_per_batch,
            input_variables = [
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.INJECTION_MASKS, 
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS,
                gf.ReturnVariables.SPECTROGRAM_ONSOURCE,
                gf.WaveformParameters.MASS_1_MSUN, 
                gf.WaveformParameters.MASS_2_MSUN,
            ],
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
                ifos=gf.IFO.L1
            )

        noise_dataset : tf.data.Dataset = gf.Dataset(
            # Random Seed:
            seed= 1000,
            # Temporal components:
            sample_rate_hertz=sample_rate_hertz,   
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
            # Noise: 
            noise_obtainer=noise_obtainer,
            # Output configuration:
            num_examples_per_batch=num_examples_per_batch,
            input_variables = [
                gf.ReturnVariables.WHITENED_ONSOURCE, 
                gf.ReturnVariables.SPECTROGRAM_ONSOURCE,
            ]
        )
        
        input_dict, _ = next(iter(noise_dataset))

        onsource = input_dict[gf.ReturnVariables.WHITENED_ONSOURCE.name].numpy()
        
        layout1 = [
            [gf.generate_strain_plot(
                {
                    "Whitened Onsouce + Injection": onsource_,
                },
                sample_rate_hertz,
                onsource_duration_seconds,
                scale_factor=scale_factor,
                height=height,
                has_legend=False
            ), 
            gf.generate_spectrogram(
                onsource_, 
                sample_rate_hertz,
                height=height
            )]
            for onsource_, in zip(
                onsource
            )
        ]
        
        layout = [list(row) for row in zip(*layout1)]

        input_dict, _ = next(iter(phenom_d_dataset))

        onsource = input_dict[gf.ReturnVariables.WHITENED_ONSOURCE.name].numpy()
        injections = input_dict[gf.ReturnVariables.INJECTIONS.name].numpy()
        whitened_injections = input_dict[
            gf.ReturnVariables.WHITENED_INJECTIONS.name
        ].numpy()
        masks = input_dict[gf.ReturnVariables.INJECTION_MASKS.name].numpy()
        mass_1_msun = input_dict[gf.WaveformParameters.MASS_1_MSUN.name].numpy()
        mass_2_msun = input_dict[gf.WaveformParameters.MASS_2_MSUN.name].numpy()

        layout2 = [
            [gf.generate_strain_plot(
                {
                    "Whitened Onsouce + Injection": onsource_,
                    "Whitened Injection" : whitened_injection,
                    "Injection": injection
                },
                sample_rate_hertz,
                onsource_duration_seconds,
                scale_factor=scale_factor,
                height=height,
                has_legend=False
            ), 
            gf.generate_spectrogram(
                onsource_, 
                sample_rate_hertz,
                height=height
            )]
            for onsource_, whitened_injection, injection, m1, m2 in zip(
                onsource,
                whitened_injections[0],
                injections[0], 
                mass_1_msun[0], 
                mass_2_msun[0]
            )
        ]
        
        layout += [list(row) for row in zip(*layout2)]
        
        input_dict, _ = next(iter(wnb_dataset))

        onsource = input_dict[gf.ReturnVariables.WHITENED_ONSOURCE.name].numpy()
        injections = input_dict[gf.ReturnVariables.INJECTIONS.name].numpy()
        whitened_injections = input_dict[
            gf.ReturnVariables.WHITENED_INJECTIONS.name
        ].numpy()
        masks = input_dict[gf.ReturnVariables.INJECTION_MASKS.name].numpy()

        layout3 = [
            [gf.generate_strain_plot(
                {
                    "Whitened Onsouce + Injection": onsource_,
                    "Whitened Injection" : whitened_injection,
                    "Injection": injection
                },
                sample_rate_hertz,
                onsource_duration_seconds,
                scale_factor=scale_factor,
                height=height,
                has_legend=False
            ), 
            gf.generate_spectrogram(
                onsource_, 
                sample_rate_hertz,
                height=height
            )]
            for onsource_, whitened_injection, injection in zip(
                onsource,
                whitened_injections[0],
                injections[0]
            )
        ]
        
        layout +=  [list(row) for row in zip(*layout3)]

        # Ensure output directory exists
        gf.ensure_directory_exists(output_diretory_path)

        # Define an output path for the dashboard
        output_file(output_diretory_path / "04_spectrogram_examples.html")
        
        # Arrange the plots in a grid. 
        grid = gridplot(layout)

        save(grid)
        
if __name__ == "__main__":
        
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    plot_spectrograms()
