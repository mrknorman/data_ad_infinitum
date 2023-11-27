
from pathlib import Path
import sys
import os

from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import gravyflow as gf

def plot_single_waveform(
    num_tests : int = 1,
    output_diretory_path : Path = Path("./figures/")
    ):
    
    with gf.env():
    
        # Test Parameters:
        num_examples_per_generation_batch : int = 2048
        num_examples_per_batch : int = num_tests
        sample_rate_hertz : float = 2048.0
        onsource_duration_seconds : float = 1.0
        offsource_duration_seconds : float = 16.0
        crop_duration_seconds : float = 0.5
        scale_factor : float = 1.0E21

        # Define injection directory path:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        injection_directory_path : Path = \
            Path(current_dir / "../injection_parameters")

        # Intilise Scaling Method:
        scaling_method = \
            gf.ScalingMethod(
                gf.Distribution(
                    min_=8.0,
                    max_=15.0,
                    type_=gf.DistributionType.UNIFORM
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
        phenom_d_generator.injection_chance = 1.0

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = \
            gf.IFODataObtainer(
                gf.ObservingRun.O3, 
                gf.DataQuality.BEST, 
                [
                    gf.DataLabel.NOISE
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
        
        dataset : tf.data.Dataset = gf.Dataset(
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
                gf.ReturnVariables.INJECTIONS,
                gf.ReturnVariables.WHITENED_INJECTIONS
            ],
        )

        input_dict, _ = next(iter(dataset))

        onsource = input_dict[gf.ReturnVariables.WHITENED_ONSOURCE.name].numpy()
        injections = input_dict[gf.ReturnVariables.INJECTIONS.name].numpy()
        whitened_injections = input_dict[
            gf.ReturnVariables.WHITENED_INJECTIONS.name
        ].numpy()


        layout = [
            [gf.generate_strain_plot(
                {
                    "Whitened Onsouce + Injection": onsource_,
                    "Whitened Injection" : whitened_injection,
                    "Injection": injection
                },
                sample_rate_hertz,
                scale_factor=scale_factor,
                width=2000,
                has_legend=False
            )
            for onsource_, whitened_injection, injection in zip(
                onsource,
                whitened_injections[0],
                injections[0]
            )]
        ]

        # Ensure output directory exists
        gf.ensure_directory_exists(output_diretory_path)

        # Define an output path for the dashboard
        output_file(output_diretory_path / "06_single_waveform.html")

        # Arrange the plots in a grid. 
        grid = gridplot(layout)

        save(grid)

if __name__ == "__main__":
    plot_single_waveform()
