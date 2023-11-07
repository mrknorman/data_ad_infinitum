# Built-In imports:
import logging
from pathlib import Path
import os
from typing import List, Union, Dict

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Local imports:
import gravyflow as gf

def validate_perceptrons(
    model_name : str, 
    model_path : str,
    output_directory_path : Path = Path("./validation_results/"),
    noise_directory_path : Path = Path("./validation_datasets/"),
    minimum_snr : float = 8.0,
    maximum_snr : float = 15.0,
    ifos : List[gf.IFO] = [gf.IFO.L1]
    ):

    efficiency_config : Dict[str, Union[float, int]] = {
            "max_scaling" : 15.0, 
            "num_scaling_steps" : 31, 
            "num_examples_per_scaling_step" : 2048
        }
    far_config : Dict[str, float] = {
            "num_examples" : 1.0E4
        }
    roc_config : Dict[str, Union[float, List]] = {
            "num_examples" : 1.0E4,
            "scaling_ranges" :  [
                (8.0, 20.0),
                6.0
            ]
        }
    
    # Intilise Scaling Method:
    scaling_method : gf.ScalingMethod = gf.ScalingMethod(
        gf.Distribution(
            min_= minimum_snr,
            max_= maximum_snr,
            type_=gf.DistributionType.UNIFORM
        ),
        gf.ScalingTypes.SNR
    )

    # Load injection config:
    phenom_d_generator : gf.cuPhenomDGenerator = gf.WaveformGenerator.load(
        injection_directory_path / "baseline_phenom_d.json", 
        scaling_method=scaling_method,    
        network = None # Single detector
    )

    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        [
            gf.DataLabel.NOISE,
            gf.DataLabel.GLITCHES
        ],
        gf.SegmentOrder.RANDOM,
        cache_segments=True,
        logging_level=logging.ERROR
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        data_directory_path=noise_directory_path,
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=ifos
    )

    # Set requested data to be used as model input:
    input_variables = [
        gf.ReturnVariables.ONSOURCE,
        gf.ReturnVariables.OFFSOURCE
    ]
    
    # Set requested data to be used as model output:
    output_variables = [
        gf.ReturnVariables.INJECTION_MASKS
    ]

    dataset_arguments : Dict = {
        # Noise: 
        "noise_obtainer" : noise_obtainer,
        # Injections:
        "injection_generators" : phenom_d_generator, 
        # Output configuration:
        "input_variables" : input_variables,
        "output_variables": output_variables
    }
    
    logging.info(f"Loading example model...")

    model = tf.keras.models.load_model(
        model_path / model_name
    )
    logging.info("Done.")

    # Validate model:
    validator = gf.Validator.validate(
            model=model, 
            name=model_name,
            dataset_args=dataset_arguments,
            efficiency_config=efficiency_config,
            far_config=far_config,
            roc_config=roc_config
        )

    # Save validation data:
    validator.save(
        output_directory_path / f"{model_name}_validation_data.h5", 
    )

    validator.plot(
        output_directory_path / f"{model_name}_validation_plots.html", 
    )

    return validator

def plot_validation(
    validators,
    output_directory_path : Path = Path("./validation_results/"),
    ):

    # Plot all model validation data comparison:            
    validators[0].plot(
        output_directory_path / "perceptron_cbc_validation_plots.html",
        comparison_validators = validators[1:]
    )

if __name__ == "__main__":
        
    # Set logging level:
    logging.basicConfig(level=logging.INFO)

    gf.Defaults.set(
        seed = 1000,
        num_examples_per_generation_batch=2048,
        num_examples_per_batch=32,
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        scale_factor=1.0E21
    )

    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = current_dir / "injection_parameters"

    model_path = current_dir / "models/chapter_04/"
    model_names = [
        model for model in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, model))
    ]

    model_names = model_names[:2]
    print(model_names)

    with gf.env():
        
        validators = []

        for model_name in model_names: 
            
            validators.append(
                validate_perceptrons(
                    model_name,
                    model_path
                )
            )
        
        plot_validation(
            validators,
            model_name,
            model_path
        )
