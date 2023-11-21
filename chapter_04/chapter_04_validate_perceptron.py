# Built-In imports:
import logging
from pathlib import Path
import os
from typing import List, Union, Dict
import prctl
import logging
import sys
import argparse

# Library imports:
import numpy as np
import tensorflow as tf
from bokeh.io import output_file, save
from bokeh.layouts import gridplot

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Local imports:
import gravyflow as gf

def validate_perceptrons(
    model_name : str, 
    model_path : str,
    output_directory_path : Path = Path("./validation_results/"),
    noise_directory_path : Path = Path("./validation_noise_files/"),
    minimum_snr : float = 8.0,
    maximum_snr : float = 15.0,
    ifos : List[gf.IFO] = [gf.IFO.L1],
    heart : gf.Heart = None
    ):
    
    efficiency_config = {
            "max_scaling" : 15.0, 
            "num_scaling_steps" : 31, 
            "num_examples_per_scaling_step" : 16384 // 2
        }
    far_config = {
            "num_examples" : 1.0E5
        }
    roc_config : dict = {
            "num_examples" : 1.0E5,
            "scaling_ranges" :  [
                #(8.0, 20.0),
                #6.0,
                8.0,
                #10.0,
                #12.0
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

     # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = current_dir / "../injection_parameters"

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
        cache_segments=False,
        logging_level=logging.INFO
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
            roc_config=roc_config,
            heart=heart
        )

    # Save validation data:
    validator.save(
        output_directory_path / f"{model_name}/validation_data.h5", 
    )

    validator.plot(
        output_directory_path / f"{model_name}/validation_plots.html", 
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
    
    # Read command line arguments:
    parser = argparse.ArgumentParser(
        description = (
            "Train a multi-layer perceptron "
            "for gravitational-wave detection."
        )
    )
    parser.add_argument(
        "--layers", 
        type = int, 
        nargs = "*", 
        default = [],
        help = (
            "A list of integers representing the number of "
            "neurons in each hidden layer."
        )
    )
    parser.add_argument(
        "--gpu",
        type = int, 
        default = None,
        help = (
            "Specify a gpu to use."
        )
    )

    parser.add_argument(
        "--request_memory",
        type = int, 
        default = 4000,
        help = (
            "Specify a how much memory to give tf."
        )
    )

    parser.add_argument(
        "--restart_count",
        type = int, 
        default = 0,
        help = (
            "Number of times model has been trained,"
            " if 0, model will be overwritten."
        )
    )

    parser.add_argument(
        "--name",
        type = str, 
        default = None,
        help = (
            "Name of perceptron."
        )
    )

    args = parser.parse_args()

    # Set parameters based on command line arguments:
    num_neurons_in_hidden_layers = args.layers
    gpu = args.gpu
    memory_to_allocate_tf = args.request_memory
    restart_count = args.restart_count
    name = args.name

    # Set process name:
    prctl.set_name(f"gwflow_training_{num_neurons_in_hidden_layers}")

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

    # Set up TensorBoard logging directory
    logs = "logs"

    if gf.is_redirected():
        heart = gf.Heart(name)
    else:
        heart = None
        
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

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    string_id = "_".join(map(str, num_neurons_in_hidden_layers))
    model_path = current_dir / f"../models/chapter_04_perceptrons_single/perceptron_{string_id}"

    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
            gpus=gpu
        ):            
        validate_perceptrons(
            model_path,
            model_path,
            heart=heart
        )
        
