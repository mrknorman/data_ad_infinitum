import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict
from copy import deepcopy


from tensorflow.keras import losses, metrics, optimizers

import gravyflow as gf

def create_perceptron_plan(
        num_neurons_in_hidden_layers : List[int]
    ):
    
    # Calculate derived arguments:
    hidden_layers = []
    for num_neurons in num_neurons_in_hidden_layers:
        hidden_layers.append(gf.DenseLayer(num_neurons))
        
    return hidden_layers

def train_perceptron(
        # Model Arguments:
        num_neurons_in_hidden_layers : List[int],
        # Training Arguments:
        num_examples_per_epoc : int = int(1E6),
        patience : int = 10,
        learning_rate : float = 1.0E-4,
        max_epochs : int = 500,
        model_path : Path = Path("./models"),
        # Dataset Arguments: 
        num_train_examples : int = int(1E6),
        num_examples_per_generation_batch : int = 2048,
        num_examples_per_batch : int = 32,
        sample_rate_hertz : float = 2048.0,
        onsource_duration_seconds : float = 1.0,
        offsource_duration_seconds : float = 16.0,
        crop_duration_seconds : float = 0.5,
        scale_factor : float = 1.0E21,
        minimum_snr : float = 8.0,
        maximum_snr : float = 15.0,
        ifos : List[gf.IFO] = [gf.IFO.L1]
    ):
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = \
        Path(current_dir / "injection_parameters")
        
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
        sample_rate_hertz, 
        onsource_duration_seconds,
        scaling_method=scaling_method,    
        network = None # Single detector
    )

    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        [
            gf.DataLabel.NOISE,
        ],
        gf.SegmentOrder.RANDOM
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=ifos
    )

    # Set requested data to be used as model input:
    input_variables = [
        gf.ReturnVariables.WHITENED_ONSOURCE
    ]
    
    # Set requested data to be used as model output:
    output_variables = [
        gf.ReturnVariables.INJECTION_MASKS
    ]

    dataset_arguments : Dict = {
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
        "input_variables" : input_variables,
        "output_variables": output_variables
    }
    
    train_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="train"
    )
    
    validate_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="test"
    )
    
    test_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="validate"
    )
    
    num_samples = int(onsource_duration_seconds*sample_rate_hertz)
    
    hidden_layers = create_perceptron_plan(
        num_neurons_in_hidden_layers
    )

    # Initilise model
    builder = gf.ModelBuilder(
        hidden_layers, 
        optimizer = \
            optimizers.Adam(learning_rate=learning_rate), 
        loss = losses.BinaryCrossentropy(), 
        batch_size = num_examples_per_batch
    )
        
    input_config = {
        "name" : gf.ReturnVariables.WHITENED_ONSOURCE.name,
        "shape" : (num_samples,)
    }
    
    output_config = {
        "name" : gf.ReturnVariables.INJECTION_MASKS.name,
        "type" : "binary"
    }
    
    training_config = {
        "num_examples_per_epoc" : num_train_examples,
        "patience" : patience,
        "learning_rate" : learning_rate,
        "max_epochs" : max_epochs,
        "model_path" : model_path
    }
    
    builder.build_model(
        input_config,
        output_config
    )
    
    builder.summary()
    builder.train_model(
        train_dataset,
        validate_dataset,
        training_config
    )

if __name__ == "__main__":

    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Read command line arguments:
    parser = argparse.ArgumentParser(
        description = \
            "Train a multi-layer perceptron for gravitational-wave detection."
    )
    parser.add_argument(
        "--layers", 
        type = int, 
        nargs = "*", 
        default = [],
        help = \
            """
            A list of integers representing the number of neurons in each hidden 
            layer.
            """
    )
    args = parser.parse_args()
    
    # Set parameters based on command line arguments:
    num_neurons_in_hidden_layers = args.layers
    
    with gf.env():
        train_perceptron(num_neurons_in_hidden_layers=num_neurons_in_hidden_layers)