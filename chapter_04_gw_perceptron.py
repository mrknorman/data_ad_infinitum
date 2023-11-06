import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict
from copy import deepcopy


from tensorflow.keras import losses, optimizers

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
        patience : int = 10,
        learning_rate : float = 1.0E-4,
        max_epochs : int = 500,
        model_path : Path = None,
        # Dataset Arguments: 
        num_train_examples : int = int(1E5),
        num_validation_examples : int = int(1E3),
        minimum_snr : float = 8.0,
        maximum_snr : float = 15.0,
        ifos : List[gf.IFO] = [gf.IFO.L1]
    ):
    
    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = current_dir / "injection_parameters"

    if model_path is None:
        model_path = current_dir / f"models/chapter_04/perceptron_{num_neurons_in_hidden_layers}"
    
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
            #gf.DataLabel.GLITCHES
        ],
        gf.SegmentOrder.RANDOM,
        logging_level=logging.INFO
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
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

    def adjust_features(features, labels):
        labels['INJECTION_MASKS'] = labels['INJECTION_MASKS'][0]
        return features, labels
    
    train_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="train"
    ).map(adjust_features)
    
    validate_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="validate"
    ).map(adjust_features)
    
    test_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="test"
    ).map(adjust_features)
    
    num_onsource_samples = int(
        (gf.Defaults.onsource_duration_seconds + 2.0*gf.Defaults.crop_duration_seconds)*
        gf.Defaults.sample_rate_hertz
    )
    num_offsource_samples = int(
        gf.Defaults.offsource_duration_seconds*
        gf.Defaults.sample_rate_hertz
    )

    hidden_layers = [gf.WhitenLayer()]
    
    hidden_layers += create_perceptron_plan(
        num_neurons_in_hidden_layers
    )

    # Initilise model
    builder = gf.ModelBuilder(
        hidden_layers, 
        optimizer = \
            optimizers.Adam(learning_rate=learning_rate), 
        loss = losses.BinaryCrossentropy()
    )
        
    input_configs = [
        {
            "name" : gf.ReturnVariables.ONSOURCE.name,
            "shape" : (num_onsource_samples ,)
        },
        {
            "name" : gf.ReturnVariables.OFFSOURCE.name,
            "shape" : (num_offsource_samples,)
        }
    ]

    
    output_config = {
        "name" : gf.ReturnVariables.INJECTION_MASKS.name,
        "type" : "binary"
    }
    
    training_config = {
        "num_examples_per_epoc" : num_train_examples,
        "num_validation_examples" : num_validation_examples,
        "patience" : patience,
        "learning_rate" : learning_rate,
        "max_epochs" : max_epochs,
        "model_path" : model_path
    }
    
    builder.build_model(
        input_configs,
        output_config
    )
    
    builder.summary()
    builder.train_model(
        train_dataset,
        validate_dataset,
        training_config
    )

    print(builder.metrics[0].history)

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
    
    with gf.env(
            min_gpu_memory_mb=8000,
            memory_to_allocate_tf=4000
        ):
        
        train_perceptron(num_neurons_in_hidden_layers=num_neurons_in_hidden_layers)