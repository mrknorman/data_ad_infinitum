import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict
from copy import deepcopy
import sys

import tensorflow as tf
from tensorflow.keras import losses, optimizers
import prctl


# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import gravyflow as gf

def create_perceptron_plan(
        num_neurons_in_hidden_layers : List[int]
    ):
    
    # Calculate derived arguments:
    hidden_layers = []
    for num_neurons in num_neurons_in_hidden_layers:
        hidden_layers.append(gf.DenseLayer(num_neurons))
        
    return hidden_layers

def load_or_build_model(builder, model_filename, input_configs, output_config, force_overwrite = False):
    # Check if the model file exists
    if os.path.exists(model_filename) and not force_overwrite:
        try:
            # Try to load the model
            print(f"Loading model from {model_filename}")
            builder.model = tf.keras.models.load_model(model_filename)
            builder.model_path = model_filename

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Building new model...")
            builder.build_model(input_configs, output_config, model_path=model_filename)
    else:
        # If the model doesn't exist, build a new one
        print("No saved model found. Building new model...")
        builder.build_model(input_configs, output_config, model_path=model_filename)

def train_perceptron(
        heartbeat_object,
        # Model Arguments:
        num_neurons_in_hidden_layers : List[int],
        cache_segments : bool = True,
        # Training Arguments:
        patience : int = 10,
        learning_rate : float = 1.0E-4,
        max_epochs : int = 1000,
        model_path : Path = None,
        # Dataset Arguments: 
        num_train_examples : int = int(1E5),
        num_validation_examples : int = int(1E4),
        minimum_snr : float = 8.0,
        maximum_snr : float = 15.0,
        ifos : List[gf.IFO] = [gf.IFO.L1],
        # Manage args
        restart_count : int = 0
    ):

    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = current_dir / "../../injection_parameters"

    string_id = "_".join(map(str, num_neurons_in_hidden_layers))
    if model_path is None:
        model_path = current_dir / f"./models/perceptron_{string_id}"
    
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
    phenom_d_generator.injection_chance = 0.5

    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        gf.ObservingRun.O3, 
        gf.DataQuality.BEST, 
        [
            gf.DataLabel.NOISE,
            gf.DataLabel.GLITCHES
        ],
        gf.SegmentOrder.RANDOM,
        cache_segments=cache_segments,
        force_acquisition=False,
        logging_level=logging.ERROR
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

    load_or_build_model(
        builder, 
        model_path, 
        input_configs, 
        output_config,
        force_overwrite=(restart_count==0)
    )
    
    if (restart_count==0):
        builder.summary()
    else:
        print(f"Attempt {restart_count + 1}: Restarting training from where we left off...")
    
    builder.train_model(
        train_dataset,
        test_dataset,
        training_config,
        force_retrain=(restart_count==0), 
        heart=heartbeat_object
    )

    if heartbeat_object is not None:
        heartbeat_object.beat()

    gf.save_dict_to_hdf5(
        builder.metrics[0].history, 
        model_path / "metrics", 
        force_overwrite=False
    )    
    
    if heartbeat_object is not None:
        heartbeat_object.complete()

    return 0

if __name__ == "__main__":

    #signal.signal(signal.SIGINT, gf.signal_handler)  # Handle Ctrl+C
    #signal.signal(signal.SIGTERM, gf.signal_handler)  # Handle termination signal
    
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
    
    with gf.env(
            memory_to_allocate_tf=memory_to_allocate_tf,
            gpus=gpu
        ):
           
        # Start profiling
        #tf.profiler.experimental.start(logs)

        if train_perceptron(
            heart,
            num_neurons_in_hidden_layers=num_neurons_in_hidden_layers,
            restart_count=restart_count
        ) == 0:
            logging.info("Training completed, do a shot!")
            os._exit(0)
        
        else:
            logging.error("Training failed for some reason.")
            os._exit(1)

        # Stop profiling
        #tf.profiler.experimental.stop()
    
    os._exit(1)