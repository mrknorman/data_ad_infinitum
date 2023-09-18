import argparse
from pathlib import Path

from tensorflow.keras import losses, metrics, optimizers

from py_ml_tools.dataset import group_split_dataset, O3
from py_ml_tools.model import DenseLayer, ModelBuilder
from py_ml_tools.setup import (find_available_GPUs, read_injection_config_file, 
                               setup_cuda)

if __name__ == "__main__":
    
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
    
    num_neurons_in_hidden_layer = args.layers
    
    # User parameters:
    num_train_examples = int(1.0E5)
    num_validate_examples = int(1.0E4)
    num_test_examples = int(1.0E4)
    
    num_examples_per_batch = 32
    sample_rate_hertz = 8192.0
    onsource_duration_seconds = 1.0
    max_segment_duration_seconds = 2048.0
    min_gpu_memory_mb = 10000
    num_gpus_requested = 1
    tensorflow_memory_mb = 8000
    input_name = "onsource"
    output_name = "injection_masks"
    data_directory_path = Path("./")
    injection_config_paths = [Path("./injection_parameters/phenom_d_parameters_standard.json")]
    
    training_config = \
    {
        "num_examples_per_epoc" : num_train_examples,
        "patience" : 10,
        "learning_rate" : 1e-4,
        "max_epochs" : 500,
        "model_path" : Path("./models")
    }
    
    # Calculate derived arguments:
    num_samples = int(onsource_duration_seconds*sample_rate_hertz)
    hidden_layers = []
    for num_neurons in num_neurons_in_hidden_layer:
        hidden_layers.append(DenseLayer(num_neurons))
    
    # Load injection configs:
    injection_configs = \
        [
            read_injection_config_file(
                path,
                sample_rate_hertz,
                onsource_duration_seconds
            ) for path in injection_config_paths
        ]
    
    # Setup dataset arguments:
    generator_args = {
        "time_interval" : O3,
        "data_labels" : ["noise", "glitches"],
        "ifo" : "L1",
        "injection_configs" : injection_configs,
        "sample_rate_hertz" : sample_rate_hertz,
        "onsource_duration_seconds" : onsource_duration_seconds,
        "max_segment_size" : max_segment_duration_seconds,
        "num_examples_per_batch" : num_examples_per_batch,
        "data_directory" : data_directory_path,
        "order" : "random",
        "seed" : 100,
        "apply_whitening" : True,
        "input_keys" : [input_name], 
        "output_keys" : [output_name],
        "save_segment_data" : True
    }
    
    #Setup CUDA:
    gpus = find_available_GPUs(min_gpu_memory_mb, num_gpus_requested)
    strategy = setup_cuda(gpus, tensorflow_memory_mb, verbose = True)
    
    def adjust_features(features, labels):
        labels['injection_masks'] = labels['injection_masks'][0]
        return features, labels
    
    # Setup dataset generators:
    train_dataset = \
        group_split_dataset(
            generator_args, "train", num_train_examples
        ).map(adjust_features)
    
    validate_dataset = \
        group_split_dataset(
            generator_args, "validate", num_validate_examples
        ).map(adjust_features)

    test_dataset = \
        group_split_dataset(
            generator_args, "test", num_test_examples
        ).map(adjust_features)

    # Initilise model
    builder = ModelBuilder(
        hidden_layers, 
        optimizer = \
            optimizers.Adam(learning_rate=training_config["learning_rate"]), 
        loss = losses.BinaryCrossentropy(), 
        batch_size = num_examples_per_batch
    )
        
    input_config = {
        "name" : input_name,
        "shape" : (num_samples,)
    }
    
    output_config = {
        "name" : output_name,
        "type" : "binary"
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