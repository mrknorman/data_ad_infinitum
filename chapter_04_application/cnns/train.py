import json
import logging
import argparse
import prctl
from copy import deepcopy
from pathlib import Path
from typing import List
import sys
import os

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from tensorflow.keras import losses, optimizers
import gravyflow as gf

def train_cnn(
        heart,
        # Model Arguments:
        model_config_path : Path,
        model_name : str, 
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

    if gf.is_redirected():
        cache_segments = False

    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    injection_directory_path : Path = current_dir / "../../injection_parameters"

    if model_path is None:
        model_path = current_dir / f"./models/cnn_{model_name}"
    
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
        
    input_configs = [
        {
            "name" : gf.ReturnVariables.ONSOURCE.name,
            "shape" : (num_onsource_samples,)
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

    # Load or build model:
    model = gf.Model.load(
        name=model_name,
        model_load_path=model_path,
        num_ifos=len(ifos),
        optimizer=optimizers.Adam(learning_rate=learning_rate), 
        loss=losses.BinaryCrossentropy(),
        input_configs=input_configs,
        output_config=output_config,
        model_config_path=model_config_path,
        force_overwrite=(restart_count==0)
    )
    
    if (restart_count==0):
        model.summary()
    else:
        print(f"Attempt {restart_count + 1}: Restarting training from where we left off...")
    
    model.train(
        train_dataset,
        test_dataset,
        training_config,
        force_retrain=(restart_count==0), 
        heart=heart
    )

    # Validation configs:
    efficiency_config = {
            "max_scaling" : 15.0, 
            "num_scaling_steps" : 3, 
            "num_examples_per_scaling_step" : 16384
        }
    far_config = {
            "num_examples" : 1.0E6
        }
    roc_config : dict = {
            "num_examples" : 1.0E6,
            "scaling_ranges" : [
                (8.0, 20.0),
                6.0,
                8.0,
                10.0,
                12.0
            ]
        } 
    
    model.validate(
        dataset_arguments,
        efficiency_config=efficiency_config,
        far_config=far_config,
        roc_config=roc_config,
        model_path=model_path,
        heart=heart
    )

    if heart is not None:
        heart.complete()

    return 0

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
        "model_config_path", 
        type = str, 
        help = (
            "Path to model config file. Required."
        )
    )
    parser.add_argument(
        "--name",
        type = str, 
        default = "Aaggghghghghh!",
        help = (
            "Name of cnn model."
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
        default = 1,
        help = (
            "Number of times model has been trained,"
            " if 0, model will be overwritten."
        )
    )

    args = parser.parse_args()

    # Set parameters based on command line arguments:
    model_config_path = Path(args.model_config_path)
    gpu = args.gpu
    memory_to_allocate_tf = args.request_memory
    restart_count = args.restart_count
    name = args.name

    # Set process name:
    prctl.set_name(f"gwflow_training_{name}")

    gf.Defaults.set(
        seed=1000,
        num_examples_per_generation_batch=256,
        num_examples_per_batch=32,
        sample_rate_hertz=8196.0,
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

        if train_cnn(
            heart,
            model_config_path=model_config_path,
            restart_count=restart_count,
            model_name=name
        ) == 0:
            logging.info("Training completed, do a shot!")
            os._exit(0)
        
        else:
            logging.error("Training failed for some reason.")
            os._exit(1)

        # Stop profiling
        #tf.profiler.experimental.stop()
    
    os._exit(1)
