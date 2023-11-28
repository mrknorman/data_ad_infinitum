import gravyflow as gf
from typing import List
import logging
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
from tqdm import tqdm

def pathfind(
        heartbeat_object,
        restart_count : int = 0,
        # Model Arguments:
        num_train_examples : int = int(2E7),
        num_test_examples : int = int(1E4),
        # Dataset Arguments: 
        ifos : List[gf.IFO] = [gf.IFO.L1],
    ):

    # Define injection directory path:
    current_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

    data_directory_path = current_dir / "noise_files"
    
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
        force_acquisition=False,
        logging_level=logging.INFO
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        data_directory_path=data_directory_path,
        noise_type=gf.NoiseType.REAL,
        ifos=ifos
    )

    # Set requested data to be used as model input:
    input_variables = [
        gf.ReturnVariables.ONSOURCE
    ]

    dataset_arguments : Dict = {
        # Noise: 
        "noise_obtainer" : noise_obtainer,
        # Output configuration:
        "input_variables" : input_variables
    }

    def adjust_features(features, labels):
        labels['INJECTION_MASKS'] = labels['INJECTION_MASKS'][0]
        return features, labels
    
    test_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="test"
    ).take(num_test_examples // gf.Defaults.num_examples_per_batch)

    for i in tqdm(test_dataset):
        pass

    train_dataset : tf.data.Dataset = gf.Dataset(
        **deepcopy(dataset_arguments),
        group="train"
    ).take(num_train_examples // gf.Defaults.num_examples_per_batch)

    for i in tqdm(train_dataset):
        pass

    return 0

if __name__ == "__main__":
     # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Read command line arguments:
    parser = argparse.ArgumentParser(
        description = (
            "Get ifo data... "
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
            "Name of pathfinder."
        )
    )

    args = parser.parse_args()

    # Set parameters based on command line arguments:
    gpu = args.gpu
    memory_to_allocate_tf = args.request_memory
    restart_count = args.restart_count
    name = args.name

    # Set process name:
    prctl.set_name(f"gwflow_data_pathfinder")

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

        if pathfind(
            heart,
            restart_count=restart_count
        ) == 0:
            logging.info("Found a path.")
            os._exit(0)
        
        else:
            logging.error("Pathfinding failed!")
            os._exit(1)

        # Stop profiling
        #tf.profiler.experimental.stop()
    
    os._exit(1)