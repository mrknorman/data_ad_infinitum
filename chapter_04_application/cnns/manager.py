from typing import List
from pathlib import Path
import copy
import sys
import os

import time
import subprocess
import logging
from datetime import datetime
import threading
import numpy as np

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import gravyflow as gf

# Configure logging
def configure_logging():
    log_filename = datetime.now().strftime('process_error_log_%Y%m%d.log')
    logging.basicConfig(filename=log_filename, level=logging.ERROR,
                        format='%(asctime)s:%(levelname)s:%(message)s')

# Main function
def main(
        initial_processes,
        max_restarts : int = 20, 
        restart_timeout_seconds : float = 1200.0, 
        process_start_wait_seconds : float = 1.0,
        management_tick_length_seconds : float = 5.0,
        max_num_concurent_processes : int = 7
    ):

    configure_logging()

    manager = gf.Manager(
        initial_processes,
        max_restarts=max_restarts,
        restart_timeout_seconds=restart_timeout_seconds, 
        process_start_wait_seconds=process_start_wait_seconds, 
        management_tick_length_seconds=management_tick_length_seconds,
        max_num_concurent_processes=max_num_concurent_processes,
        log_directory_path = Path(f"{current_dir}/models/logs/")
    )

    while manager:
        manager()

        manager.tabulate()

if __name__ == "__main__":

    tensorflow_memory_mb = 8000
    
    cuda_overhead_mb = 4000

    training_script_path : Path = Path(f"{current_dir}/train.py")

    model_names = [
        "gabbard",
        "george_small",
        "george_large"
    ]
    
    config_paths = [
        f"{current_dir}/../../model_parameters/gabbard.json",
        f"{current_dir}/../../model_parameters/george_small.json",
        f"{current_dir}/../../model_parameters/george_large.json",
    ]
    
    commands_to_run = [
        gf.Process(f"python {training_script_path} {config}", name, tensorflow_memory_mb, cuda_overhead_mb, initial_restart_count=1)
        for name, config in zip(model_names, config_paths)
    ]
    
    main(commands_to_run)

