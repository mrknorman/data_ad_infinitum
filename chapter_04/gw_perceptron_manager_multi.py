from typing import List
from pathlib import Path
import copy
import os
import sys

import time
import subprocess
import logging
from datetime import datetime
import threading
import numpy as np

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
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
        max_num_concurent_processes : int = 1
    ):

    configure_logging()

    manager = gf.Manager(
        initial_processes,
        max_restarts=max_restarts,
        restart_timeout_seconds=restart_timeout_seconds, 
        process_start_wait_seconds=process_start_wait_seconds, 
        management_tick_length_seconds=management_tick_length_seconds,
        max_num_concurent_processes=max_num_concurent_processes,
        log_directory_path = Path("./models/chapter_04_perceptrons_multi/logs")
    )

    while manager:
        manager()

        manager.tabulate()

if __name__ == "__main__":

    tensorflow_memory_mb = 8000
    cuda_overhead_mb = 2000
        
    commands_to_run = [
        #gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py", "1", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 64", "64", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 128", "128", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 256", "256", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 64 64", "64_64",tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 128 64", "64_64",tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 128 128", "128_128", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 256 64", "256_64",tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 256 128", "256_128", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 256 256", "256_256", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 64 64 64", "64_64_64", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 128 128 128", "128_128_128", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 256 256 256", "256_256_256", tensorflow_memory_mb, cuda_overhead_mb),
        gf.Process("python ./chapter_04/chapter_04_gw_perceptron_multi.py --layers 512", "512", tensorflow_memory_mb, cuda_overhead_mb)
    ]
    
    main(commands_to_run)


