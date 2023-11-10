import time
import subprocess
import logging
from datetime import datetime
import sys
import numpy as np
import signal
import os

import gravyflow as gf

# Configure logging
def configure_logging():
    log_filename = datetime.now().strftime('process_error_log_%Y%m%d.log')
    logging.basicConfig(filename=log_filename, level=logging.ERROR,
                        format='%(asctime)s:%(levelname)s:%(message)s')

# Modify the start_process function to track first-time start
def start_process(command, gpu_id, memory_to_request):
    try:
        time.sleep(1)  # Space out so not too many are run at the same time

        # Determine the mode for log files based on whether it's a first-time start
        mode = "w" if command.restart_count == 0 else "a"

        with open(f"perceptron_logs/{command.name}_log.txt", mode) as stdout_file, \
             open(f"perceptron_logs/{command.name}_error.txt", mode) as stderr_file:
            full_command = f"{command.full} --gpu {gpu_id} --request_memory {memory_to_request}"
            process = subprocess.Popen(full_command, shell=True, stdout=stdout_file, stderr=stderr_file)

            command.id = process.pid
            print(f"\n Process: {command.name} started at {command.id}")

            return process
    except Exception as e:
        logging.exception(f"\n Failed to start process {command.name} on GPU {gpu_id}.")
        command.restart_count += 1
        return None

# Check if a process needs to be marked as failed due to excessive restarts
def check_and_mark_failed(command, retcode, stdout, stderr, max_restarts, restart_time_window, commands_to_run):
    logging.error(f"\n Process {command.name} at {command.id} failed with return code {retcode}.")

    # Check if the process has been restarted more than N times in X seconds
    command.restart_count += 1
    if command.restart_count > max_restarts:
        logging.error(f"\n Process {command.name} has been restarted {command.restart_count} times within {restart_time_window} seconds. Marking as failed.")
        command.has_failed = True  # Mark process as failed
        return True

    if stdout:
        logging.error(f"\n Process {command.name} at {command.id} - STDOUT: {stdout.decode()}")
    if stderr:
        logging.error(f"\n Process {command.name} at {command.id} - STDERR: {stderr.decode()}")
    commands_to_run.append(command)

    return False

# Clean up restart counts older than X seconds
def clean_restart_counts(commands, restart_time_window):
    current_time = time.time()
    for command in commands:
        if current_time - command.restart_counter > restart_time_window:
            command.restart_counter = current_time
            command.restart_count = 0

# Handle termination signals and shut down gracefully
def signal_handler(signum, frame):
    logging.info("\n Received termination signal. Shutting down gracefully...")
    sys.exit(0)

# Main function
def main(commands_to_run, wait_timer_seconds=10, tensorflow_memory_per_process_mb=2000, cuda_overhead_per_process_mb=1000, max_restarts=4, restart_time_window=1200):
    total_memory_per_process = tensorflow_memory_per_process_mb + cuda_overhead_per_process_mb

    configure_logging()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting the process management system...")
    print(f"Monitoring {len(commands_to_run)} processes across available GPUs.")

    try:
        free_memory = gf.get_memory_array()
        logging.info(f"Initial GPU Memory Status: {free_memory}")
    except Exception as e:
        logging.exception("Failed to get initial free memory array.")
        free_memory = []
        logging.error("Failed to get initial free memory array. See logs for details.")

    running_processes = []
    num_restarted = 0
    num_failed = 0
    num_completed = 0
    total = len(commands_to_run)

    spinner = "|/-\\"
    idx = 0

    while commands_to_run or running_processes:
        idx = (idx + 1) % len(spinner)
        sys.stdout.write(f"\r{spinner[idx]} {len(running_processes)}/{total} running, {num_completed}/{total} completed, {num_failed}/{total} failed, {len(commands_to_run)}/{total} in queue. {num_restarted} restarts.")
        sys.stdout.flush()

        try:
            free_memory = gf.get_memory_array()
        except Exception as e:
            logging.exception("\n Failed to update free memory array.")
            free_memory = []

        for proc, command in running_processes[:]:
            if proc is not None:
                retcode = proc.poll()
                if retcode is not None:  # Process finished.
                    running_processes.remove((proc, command))
                    stdout, stderr = proc.communicate()

                    # Check if the process should be marked as failed
                    if retcode != 0:  # Process failed, log the error
                        if check_and_mark_failed(command, retcode, stdout, stderr, max_restarts, restart_time_window, commands_to_run):
                            num_failed += 1
                            continue  # Skip re-queueing the process
                        else:
                            num_restarted += 1
                    else:
                        print(f"\n Process {command.name} at {command.id} completed sucessfully with return code {retcode}.")
                        num_completed += 1
        
        num_processes = len(commands_to_run)
        assignment_array = np.full(num_processes, -1)

        process_index = 0
        for gpu_index, gpu_memory in enumerate(free_memory):
            while gpu_memory >= total_memory_per_process and process_index < num_processes:
                gpu_memory -= total_memory_per_process
                assignment_array[process_index] = gpu_index
                process_index += 1

        for i, gpu in enumerate(assignment_array):
            if gpu > -1 and commands_to_run:
                command = commands_to_run.pop(0)

                process = start_process(command, gpu, tensorflow_memory_per_process_mb)
                if process is not None:
                    running_processes.append((process, command))
                elif not command.has_failed:
                    print(f"\n Attempting restart of {command.name}.")
                    commands_to_run.append(command)  # Re-queue if start_process failed
                    num_restarted += 1
                    time.sleep(wait_timer_seconds)

        clean_restart_counts(commands_to_run, restart_time_window)
    
    print(f"All processes finished. {num_completed}/{total}  completed, {num_failed}/{total}  failed.  {num_restarted} attempted restarts.")

class processCommand:
    def __init__(self, command_string: str):
        self.id = -1
        self.restart_count = 0
        self.restart_counter = time.time()
        self.full = command_string
        self.has_failed = False

        parts = command_string.split()
    
        # Check if the command starts with "python"
        if parts and parts[0] == "python":
            parts.pop(0)  # Remove 'python' from the command
        else:
            raise ValueError("Command does not start with 'python'.")

        # Extract the script path and name
        self.path = parts.pop(0)
        self.name = self.path.split("/")[-1].replace(".py", "")

        # Parse arguments
        self.args = {}
        current_key = None
        for part in parts:
            if part.startswith("--"):
                current_key = part[2:]
                self.args[current_key] = []
            elif current_key is not None:
                self.args[current_key].append(part)

        # Join the argument values if they were split due to spaces
        for key, value in self.args.items():
            self.args[key] = " ".join(value)
            self.name += f"_{key}_{'_'.join(value)}"

if __name__ == "__main__":
    commands_to_run = [
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 64"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 64 64"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128 64"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128 128"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 64"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 128"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 256"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 64 64 64"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128 128 128"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 256 256"),
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py --layers 512")
    ]

    main(commands_to_run)


