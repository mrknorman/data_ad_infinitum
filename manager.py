import time
import subprocess
import logging
from datetime import datetime
import sys
import numpy as np
import signal
import os
import threading
from tqdm import tqdm

import gravyflow as gf

# Configure logging
def configure_logging():
    log_filename = datetime.now().strftime('process_error_log_%Y%m%d.log')
    logging.basicConfig(filename=log_filename, level=logging.ERROR,
                        format='%(asctime)s:%(levelname)s:%(message)s')

# Modify the start_process function to track first-time start
def start_process(command, memory_to_request):
    try:

        process_gap_seconds = 3
        print(f"\nWaiting {process_gap_seconds} s to space out process activations.")
        time.sleep(process_gap_seconds)  # Space out so not too many are run at the same time

        # Determine the mode for log files based on whether it's a first-time start
        mode = "w" if command.total_restart_count == 0 else "a"

        with open(f"perceptron_logs/{command.name}_log.txt", mode) as stdout_file, \
             open(f"perceptron_logs/{command.name}_error.txt", mode) as stderr_file:
            full_command = f"{command.full} --gpu {command.current_gpu} --request_memory {memory_to_request} --restart_count {command.total_restart_count} --name {command.name}"

            command.pipe_name = f"./tmp/heartbeat_{command.name}"
            gf.create_named_pipe(command.pipe_name)

            command.flags = {"has_died" : threading.Event(), "should_exit" : threading.Event()}
            command.pipe_monitor_thread = gf.start_monitoring_thread(command, command.flags)

            process = subprocess.Popen(
                full_command, 
                shell=True, 
                stdout=stdout_file, 
                stderr=stderr_file, 
            )

            command.id = process.pid
            logging.info(f"\nProcess: {command.name} started at {command.id}")

            return process
    except Exception as e:
        logging.exception(f"\nFailed to start process {command.name} on GPU {gpu_id}.")
        command.restart_count += 1
        command.total_restart_count += 1
        return None

# Check if a process needs to be marked as failed due to excessive restarts
def check_and_mark_failed(command, max_restarts, restart_time_window, commands_to_run):

    # Check if the process has been restarted more than N times in X seconds
    command.restart_count += 1
    command.total_restart_count += 1
    if command.restart_count > max_restarts:
        logging.error(f"\nProcess {command.name} has been restarted {command.restart_count} times within {restart_time_window} seconds. Marking as failed.")
        command.has_failed = True  # Mark process as failed
        gf.cleanup_named_pipe(command.pipe_name)
        return True
    
    commands_to_run.insert(0, command)

    return False

# Clean up restart counts older than X seconds
def clean_restart_counts(commands, restart_time_window):
    current_time = time.time()
    for command in commands:
        if current_time - command.restart_counter > restart_time_window:
            command.restart_counter = current_time
            command.restart_count = 0

    
running_processes = []

# Handle termination signals and shut down gracefully
def signal_handler(signum, frame):
    logging.info("\nReceived termination signal. Shutting down gracefully...")

    for proc, command in tqdm(running_processes):
        kill_command(command, proc, running_processes, None)

    sys.exit(0)

def kill_command(command, proc, running_processes, allocation_array):
    gf.kill_process(command.id)
    running_processes.remove((proc, command))

    if allocation_array is not None:
        allocation_array[command.current_gpu] -= command.memory_assigned
    
    command.current_gpu = -1
    command.memory_assigned = 0
    command.flags["should_exit"].set()
    gf.cleanup_named_pipe(command.pipe_name)

# Main function
def main(
        commands_to_run,
        wait_timer_seconds : int = 3, 
        tensorflow_memory_per_process_mb : int = 2000, 
        cuda_overhead_per_process_mb : int = 1000, 
        max_restarts : int = 4, 
        restart_time_window : int = 1200, 
        max_use : int = 95,
        max_num_processes : int = 7
    ):

    total_memory_per_process = tensorflow_memory_per_process_mb + cuda_overhead_per_process_mb

    configure_logging()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting the process management system...")
    print(f"Monitoring {len(commands_to_run)} processes across available GPUs.")

    try:
        free_memory = gf.get_memory_array()
        gpu_use = gf.get_gpu_utilization_array()
        logging.info(f"Initial GPU Memory Status: {free_memory}")
    except Exception as e:
        raise ValueError("Failed to get initial free memory array.")
        logging.error("Failed to get initial free memory array. See logs for details.")

    num_restarted = 0
    num_failed = 0
    num_completed = 0
    total = len(commands_to_run)
    allocation_array = np.full(len(free_memory), -1)

    spinner = "|/-\\"
    idx = 0

    while commands_to_run or running_processes:
        idx = (idx + 1) % len(spinner)
        if not gf.is_redirected():
            sys.stdout.write(f"\r{spinner[idx]} {len(running_processes)}/{total} running, {num_completed}/{total} completed, {num_failed}/{total} failed, {len(commands_to_run)}/{total} in queue. {num_restarted} restarts.")
            sys.stdout.flush()
        else:
            logging.info(f"{len(running_processes)}/{total} running, {num_completed}/{total} completed, {num_failed}/{total} failed, {len(commands_to_run)}/{total} in queue. {num_restarted} restarts.")

        try:
            free_memory = gf.get_memory_array()
            free_memory = free_memory + allocation_array
            gpu_use = gf.get_gpu_utilization_array()
        except Exception as e:
            logging.exception("\nFailed to update free memory array.")
        
        for proc, command in running_processes[:]:
            if proc is not None:
                retcode = proc.poll()
                if retcode is not None:  # Process finished.
                    kill_command(command, proc, running_processes, allocation_array)

                    stdout, stderr = proc.communicate()

                    # Check if the process should be marked as failed
                    if retcode != 0:  # Process failed, log the error

                        logging.error(f"\nProcess {command.name} at {command.id} failed with return code {retcode} : {gf.explain_exit_code(retcode)}.")

                        if stdout:
                            logging.error(f"\nProcess {command.name} at {command.id} - STDOUT: {stdout.decode()}")
                        if stderr:
                            logging.error(f"\nProcess {command.name} at {command.id} - STDERR: {stderr.decode()}")

                        if check_and_mark_failed(command, max_restarts, restart_time_window, commands_to_run):
                            num_failed += 1
                            continue  # Skip re-queueing the process
                        else:
                            num_restarted += 1
                    else:
                        logging.info(f"\nProcess {command.name} at {command.id} completed sucessfully with return code {retcode}: {gf.explain_exit_code(retcode)}.")
                        num_completed += 1

                elif command.flags["has_died"].is_set():

                    logging.error(f"\nProcess {command.name} at {command.id} heartbeat lost. Terminating.")

                    kill_command(command, proc, running_processes, allocation_array)

                    if check_and_mark_failed(command, max_restarts, restart_time_window, commands_to_run):
                        num_failed += 1
                        continue  # Skip re-queueing the process
                    else:
                        num_restarted += 1

        num_processes = len(commands_to_run)

        process_index = 0
        for gpu_index, (gpu_memory, use) in enumerate(zip(free_memory, gpu_use)):
            while gpu_memory >= total_memory_per_process and process_index < num_processes:
                gpu_memory -= total_memory_per_process

                commands_to_run[process_index].current_gpu = gpu_index
                commands_to_run[process_index].memory_assigned = total_memory_per_process

                process_index += 1

        for command in commands_to_run:
            if command.current_gpu > -1 and commands_to_run and len(running_processes) < max_num_processes:
                commands_to_run.remove(command)

                process = start_process(command, tensorflow_memory_per_process_mb)
                if process is not None:
                    running_processes.append((process, command))
                elif not command.has_failed:
                    logging.info(f"\nAttempting restart of {command.name}.")
                    commands_to_run.append(command)  # Re-queue if start_process failed
                    num_restarted += 1
                    time.sleep(wait_timer_seconds)

        clean_restart_counts(commands_to_run, restart_time_window)

        if gf.is_redirected():
            time.sleep(100)
    
    logging.info(f"\nAll processes finished. {num_completed}/{total}  completed, {num_failed}/{total}  failed.  {num_restarted} attempted restarts.")

class processCommand:
    def __init__(self, command_string: str):

        self.current_gpu = -1
        self.memory_assigned = 0

        self.flags = None
        self.pipe_name = None
        self.pipe_monitor_thread = None

        self.id = -1
        self.restart_count = 0
        self.total_restart_count = 0
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
        processCommand("python ./chapter_04/chapter_04_gw_perceptron.py"),
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


