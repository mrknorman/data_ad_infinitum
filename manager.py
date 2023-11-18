from typing import List

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
    
running_processes = []

# Handle termination signals and shut down gracefully
def signal_handler(signum, frame):
    logging.info(
        "\nReceived termination signal. Shutting down gracefully..."
    )

    for command in tqdm(running_processes):
        command.kill(running_processes, None)

    sys.exit(0)

# Main function
def main(
        commands_to_run,
        wait_timer_seconds : int = 3, 
        tensorflow_memory_per_process_mb : int = 2000, 
        cuda_overhead_per_process_mb : int = 1000, 
        max_restarts : int = 20, 
        restart_time_window : int = 1200, 
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
        logging.info(f"Initial GPU Memory Status: {free_memory}")
    except Exception as e:
        logging.error("Failed to get initial free memory array. See logs for details.")
        raise ValueError("Failed to get initial free memory array.")

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
            sys.stdout.write(
                f"\r{spinner[idx]} {len(running_processes)}/{total} running, {num_completed}/{total} completed, {num_failed}/{total} failed, {len(commands_to_run)}/{total} in queue. {num_restarted} restarts."
            )
            sys.stdout.flush()
        else:
            logging.info(
                f"{len(running_processes)}/{total} running, {num_completed}/{total} completed, {num_failed}/{total} failed, {len(commands_to_run)}/{total} in queue. {num_restarted} restarts."
            )

        try:
            free_memory = gf.get_memory_array()
            free_memory = free_memory + allocation_array
        except Exception as e:
            logging.exception("\nFailed to update free memory array.")
        
        for command in running_processes:
            if command.process is not None:
                retcode = command.process.poll()
                if retcode is not None:  # Process finished.

                    command.kill(running_processes, allocation_array)

                    stdout, stderr = command.process.communicate()

                    # Check if the process should be marked as failed
                    if retcode != 0:  # Process failed, log the error

                        logging.error((
                            f"\nProcess {command.name} at {command.id}"
                            f" failed with return code {retcode} :"
                            f" {gf.explain_exit_code(retcode)}."
                        ))

                        if stdout:
                            logging.error(
                                f"\nProcess {command.name} at {command.id} - STDOUT: {stdout.decode()}"
                            )
                        if stderr:
                            logging.error(
                                f"\nProcess {command.name} at {command.id} - STDERR: {stderr.decode()}"
                            )

                        if command.check_if_failed(max_restarts, restart_time_window):
                            num_failed += 1
                        else:
                            num_restarted += 1
                            commands_to_run.insert(0, command)
                    else:
                        logging.info((
                            f"\nProcess {command.name} at {command.id} "
                            f"completed sucessfully with return code {retcode}: "
                            f"{gf.explain_exit_code(retcode)}."
                        ))
                        num_completed += 1
                        command.has_completed = True

                elif command.flags["has_completed"].is_set():

                    logging.error((
                        f"\nProcess {command.name} at {command.id}"
                        " failed to complete gracefully. Forcing termination."
                    ))
                    command.has_completed = True

                    command.kill(running_processes, allocation_array)

                elif command.flags["has_died"].is_set():

                    logging.error(
                        f"\nProcess {command.name} at {command.id} heartbeat lost. Terminating."
                    )

                    command.kill(running_processes, allocation_array)

                    if command.check_if_failed(max_restarts, restart_time_window):
                        num_failed += 1
                    else:
                        commands_to_run.insert(0, command)
                        num_restarted += 1

        num_processes = len(commands_to_run)

        process_index = 0
        for gpu_index, gpu_memory in enumerate(free_memory):
            while gpu_memory >= total_memory_per_process \
                and process_index < num_processes:

                gpu_memory -= total_memory_per_process

                commands_to_run[process_index].current_gpu = gpu_index
                commands_to_run[process_index].memory_assigned = total_memory_per_process

                process_index += 1

        for command in commands_to_run:
            if command.current_gpu > -1 and commands_to_run \
                and len(running_processes) < max_num_processes:

                commands_to_run.remove(command)

                command.start(tensorflow_memory_per_process_mb)
                
                if command.process is not None:
                    running_processes.append(command)
                elif not command.has_failed or not command.has_completed:
                    logging.info(
                        f"\nAttempting restart of {command.name}."
                    )
                    commands_to_run.append(command)  # Re-queue if start_process failed
                    num_restarted += 1
                    time.sleep(wait_timer_seconds)

        if gf.is_redirected():
            time.sleep(100)
    
    logging.info((
        f"\nAll processes finished. {num_completed}/{total}"Process
        f" completed, {num_failed}/{total}  failed."
        f" {num_restarted} attempted restarts."
    ))

class Process:
    def __init__(self, command_string: str, memory_to_request : int):

        self.current_gpu = -1
        self.memory_assigned = 0
        self.memory_desired = memory_to_request

        self.flags = None
        self.pipe_name = None
        self.pipe_monitor_thread = None

        self.process = None
        self.id = -1
        self.restart_count = 0
        self.total_restart_count = 0
        self.restart_counter = time.time()
        self.full = command_string
        self.has_failed = False
        self.has_completed = False

        self.manager = None

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

    # Modify the start_process function to track first-time start
    def start(self):
        try:
            process_gap_seconds = 3
            logging.info((
                f"\nWaiting {process_gap_seconds} s"
                " to space out process activations."
            ))
            time.sleep(process_gap_seconds)  # Space out so not too many are run at the same time

            # Determine the mode for log files based on whether it's a first-time start
            mode = "w" if self.total_restart_count == 0 else "a"

            with open(f"perceptron_logs/{self.name}_log.txt", mode) as stdout_file, \
                open(f"perceptron_logs/{self.name}_error.txt", mode) as stderr_file:
                full_command = (
                    f"{self.full} --gpu {self.current_gpu}"
                    f" --request_memory {self.memory_desired}"
                    f" --restart_count {self.total_restart_count}"
                    f" --name {self.name}"
                )

                self.pipe_name = f"./tmp/heartbeat_{self.name}"
                gf.create_named_pipe(self.pipe_name)

                self.flags = {
                    "has_died" : threading.Event(), 
                    "should_exit" : threading.Event(), 
                    "has_completed": threading.Event()
                }
                self.pipe_monitor_thread = gf.start_monitoring_thread(
                    self, self.flags
                )

                self.process = subprocess.Popen(
                    full_command, 
                    shell=True, 
                    stdout=stdout_file, 
                    stderr=stderr_file, 
                )

                self.id = self.process.pid
                logging.info(
                    f"\nProcess: {self.name} started at {self.id}"
                )

        except Exception as e:
            logging.exception((
                f"\nFailed to start process {self.name}"
                " on GPU {self.current_gpu}."
            ))
            self.restart_count += 1
            self.total_restart_count += 1
            return None

    def kill(self):
        gf.kill_process(self.id)
        self.manager.running.remove(self)

        if self.manager.allocated_memory is not None:
            self.manager.allocated_memory[
                self.current_gpu
            ] -= self.memory_assigned
        
        self.process = None
        self.current_gpu = -1
        self.memory_assigned = 0
        self.flags["should_exit"].set()
        gf.cleanup_named_pipe(self.pipe_name)

    def check_if_completed(self):
        if self.has_completed:
            self.kill()
            self.has_completed = True
            self.manager.completed.append(self)

    def check_if_failed(self):

        # Check if already failed
        if self.has_failed or (self in self.manager.failed):
            self.kill()
            self.has_failed = True  # Mark process as failed
            self.manager.failed.append(self)

            return True
        
        # Update fail restart timer:
        current_time = time.time()
        if current_time - self.restart_counter > self.manager.restart_time_window:
            self.restart_counter = current_time
            self.restart_count = 0

        # Check if the process has been restarted more than N times in X seconds
        self.restart_count += 1
        self.total_restart_count += 1
        if self.restart_count > self.manager.max_restarts:
            logging.error((
                f"\nProcess {self.name} has been restarted "
                f"{self.restart_count} times within {self.manager.restart_time_window}"
                f" seconds. Marking as failed."
            ))
            self.has_failed = True  # Mark process as failed
            self.kill()
            self.manager.failed.append(self)
            return True
        
        return False      

    def get_retcode(self):
        return self.process.poll()

    def print_stderr(self):
        stdout, stderr = self.process.communicate()
        if stdout:
            logging.error(
                f"\nProcess {self.name} at {self.id} - STDOUT: {stdout.decode()}"
            )
        if stderr:
            logging.error(
                f"\nProcess {self.name} at {self.id} - STDERR: {stderr.decode()}"
            )

    def complete(self):
        self.kill()
        self.has_completed = True
        self.manager.completed.append(self)

    def requeue(self):
        if not self.check_if_failed():
            self.manager.queue.insert(0, self)
            self.kill()


class Manager:

    running = []
    failed = []
    completed = []
    free_memory = []
    allocated_memory =[]

    def __init__(
            self, 
            to_run : List[Process],
            max_restarts : int = 10,
            restart_timeout_seconds : float = 1200
        ):

        self.queue = [to_run]

        for process in self.queue:
            process.manager = self

        self.max_restarts = max_restarts
        self.restart_timeout_seconds = restart_timeout_seconds

    def run(self):

        while self.queue or self.running:

            self.update_memory_array()
            yield 0

    def update_memory_array(self):
        try:
            self.free_memory = gf.get_memory_array()
            self.free_memory += self.allocated_memory
        except Exception as e:
            logging.exception("\nFailed to update free memory array.")

    def manage_running_processes(self):

        for process in self.running[:]:
            if process.process is not None:

                if process.check_if_failed():
                    logging.warning(
                        "Failed process found in running jobs for some reason! This is concerning..."
                    )
                    self.queue.remove(process)
                    continue
                if process.check_if_completed():
                    logging.warning(
                        "Completed process found in running jobs for some reason! This is concerning..."
                    )
                    self.queue.remove(process)
                    continue
                
                # Manage process exit:
                retcode = process.get_retcode()
                if retcode is not None:  # Process finished.

                    process.print_stderr()
                    process.kill()

                    # Check if the process should be marked as failed
                    if retcode != 0:  # Process failed, log the error
                        logging.error((
                            f"\nProcess {process.name} at {process.id}"
                            f" failed with return code {retcode} :"
                            f" {gf.explain_exit_code(retcode)}."
                        ))

                        process.requeue()
                    else:
                        logging.info((
                            f"\nProcess {process.name} at {process.id} "
                            f"completed sucessfully with return code {retcode}: "
                            f"{gf.explain_exit_code(retcode)}."
                        ))
                        process.complete()

                # Check if monitor thread has spotted completion signal:
                elif process.flags["has_completed"].is_set():
                    logging.error((
                        f"\nProcess {process.name} at {process.id}"
                        " failed to complete gracefully. Forcing termination."
                    ))
                    process.complete()

                # Check if monitor thread has marked process as dead:
                elif process.flags["has_died"].is_set():
                    logging.error(
                        f"\nProcess {process.name} at {process.id} heartbeat lost. Terminating."
                    )
                    process.requeue()


if __name__ == "__main__":
    commands_to_run = [
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 64"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 64 64"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128 64"),
        Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128 128"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 64"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 128"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 256"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 64 64 64"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 128 128 128"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 256 256 256"),
        #Process("python ./chapter_04/chapter_04_gw_perceptron.py --layers 512")
    ]

    commands_to_run[0].total_restart_count +=1 

    main(commands_to_run)


