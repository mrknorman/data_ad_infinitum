import time
import gravyflow as gf
import numpy as np

# Placeholder for the actual process start logic.
def start_process(process_id, gpu_id):
    # Logic to start the process on the specified GPU
    pass

if __name__ == "__main__":

    wait_timer_seconds = 30
    tensorflow_memory_per_process_mb = 8000
    cuda_overhead_per_process_mb = 2000

    total_memory_per_process = tensorflow_memory_per_process_mb + cuda_overhead_per_process_mb
    
    processes = [
        1, 2, 3, 5, 6, 7, 8
    ]

    while processes:

        # Check for the possibility that getting free memory fails
        try:
            free_memory = gf.get_memory_array()
        except Exception as e:
            print(f"Failed to get free memory array: {e}")
            free_memory = []

        num_processes = len(processes)
        assignment_array = np.full(num_processes, -1)
        
        process_index = 0
        for gpu_index, gpu_memory in enumerate(free_memory):
            while gpu_memory >= total_memory_per_process and process_index < num_processes:
                gpu_memory -= total_memory_per_process
                assignment_array[process_index] = gpu_index
                process_index += 1
        
        for i, gpu in enumerate(assignment_array):
            if gpu > -1 and processes:
                process_id = processes.pop(0)
                start_process(process_id, gpu)  # Start the process here

        time.sleep(wait_timer_seconds)