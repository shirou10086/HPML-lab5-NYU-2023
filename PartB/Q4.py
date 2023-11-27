import matplotlib.pyplot as plt
import numpy as np

# Data from your experiments
K_values = [1, 5, 10, 50, 100]  # K values in millions
cpu_times = [4, 20, 41, 205, 409]  # CPU times in ms

# GPU times without Unified Memory (Step 2)
gpu_times_step2 = {
    "1 Block, 1 Thread": [0, 0, 0, 0, 0],
    "1 Block, 256 Threads": [0, 0, 0, 0, 0],
    "Multiple Blocks, 256 Threads/Block": [0, 0, 0, 0, 1]
}

# GPU times with Unified Memory (Step 3)
gpu_times_step3 = [0, 0, 0, 0, 0]  # All times are 0 ms

# Function to plot the data
def plot_data(title, cpu_times, gpu_times):
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, cpu_times, label='CPU', marker='o')

    for scenario, times in gpu_times.items():
        plt.plot(K_values, times, label=scenario, marker='o')

    plt.xlabel('K (in millions)')
    plt.ylabel('Time (ms)')
    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the charts
plot_data('Execution Time without Unified Memory (Step 2)', cpu_times, gpu_times_step2)
plot_data('Execution Time with Unified Memory (Step 3)', cpu_times, {"Unified Memory": gpu_times_step3})
