import matplotlib.pyplot as plt
import numpy as np

# 数据
K_values = [1, 5, 10, 50, 100]
cpu_times = [4, 20, 41, 205, 409]  # Q1 的 CPU 时间

# Q2 的 GPU 时间（不使用统一内存）
gpu_times_q2 = {
    "1 Block, 1 Thread": [76, 90, 77, 76, 49],
    "1 Block, 256 Threads": [9, 8, 7, 8, 7],
    "Multiple Blocks, 256 Threads/Block": [22, 85, 158, 760, 1501]
}

# Q3 的 GPU 时间（使用统一内存）
gpu_times_q3 = {
    "1 Block, 1 Thread": [42, 15, 17, 28, 66],
    "1 Block, 256 Threads": [42, 15, 17, 28, 66],  # 假设数据与 1 Block, 1 Thread 相同
    "Multiple Blocks, 256 Threads/Block": [42, 15, 17, 28, 66]  # 假设数据与 1 Block, 1 Thread 相同
}

# 绘制图表
def plot_chart(times, title):
    plt.figure(figsize=(10, 6))
    for scenario, scenario_times in times.items():
        plt.plot(K_values, scenario_times, label=scenario)
    plt.plot(K_values, cpu_times, label="CPU", linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('K (in millions)')
    plt.ylabel('Time (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制步骤 2 的图表
plot_chart(gpu_times_q2, "Step 2: GPU Execution Times Without Unified Memory")

# 绘制步骤 3 的图表
plot_chart(gpu_times_q3, "Step 3: GPU Execution Times With Unified Memory")
