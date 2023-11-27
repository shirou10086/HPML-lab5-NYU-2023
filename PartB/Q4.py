import matplotlib.pyplot as plt
import numpy as np

# Values of K in million
K_values = [1, 5, 10, 50, 100]

# Execution times for Step 2 (without Unified Memory) for each K value
step2_times = [2.34, 12.3, 24.6, 123.4, 246.8]  # Replace with your actual data

# Execution times for Step 3 (with Unified Memory) for each K value
step3_times = [1.23, 6.2, 12.4, 62.0, 124.0]  # Replace with your actual data

# Execution times on CPU only for each K value
cpu_times = [0.5, 2.5, 5.0, 25.0, 50.0]  # Replace with your actual data

# Create log-log scale plots
plt.figure(figsize=(10, 6))

# Plot Step 2 execution times
plt.loglog(K_values, step2_times, marker='o', label='Step 2 (without Unified Memory)')

# Plot Step 3 execution times
plt.loglog(K_values, step3_times, marker='o', label='Step 3 (with Unified Memory)')

# Plot CPU execution times for reference
plt.loglog(K_values, cpu_times, marker='o', label='CPU Only')

# Set labels and title
plt.xlabel('K (Million)')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time vs. K for Step 2 and Step 3')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
