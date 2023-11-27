import matplotlib.pyplot as plt

# Values of K in million
K_values = [1, 5, 10, 50, 100]

# Execution times for Q1, Q2, and Q3 (in milliseconds)
q1_times = [4, 20, 41, 205, 409]
q2_times = [0, 0, 0, 0, 1]
q3_times = [0, 0, 0, 0, 0]

# Create a log-log scale plot for Q1
plt.figure(figsize=(10, 6))
plt.plot(K_values, q1_times, marker='o', label='Q1 (Step 2)')
plt.plot(K_values, q3_times, marker='o', label='Q3 (Step 3 with Unified Memory)')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('K (Million)')
plt.ylabel('Time (ms)')
plt.title('Execution Time Comparison (Step 2 vs. Step 3 with Unified Memory)')
plt.legend()
plt.grid(True)
plt.savefig('q4_q1_vs_q3.png')
plt.show()

# Create a log-log scale plot for Q2
plt.figure(figsize=(10, 6))
plt.plot(K_values, q2_times, marker='o', label='Q2 (Step 2 with GPU)')
plt.plot(K_values, q3_times, marker='o', label='Q3 (Step 3 with Unified Memory)')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('K (Million)')
plt.ylabel('Time (ms)')
plt.title('Execution Time Comparison (Step 2 with GPU vs. Step 3 with Unified Memory)')
plt.legend()
plt.grid(True)
plt.savefig('q4_q2_vs_q3.png')
plt.show()
