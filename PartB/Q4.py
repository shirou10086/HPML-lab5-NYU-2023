import matplotlib.pyplot as plt

# 数据
k_values = [1, 5, 10, 50, 100]  # K的值（百万）
times_without_unified = [4, 20, 41, 205, 409]  # 无统一内存的执行时间（毫秒）
times_with_unified = [42, 15, 17, 28, 66]      # 有统一内存的执行时间（毫秒）

# 创建图表
plt.figure(figsize=(12, 6))

# 图表1：无统一内存
plt.subplot(1, 2, 1)
plt.plot(k_values, times_without_unified, marker='o', color='b', label='Without Unified Memory')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('K (in million)')
plt.ylabel('Time to Execute (ms)')
plt.title('Execution Time without Unified Memory')
plt.legend()

# 图表2：有统一内存
plt.subplot(1, 2, 2)
plt.plot(k_values, times_with_unified, marker='o', color='r', label='With Unified Memory')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('K (in million)')
plt.ylabel('Time to Execute (ms)')
plt.title('Execution Time with Unified Memory')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
