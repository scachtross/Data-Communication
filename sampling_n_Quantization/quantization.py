import numpy as np
import matplotlib.pyplot as plt

# Parameters for message signal
tot = 1
td = 0.002
t = np.arange(0, tot, td)
ts = 0.02
n = np.arange(0, tot, ts)
x_sampled = np.sin(2 * np.pi * n) - np.sin(6 * np.pi * n)

# Define quantization levels
levels = 16
x_min = np.min(x_sampled)
x_max = np.max(x_sampled)
step = (x_max - x_min) / levels

# Quantize the sampled signal
x_quantized = step * np.round((x_sampled - x_min) / step) + x_min

# Plot quantized vs. sampled signal
plt.figure()
plt.stem(n, x_sampled, 'r', markerfmt='ro', basefmt=" ", label='Sampled Signal')
plt.stem(n, x_quantized, 'b--', markerfmt='bo', basefmt=" ", label='Quantized Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampled Signal vs. Quantized Signal')
plt.legend()
plt.grid(True)
plt.show()

# Quantization error
quantization_error = x_sampled - x_quantized
plt.figure()
plt.stem(n, quantization_error, basefmt=" ")
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('Quantization Error')
plt.grid(True)
plt.show()