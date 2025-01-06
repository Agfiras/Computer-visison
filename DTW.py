import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Generate or load two signals
x = np.linspace(0, 2 * np.pi, 100)
signal1 = np.sin(x)
signal2 = np.sin(x + np.pi / 4)

# Ensure signals are 1-D arrays
signal1 = signal1.flatten()
signal2 = signal2.flatten()

# Compute the DTW path
distance, path = fastdtw(signal1, signal2, dist=euclidean)

# Plot the signals
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(signal1, label='Signal 1')
plt.plot(signal2, label='Signal 2')
plt.legend()
plt.title('Signals')

# Plot the DTW path
plt.subplot(2, 1, 2)
for (i, j) in path:
    plt.plot([i, j], [signal1[i], signal2[j]], color='gray')
plt.plot(signal1, label='Signal 1')
plt.plot(signal2, label='Signal 2')
plt.legend()
plt.title('DTW Path')

plt.tight_layout()
plt.show()