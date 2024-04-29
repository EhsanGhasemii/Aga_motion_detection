import numpy as np
import matplotlib.pyplot as plt

N = 256  # Number of points
t = np.arange(N)

# Create a sine and cosine wave
x = np.sin(2 * np.pi * t / N)
y = np.cos(2 * np.pi * t / N)

# Combine the sine and cosine waves into a complex signal
signal = x + 1j*y

# Compute the FFT of the signal
fft_result = np.fft.fft(signal)

# Print the FFT results
for i in range(N):
    print(f"({fft_result[i].real}, {fft_result[i].imag})")

# If you want to plot the FFT result
plt.plot(np.abs(fft_result))
plt.show()
