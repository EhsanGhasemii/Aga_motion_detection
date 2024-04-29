import numpy as np
import matplotlib.pyplot as plt

N = 256  # Number of points
t = np.arange(N)

# Create a sine wave
x = np.sin(2 * np.pi * t / N)

# Compute the FFT of the signal
fft_result = np.fft.fft(x)

# Print the FFT results
for i in range(N):
    print(f"{i}: ({fft_result[i].real}, {fft_result[i].imag})")

# If you want to plot the FFT result
plt.plot(np.abs(fft_result))
plt.show()
