import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Simulated frequency range (Hz)
freq = np.logspace(-3, 3, 500)
omega = 2 * np.pi * freq

# Simulated imaginary part of impedance (Z_im)
Z_im = np.log10(freq)  # Replace with your actual data

# Apply Hilbert transform to get real part (Z_re)
Z_re = np.imag(hilbert(Z_im))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(freq, Z_im, label='Imaginary Part (Input)')
plt.plot(freq, Z_re, label='Real Part (Kramers-Kronig)')
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance')
plt.title('Kramers-Kronig Relation via Hilbert Transform')
plt.legend()
plt.grid(True)
plt.show()
