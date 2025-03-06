import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal

# simulation parameters:
sim_duration = 10
resolution = 0.1  # [%]
low_pass_cutoff_freq = 2  # [Hz]


# Generate a synthetic signal out of 3 random signals
def synthetic_signal(sim_dura, res):
    Amp = [np.random.uniform(0.1, 1),
           np.random.uniform(0.1, 1),
           np.random.uniform(0.1, 1)]  # amplitude between 0 and 1
    Phase = [np.random.uniform(-np.pi, np.pi),
             np.random.uniform(-np.pi, np.pi),
             np.random.uniform(-np.pi, np.pi)]  # phase between -pi and pi
    Omega = [np.random.uniform(0.1, sim_duration * 5),
             np.random.uniform(0.1, sim_duration * 5),
             np.random.uniform(0.1, sim_duration * 5)]  # frequency between 0.1Hz and 5Hz
    time = np.linspace(0, sim_dura, int(abs(100 / res)))
    syn_signal = (Amp[0] * np.sin((np.pi * Omega[0] * time) - Phase[0]) +
              Amp[1] * np.sin((np.pi * Omega[1] * time) - Phase[1]) +
              Amp[2] * np.sin((np.pi * Omega[2] * time) - Phase[2]))
    return syn_signal, time


#  Here we generated a random signal
[data, t] = synthetic_signal(sim_duration, resolution)
# Apply a low-pass filter
fs = len(t) / (t[-1] - t[0])  # [Hz] the t[-1] means we get the last element of array t
sos = signal.butter(3, low_pass_cutoff_freq, 'low', fs=fs, output='sos')  # 5th order Butterworth filter
filtered_data = signal.sosfilt(sos, data)

# calculate fft
frequencies = fft.fftfreq(len(t), 1 / fs)  # frequency bins
fft_values = fft.fft(data)  # fft of original data
positive_frequencies = frequencies[frequencies >= 0]  # returns only the positive frequency bins.
magnitudes = np.abs(fft_values)
positive_magnitudes = magnitudes[magnitudes >= 0]  # returns the right side magnitudes only.

# plot data
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, data, label='Original data', color='blue', alpha=0.5)
plt.plot(t, filtered_data, label='Filtered data', color='red')
plt.show()