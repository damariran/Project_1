import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel
from scipy import fft, signal

# simulation parameters:
sim_duration = 50
resolution = 0.1  # [%]
low_pass_cutoff_freq = 3  # [Hz]
number_of_waves = 10
frequency_range_of_waves = [0.1, 10]


# Generate a synthetic signal out of 3 random signals
def synthetic_signal(sim_dura, res, num_wave, freq_range):
    time = np.linspace(0, sim_dura, int(abs(100 / res)))
    syn_signal = np.zeros(len(time))
    for i in range(1,num_wave):
        Amp = np.random.uniform(0.1, 1) # amplitude between 0 and 1
        Phase = np.random.uniform(-np.pi, np.pi) # phase between -pi and pi
        Omega = np.random.uniform(freq_range[0], freq_range[1]) # frequency range
        syn_signal = syn_signal + Amp * np.sin((np.pi * Omega * time) - Phase)
    return syn_signal, time


#  Here we generated a random signal
[data, t] = synthetic_signal(sim_duration, resolution, number_of_waves, frequency_range_of_waves)
# Apply a low-pass filter
fs = len(t) / (t[-1] - t[0])  # [Hz] the t[-1] means we get the last element of array t
sos = signal.butter(3, low_pass_cutoff_freq, 'low', fs=fs, output='sos')  # 5th order Butterworth filter
filtered_data = signal.sosfilt(sos, data)

# calculate fft
frequencies = fft.fftfreq(len(t), 1/fs)  # frequency bins
fft_values = fft.fft(data)  # fft of original data
positive_frequencies = frequencies[frequencies >= 0]  # returns only the positive frequency bins.
magnitudes = np.abs(fft_values)
positive_magnitudes = magnitudes[frequencies >= 0]  # returns the right side magnitudes only.

# plot data
plt.figure(figsize=(8, 6)); plt.subplot(3, 1, 1)
plt.plot(t, data, label='Original data', color='blue', alpha=0.5)
plt.plot(t, filtered_data, label='Filtered data', color='red')
plt.xlabel('Time[S]'); plt.ylabel('Amplitude')
plt.title('Original vs Filtered Signal')
plt.legend(); plt.grid(True)

#  plot the fft
plt.subplot(3,1,2)
plt.plot(positive_frequencies, positive_magnitudes, label='fft', color='purple')
plt.xlabel('Frequency[Hz]'); plt.ylabel('Magnitude')
plt.title('Frequency domain'); plt.grid(True)
plt.xlim(frequency_range_of_waves[0],frequency_range_of_waves[1])

# Plot zoomed-in filtered datat
plt.subplot(3, 1, 3)
plt.plot(t, filtered_data, label='Filtered Data', color='red')
plt.xlabel('Time[s]')
plt.ylabel('Magnitude')
plt.title('Zommed Filtered Signal')
plt.grid(True)
plt.xlim(0, low_pass_cutoff_freq)
plt.legend()
plt.tight_layout()
plt.show()

# print dominant frequencies
peak_indices = signal.find_peaks(positive_magnitudes,
                                 height=np.max(positive_magnitudes)*0.2)
peak_indices = peak_indices[0]
dominant_frequencies = positive_frequencies[peak_indices]
print(f'Dominant frequencies: {dominant_frequencies}')
