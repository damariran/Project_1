import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal

# simulation parameters:
sim_duration = 50
resolution = 0.1  # [%]
low_pass_cutoff_freq = 3  # [Hz]
number_of_waves = 10
frequency_range_of_waves = [0.1, 10]


# Generate a synthetic signal out of 3 random signals
def synthetic_signal(sim_dura, res, num_wave, freq_range):
    time = np.linspace(0, sim_dura, int(abs(100 / res))) #  first, we define the simulation time
    syn_signal = np.zeros(len(time)) #  start with an empty signal.
    for i in range(1,num_wave):
        Amp = np.random.uniform(0.1, 1) # amplitude between 0 and 1
        Phase = np.random.uniform(-np.pi, np.pi) # phase between -pi and pi
        Omega = np.random.uniform(freq_range[0], freq_range[1]) # frequency range
        syn_signal = syn_signal + Amp * np.sin((np.pi * Omega * time) - Phase) # each loop we add a signal on top of the previous one.
    return syn_signal, time

def signal_fiter(syn_signal, time, low_pass_cut_freq):
    sampling_frequency = len(time) / (time[-1] - time[0]) # [Hz] the t[-1] means we get the last element of array t
    sos = signal.butter(3, low_pass_cut_freq, 'low', fs=sampling_frequency, output='sos') # 5th order Butterworth filter the output is the second-order-sections coefficients (SOS)
    filtered_signal = signal.sosfilt(sos, syn_signal) #  this function employ the SOS coefficients to the signal filter.
    return filtered_signal, sampling_frequency

def simple_fft(x, y):
    sampling_frequency = len(x) / (x[-1] - x[0])
    x_fft = fft.fftfreq(len(x), 1/sampling_frequency)
    y_fft = fft.fft(y)
    Positive_x_fft = x_fft[x_fft >= 0]  # returns only the positive frequency bins.
    y_fft = np.abs(y_fft)
    positive_y_fft = y_fft[x_fft >= 0]  # returns the right side magnitudes only.
    return Positive_x_fft, positive_y_fft

#  Here we generated a random signal
[data, t] = synthetic_signal(sim_duration, resolution, number_of_waves, frequency_range_of_waves)
# Apply a low-pass filter
[filtered_data, fs] = signal_fiter(data, t, low_pass_cutoff_freq)
# calculate fft
[positive_frequencies, positive_magnitudes] = simple_fft(t, data)


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
