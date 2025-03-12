import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
from simple_functions import simple_fft, simple_plot, simple_butter_filter

# simulation parameters:
sim_duration = 50   #[s]
resolution = 0.1  # [%]
number_of_waves = 10 #  How many sin waves will compose this random signal
frequency_range_of_waves = [0.1, 10] # What will be the range of frequencies for this random signal

l_p_c_f = 1  # [Hz] Low pass cutoff frequency for the filter

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

#  Here we generated a random signal
[data, t] = synthetic_signal(sim_duration, resolution, number_of_waves, frequency_range_of_waves)
# Apply a low-pass filter
[filtered_data, fs] = simple_butter_filter(t, data,
                                                my_low_pass_cutoff_frequency=l_p_c_f,
                                                filter_order=3)
# calculate fft
[positive_frequencies, positive_magnitudes] = simple_fft(t, data)

# plot data
simple_plot(t, data,
            sub_plot=[3,1,1],
            my_legend='Original signal',
            my_title='Original signal',
            my_x_label='Time[s]',
            my_y_label='Amplitude',
            close=False)

#  plot the fft
simple_plot(positive_frequencies, positive_magnitudes,
            my_xlimit=[frequency_range_of_waves[0],frequency_range_of_waves[1]],
            sub_plot=[3,1,2],
            my_legend='Frequency domain',
            my_x_label='Frequency[Hz]',
            my_y_label='Magnitude',
            close=False)

# Plot zoomed-in filtered data
simple_plot(t, filtered_data,
            sub_plot=[3,1,3],
            my_legend='Zoomed in signal',
            my_x_label='Time[s]',
            my_y_label='Magnitude',
            close=True)

# print dominant frequencies
peak_indices = signal.find_peaks(positive_magnitudes,
                                 height=np.max(positive_magnitudes)*0.2)
peak_indices = peak_indices[0]
dominant_frequencies = positive_frequencies[peak_indices]
print(f'Dominant frequencies: {dominant_frequencies}')
