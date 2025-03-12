import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal

def simple_fft(x, y):
    sampling_frequency = len(x) / (x[-1] - x[0])
    x_fft = fft.fftfreq(len(x), 1/sampling_frequency)
    y_fft = fft.fft(y)
    Positive_x_fft = x_fft[x_fft >= 0]  # returns only the positive frequency bins.
    y_fft = np.abs(y_fft)
    positive_y_fft = y_fft[x_fft >= 0]  # returns the right side magnitudes only.
    return Positive_x_fft, positive_y_fft

def simple_plot(x, y, my_xlimit=None, sub_plot=None, my_legend=None, my_title=None, my_x_label=None, my_y_label=None, close=True):

    if sub_plot is None:
        pass
    else:
        plt.subplot(sub_plot[0],sub_plot[1],sub_plot[2])

    if len(x) == len(y):
        plt.plot(x,y, alpha=0.5)
    else:
        print('The x and y vectors are not in the same length!')
        return None

    if my_xlimit is None:
        pass
    else:
        plt.xlim(my_xlimit[0], my_xlimit[1])

    if my_title is None:
        pass
    else:
        plt.title(my_title)

    if my_x_label is None:
        pass
    else:
        plt.xlabel(my_x_label)

    if my_y_label is None:
        pass
    else:
        plt.ylabel(my_y_label)

    if my_legend is None:
        pass
    else:
        plt.legend([my_legend])

    plt.grid(True)

    if close:
        plt.show()
    return

def simple_butter_filter(x, y, my_low_pass_cutoff_frequency=None, filter_order=3):
    if my_low_pass_cutoff_frequency is None:
        low_pass_cut_freq = 5 #[Hz]
    else:
        pass

    sampling_frequency = len(x) / (x[-1] - x[0]) # [Hz] the t[-1] means we get the last element of array t

    sos = signal.butter(filter_order, my_low_pass_cutoff_frequency, 'low', fs=sampling_frequency, output='sos') # 5th order Butterworth filter the output is the second-order-sections coefficients (SOS)

    filtered_signal = signal.sosfilt(sos, y) #  this function employ the SOS coefficients to the signal filter.

    return filtered_signal, sampling_frequency