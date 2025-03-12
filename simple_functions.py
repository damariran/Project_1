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

def simple_plot(x, y, my_title='y vs x', my_x_label='x', my_y_label='y'
                , my_legend='True', close=True):
    plt.plot(x,y, label='y vs x', alpha=0.5)
    plt.title(my_title)
    plt.xlabel(my_x_label)
    plt.ylabel(my_y_label)
    if my_legend:
        plt.legend()
    plt.grid(True)
    if close:
        plt.show()
    return