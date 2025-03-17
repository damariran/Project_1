import matplotlib.pyplot as plt
import numpy as np
from simple_functions import simple_plot

t = np.linspace(0, 50, 500)
w=0.3
phi =0
gamma = 0.1
A = np.cos(2 * np.pi * (w - phi) * t) * np.exp(-gamma * t)

number_of_direction_changes = np.sum(np.diff(A, axis=0) > 0, axis=0)

simple_plot(t, A, sub_plot=[2,1,1], close=False)

A_diff = np.diff(A)
simple_plot(np.linspace(0,1,len(A_diff)),A_diff, sub_plot=[2,1,2])
