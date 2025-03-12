import numpy as np
from simple_functions import simple_plot

x = np.linspace(0, 10, 100)
y1 = x**2 -3*x +16

simple_plot(x, y1, close=False)

y2 = -x**3 + 2*x**2

simple_plot(x,y2, close=True)