import numpy as np
from simple_functions import simple_plot

x = np.linspace(0, 10, 100)
y1 = x**2 -3*x +16

simple_plot(x, y1, sub_plot=[2,1,1], my_legend='parabola 1',my_title='Smiling parabola', my_x_label='x[s]',my_y_label='y[m]', close=False)

y2 = -x**3 + 2*x**2

simple_plot(x,y2, sub_plot=[2,1,2], my_legend='parabola 2',my_title='crying parabola', my_x_label='x[s]',my_y_label='y[m]', close=True)