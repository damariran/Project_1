import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Deine a quadratic function
def f(x, a, b, c):
    return a * x ** 2 + b * x + c


# simulation parameters
x_start = -5  # [s]
x_end = 5  # [s]
resolution = 1  # [s]

a1, b1, c1 = 1, -2, 3  # The true parameters
noise_factor = 2

steps = int(abs((x_end - x_start) / resolution))


# generate noisy data
x = np.linspace(x_start, x_end, steps)
y = f(x, a1, b1, c1)  # the true quadratic function
noise = np.random.randn(len(x)) * noise_factor
y_noisy = y + noise  # noisy data

# Fit a curve
popt, pcov = curve_fit(f, x, y_noisy)
a2, b2, c2 = popt  # Extracted fitted parameters
x_fit = np.linspace(x_start, x_end, steps * 10)
y_fit = f(x_fit, a2, b2, c2)

# Plot the data
plt.scatter(x, y_noisy, label='noisy data', color='blue', alpha=0.5)
plt.plot(x, y, label='True function', color='green')
plt.plot(x_fit, y_fit, label='fitted curve', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fit to noisy data')
plt.legend()
plt.grid(True)
plt.show()

# print results
print(f"True parameters: a={a1}, b={b1}, c={c1}")
print(f"fitted parameters: a={a2}, b={b2}, c={c2}")