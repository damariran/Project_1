import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint



# Define the ODEs (Ordinary differential equations)
def damped_oscillator(state, t, m, b, k):
    x, v = state  # state = [position, velocity]
    [dx_dt, dv_dt] = [v, -(b / m) * v - (k / m) * x]
    return [dx_dt, dv_dt]


# set parameters and initial conditions
t_start = 0 # [0]
t_end = 20 # [0]
resolution = 0.1 # [0]
m = 1.0  # [kg]
b = 0.25  # [kg//s] damping coefficient
k = 2.0  # [N/m] spring constant
x0 = 1.0  # [m] initial position
v0 = 0.0  # [m/s] initial velocity
state0 = [x0, v0]  # initial state
number_of_steps = int(abs((t_end - t_start) / resolution))
# define the time array for the solution
t = np.linspace(t_start, t_end, number_of_steps)

# solve the ODE
solution = odeint(damped_oscillator, state0, t, args=(m, b, k))
x = solution[:, 0] # the solution position vector
v = solution[:, 1] # the solution velocity vector

# compute kinetic and potential energy
Kinetic_Energy = 0.5 * m * v**2
Potential_Energy = 0.5 * k * x**2
Total_Energy = Kinetic_Energy + Potential_Energy

# plot position and velocity
plt.figure(figsize=(14,7))
plt.subplot(2, 2, 1)
plt.plot(t, x, label='position(x)', color='blue')
plt.plot(t, v, label='velocity(v)', color='red')
plt.legend()
plt.xlabel('Time[s]')
plt.ylabel('Amplitude[m, m/s]')
plt.title('Damped Harmonic Oscillator')
plt.grid(True)

# plot the energy
plt.subplot(2,2,2)
plt.plot(t, Kinetic_Energy, label='Kinetic Energy', color='green')
plt.plot(t, Potential_Energy, label='Potential Energy', color='orange')
plt.plot(t, Total_Energy, label='Total Energy', color='purple')
plt.legend()
plt.xlabel('Time[s]')
plt.ylabel('Energy[J]')
plt.title('Energy over time')
plt.grid(True)

# plot the phase space
plt.subplot(2,2,3)
plt.plot(x, v, label='phase space', color='black')
plt.legend()
plt.xlabel('Position[m]')
plt.ylabel('velocity[m/s]')
plt.title('Damped oscillator phase space')
plt.grid(True)
plt.show()
