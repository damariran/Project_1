import numpy as np
import matplotlib.pyplot as plt

# wave parameters
t_start = 0 #[s]
duration = 10 #[s]
resolution = 0.01 #[s]
frequency = 0.75 #[Hz]
amplitude = 3 #[Arb]
noise_factor = 20 #[%] of amplitude
noise_factor = noise_factor * (1/100) * amplitude
# Here we generate the time vector
t = np.arange(t_start,t_start+duration,resolution)  #[s]  0s to 10s at 10ms intervals.
# now we generate the sine wave:
signal = amplitude*np.sin(2 * np.pi * frequency * t)  #This is the sine wave
# now we will generate the noize to addon the signal
noise = np.random.randn(len(t)) * noise_factor
# Now we combine the clean signal with the noise.
noisy_signal = signal + noise

# now we will compute basic statistics.
mean_signal = np.mean(noisy_signal)
std_signal = np.std(noisy_signal)
print(f"Mean: {mean_signal:.3f}")
print(f"STD: {std_signal:.3f}")

# Now we will plot the signal and the mean
plt.plot(t, noisy_signal, label = 'Noizy Signal', color = 'blue', alpha = 0.5)
plt.plot(t,signal, label = 'Clean Signal', color = 'red', alpha = 0.5) # al[ha is the transparency.
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.title('Ran Noizy wave')
plt.legend()
plt.grid(True)
plt.axhline(y =mean_signal, color = 'red', linestyle = '--', label = f'Mean = {mean_signal:.2f}') # Mean line
plt.show()
