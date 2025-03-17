# Clustering Oscillator Trajectories in PyCharm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.cluster import KMeans

# simulation parameters
sample_size = 100
sim_duration = 10
sim_resolution = 0.1
m = 1.0 # [kg] mass
k = 2.0 # [N/m] spring constant
critical_damping  = 2 * np.sqrt(m * k) # b_critical = 2.0

# Step 1: Define oscillator function
from simple_functions import damped_oscillator

# Step 2: Generate synthetic dataset with varying b
np.random.seed()
t = np.linspace(0, sim_duration, int(sim_duration/sim_resolution)) #[s] Time array
b_under = np.linspace(0.1, 1.9, int(sample_size/2))
b_over = np.linspace(2.1, 4.0, int(sample_size/2))
b_values = np.concatenate([b_under, b_over])
x = [] # Trajectories
true_labels = [] # For comparison (not used in this clustering)

for b in b_values:
    state0 = [1.0, 0.0] # Initial position=1, velocity=0
    solution = odeint(damped_oscillator, state0, t, args=(m, b, k))
    x.append(solution[:, 0])  # store position trajectory
    true_labels.append(0 if b < critical_damping else 1) # True label (for validation only)

x = np.array(x) # shape: (sample_size, sample_size)
true_labels = np.array(true_labels)

# Step 3: Feature engineering
max_amp = np.max(np.abs(x), axis=1)
number_of_direction_changes = np.sum(np.diff(x, axis=1) > 0, axis=1)
index_half_time = int(len(t)/2)
decay_rate = -np.log(np.max(np.abs(x[:, index_half_time:]), axis=1) / np.max(np.abs(x), axis=1)) / t[index_half_time]

x_features = np.column_stack([
    max_amp,
    number_of_direction_changes,
    decay_rate
])

# Step 4: Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(x_features)



