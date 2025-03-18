# Clustering Oscillator Trajectories in PyCharm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.cluster import KMeans

# simulation parameters
sample_size = 100
sim_duration = 10
sim_resolution = 0.1
noise_factor = 0.01
m = 1.0 # [kg] mass
k = 2.0 # [N/m] spring constant
critical_damping  = 2 * np.sqrt(m * k) # b_critical = 2.0
number_of_clusters = 2

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
    true_labels.append(0 if b < critical_damping else 1) # True label (for validation only! This association of 0's and 1's are with the knowledge that the Kmeans algorithm will return those designations of labels. )

x = np.array(x) # shape: (sample_size, sample_size)
x +=np.random.randn(*x.shape)  *noise_factor  # "*x.shape" is for the unpacking (instead of (n,m) it's n m
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
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(x_features) # the default labels are 1's and 0's

# step 5: compare with true labels (for learning purposes)
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(true_labels, cluster_labels) *100 # [%]
print(f'Adjusted Rand Index (similarity to true labels): {ari:.2f}')

# step 6: Plot trajectories colored by cluster
plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
for i in range(len(x)):
    color = 'blue' if cluster_labels[i] == 0 else 'red'
    plt.plot(t, x[i], color=color, alpha=0.3)

plt.xlabel('Time[s]')
plt.ylabel('Position[X]')
plt.title('Oscillator Trajectories by cluster (Blue = Cluster 0, Red = Cluster 1')
plt.grid(True)

# Step 7: Scatter plot of features with cluster labels
plt.subplot(1,3,2)
plt.scatter(x_features[:,0],x_features[:,2],
            c=cluster_labels,
            cmap='bwr',
            alpha=0.6
            )
plt.xlabel('Max Amplitude')
plt.ylabel('Decay Rate')
plt.title('Clusters in Feature Space (0 = Blue, 1 = Red)')
plt.grid(True)

# Step 8: Plot true labels for comparison
plt.subplot(1,3,3)
plt.scatter(x_features[:, 0], x_features[:,2], c=true_labels, cmap='bwr', alpha=0.6)
plt.xlabel('Max Amplitude')
plt.ylabel('Decay Rate')
plt.title('True Labels (0 = Underdamped, 1 = Overdamped)')
plt.grid(True)
plt.show()