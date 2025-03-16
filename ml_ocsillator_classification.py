import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Simulation parameters
sim_duration = 100  # [s] how long the simulation signals will go
sim_resolution = 0.1 # [s] simulation step size
m = 1.0 # [kg] mass
k = 2.0 # [N/m] Spring constant
noise_factor = 0.01
data_set_size = 500 # The number of examples the AI model has to learn from
test_data_set = 5 # [%] how much of the set will be used for testing the model

#  Step 1: Define oscillator function
def oscillator(state, t, m, b, k):
    x, v = state
    dx_dt = v
    dv_dt = -(b / m)* v - (k / m)* x
    return [dx_dt, dv_dt]

#  Step 2: generate synthetic dataset with varying b
np.random.seed(0) # For reproducibility
t = np.linspace(0 , sim_duration , int(abs(sim_duration / sim_resolution))) # [s] Time array

b_under = np.linspace(0.1, 1.9, int(data_set_size/2)) # under damped b constants
b_over = np.linspace(2.1, 4.0, int(data_set_size/2)) # over damped b constants
b_values = np.concatenate([b_under, b_over]) #  np.concatenate is a function in NumPy, a popular Python library for numerical computing. This function allows you to join arrays along a specified axis.

x = [] #  trajectories
y = [] # Labels (0 = under damped, 1 = over damped)

critical_damping = 2 * np.sqrt(m * k)  # b_critical = 2*sqrt(m*k) = 2.0
for b in b_values:
    state0 = [1.0, 0.0]  # initial position = 1, velocity = 0
    solution = odeint(oscillator, state0, t, args=(m,b,k))
    x.append(solution[:,0])  # Store position directory
    y.append(0 if b < critical_damping else 1) # Label based of damping

x = np.array(x)  #  shape:
x += np.random.randn(*x.shape) * noise_factor
y = np.array(y)

# step 3: feature engineering
max_amplitude = np.max(np.abs(x), axis=1) # find the maximum oscillation
number_of_oscillations = np.sum(np.diff(x, axis=1) > 0, axis=1) # count the number of oscillations in the signal
max_amplitude_at_the_start = np.max(np.abs(x), axis=1)
max_amplitude_half_way = np.max(np.abs(x[:,int(len(t) / 2):]), axis=1)
decay_rate = -np.log(max_amplitude_half_way / max_amplitude_at_the_start) / t[int(len(t) / 2)]
x_features = np.column_stack([max_amplitude, number_of_oscillations, decay_rate])

# Step 4: Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_features, y, test_size=(test_data_set * 0.01), random_state=0)

# Step 5: Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Step6: Predict and evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100 #[%]
print(f'Accuracy: {accuracy:.2f} %')

# Step 7: plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Under damped', 'Overdamped'],
            yticklabels=['Under damped', 'Over damped']
            )
plt.xlabel('predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step: plot example trajectories
plt.figure(figsize=(10,4))
for i in [0, 49, 50, 99]: #plot boundary cases
    label = f'b={b_values[i]:.1f} ({"Under" if y[i] == 0 else "Over"} damped)'
    plt.plot(t, x[i], label=label, alpha=0.7)
plt.xlabel('Time[s]')
plt.ylabel('Position[X]')
plt.title('Example Oscillator Trajectories')
plt.legend()
plt.grid(True)
plt.show()
