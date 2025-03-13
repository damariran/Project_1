import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#  Step 1: Generate synthetic data
np.random.seed(0) #  The seed function gives the random function the same starting point. This means that the random function will always generate the same series of random number for us.
X = np.linspace(0,10,100)  # Feature (e.g., time)
print(X.shape)
X = X.reshape(-1, 1)
print(X.shape)