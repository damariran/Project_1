from statistics import linear_regression

import numpy as np
from simple_functions import simple_plot, simple_scatter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# simulation parameters
# linear fit
a = 3
b = 2
noise_factor = 2

#  Step 1: Generate synthetic data
np.random.seed(0) #  The seed function gives the random function the same starting point. This means that the random function will always generate the same series of random number for us.
x = np.linspace(0,10,100)  # Feature (e.g., time)
y = a * x + b + np.random.randn(100) * noise_factor # Target (linear line with noise)
x = x.reshape(-1, 1) #  reshapes the row vector into a (10,1) column vector

#Step 2: Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=0)

# Step 3: fit linear regression model
model = LinearRegression() # an instance of the LinearRegression() class in created in 'model'
model.fit(x_train, y_train) # This class has a fit option. It fit the best linear line (model) to the training data
y_fit = model.predict(x)

# Step 4: Make predictions (this is the fitted line!!!)
y_predictions = model.predict(x_test)

# step 5: Evaluate model
mse = mean_squared_error(y_test, y_predictions)
r2 = r2_score(y_test,y_predictions)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score:{r2:.2f}')

simple_scatter(x_train, y_train,
               my_legend='Training data',
               my_x_label='Feature [X]',
               my_y_label='Target [Y]',
               my_title='example of linear regression',
               close=False)
simple_scatter(x_test, y_test,
               my_legend='Test data' ,
               close=False)
simple_plot(x_test, y_predictions,
            my_legend='fit',
            my_text='mse')