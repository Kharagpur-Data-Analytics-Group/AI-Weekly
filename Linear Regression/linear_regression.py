################################ Start of Code ################################

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data file
csv_path = 'hwdata.csv'
df = pd.read_csv(csv_path)

# Split into train and test set
df_train = df.sample(frac=0.5, random_state=0)  
df_test = df.drop(df_train.index)  

# Getting feature and label columns
x_train = df_train['height'][:100]
y_train = df_train['weight'][:100]
x_test = df_test['height']
y_test = df_test['weight']

# reshapinng 
x_train = x_train.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

# Plot the data points
plt.scatter(x_train, y_train, s=5.0, c='g')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot')
plt.show()

n     = 100   # total data points
alpha = 0.0001  # learning rate 

# we want to fit y = slope*x + intercept
intercept = np.ones((n,1))*np.random.randn()
slope     = np.ones((n,1))*np.random.randn()
iteration = 0  

while(iteration < 1000):
    y = intercept  + slope * x_train
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    intercept = intercept - alpha * 2 * np.sum(error)/n 
    slope  = slope  - alpha * 2 * np.sum(error * x_train)/n
    iteration += 1

print("Mean square error : ",mean_sq_er)

axes = plt.gca()
x_vals = np.array(x_train)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
plt.scatter(x_train, y_train, s=5.0, c='g')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot')
plt.show()

# we want to fit y = slope*x + intercept
n     = 100   # total data points
alpha = 1e-8  # learning rate 
intercept = np.ones((n,1))*np.random.randn()
slope_1     = np.zeros((n,1))*np.random.randn()
slope_2 = np.zeros((n, 1))*np.random.randn()
iteration = 0  

while(iteration < 1000):
    y = intercept  + slope_1 * x_train + slope_2 * (x_train**2)
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    intercept = intercept - alpha * 2 * np.sum(error)/n 
    slope_1  = slope_1  - alpha * 2 * np.sum(error * x_train)/n
    slope_2  = slope_2  - alpha * 2 * np.sum(error * (x_train**2))/n
    iteration += 1

print("Mean square error : ",mean_sq_er)

axes = plt.gca()
x_vals = np.array(x_train)
y_vals = intercept + slope_1 * x_vals + slope_2 * (x_vals**2)
plt.scatter(x_vals, y_vals, c='b')
plt.scatter(x_train, y_train, s=1.0, c='g')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot')
plt.show()

################################### End of Code ################################
