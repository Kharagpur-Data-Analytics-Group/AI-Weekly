"""
Original file is located at
    https://colab.research.google.com/drive/1pua-eMhyKNYImVpxUjQ8Tdg7VHzydmqp
"""

#Importing the required modules
import numpy as np
from scipy.stats import mode

import math

# Euclidean Distance
def eucledian(x,y):
    dist = math.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))
    return dist

# Manhattan distance 
def manhattan(x,y):
    dist = sum(abs(a-b) for a,b in zip(x,y))
    return dist

# Chebyshev distance
def Chebyshev(x,y):
    dist=max(abs(a-b) for a,b in zip(x,y))    
    return dist

# Minkowski distance
def minkowski(x,y,p_value):
  return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x,y)),p_value)

def nth_root(value,n_root):
  root_value = 1/float(n_root)
  return round(Decimal(value)** Decimal(root_value),3)       

# Cosine similarity
def cosine(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def square_rooted(x):
  return round(math.sqrt(sum([a*a for a in x])),3)

#Function to calculate KNN
def predict(x_train, y , x_input, k):
    op_labels = []
     
    #Loop through the Datapoints to be classified
    for item in x_input: 
         
        #Array to store distances
        point_dist = []
         
        #Loop through each training Data
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            #Calculating the distance
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        #Sorting the array while preserving the index
        #Keeping the first K datapoints
        dist = np.argsort(point_dist)[:k] 
         
        #Labels of the K datapoints from above
        labels = y[dist]
         
        #Majority voting
        lab = mode(labels) 
        lab = lab.mode[0]
        op_labels.append(lab)
 
    return op_labels

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Now it’s time to test our implementation on some data.

#Importing the required modules
#Importing required modules

from sklearn.datasets import load_iris
from numpy.random import randint
 
#Loading the Data
iris= load_iris()
 
# Store features matrix in X
X= iris.data
#Store target vector in 
y= iris.target
 
#Creating the training Data
train_idx = randint(0,150,100)
X_train = X[train_idx]
y_train = y[train_idx]
 
#Creating the testing Data
test_idx = randint(0,150,50) #taking 50 random samples
X_test = X[test_idx]
y_test = y[test_idx]
 
#Applying our function 
y_pred = predict(X_train,y_train,X_test , 7)
 
#Checking the accuracy
accuracy_metric(y_test, y_pred)

