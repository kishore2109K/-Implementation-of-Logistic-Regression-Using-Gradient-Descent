# Implementation-of-Logistic-Regression-Using-Gradient-Descent
# Developed by: KISHORE K

# RegisterNumber: 212223040101
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```
## Output:
Dataset
![image](https://github.com/user-attachments/assets/ffbdc3b5-e6c4-4be0-a422-bc5b4b65215b)

![image](https://github.com/user-attachments/assets/1fec1437-641b-49fc-a115-17c320512034)

![image](https://github.com/user-attachments/assets/3ee15d56-a9c3-4b68-b8d7-ee91a6caea17)

![image](https://github.com/user-attachments/assets/74669292-c6ec-4789-98fa-70d437bda7f5)

Accuracy and Predicted Values
![image](https://github.com/user-attachments/assets/2515170e-dc63-47cf-8efc-03b6180061e9)

![image](https://github.com/user-attachments/assets/3ac88448-3dc6-441b-ba16-a5d475184de7)

![image](https://github.com/user-attachments/assets/45502ea6-d8aa-48ca-b51a-f9b878820424)

![image](https://github.com/user-attachments/assets/fc478759-920c-4725-9846-e7f3f9d03764)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

