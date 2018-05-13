import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import time
from scipy.special import expit
#global variables
epsilon = 1e-20
num_points = 50

def boundary(X, theta):
	return -(theta[0] + theta[1]* X)/theta[2]

def sigmoid(X, theta):
	return 1.0/(1.0 + np.exp(-1.0 * np.dot(X, theta)))


def Jtheta(X, Y, theta):
	return -(np.dot(Y.T, np.log(sigmoid(X, theta))) + np.dot((1-Y).T, np.log(1 - sigmoid(X, theta))))/2

def gradJ(X, Y, theta):
	return np.dot(X.T, (Y - sigmoid(X, theta)))

def HessianJ(X, Y, theta):
	Derivative_Matrix = np.diagflat(np.multiply(sigmoid(X, theta), 1-sigmoid(X, theta)))
	return -(np.dot(np.dot(X.T, Derivative_Matrix), X))


def newton(X, Y, theta_intial):
	theta = theta_intial
	iteration = 0
	Jold = Jtheta(X, Y, theta)
	while(True):
		new_theta = theta - np.dot(np.linalg.inv(HessianJ(X, Y, theta)), gradJ(X, Y, theta))
		iteration+=1
		Jnew = Jtheta(X, Y, new_theta)
		if np.fabs(Jnew - Jold) < epsilon:
			break
		theta = new_theta
		Jold = Jnew
	return new_theta



#load the matrices
Xtemp= np.loadtxt('logisticX.csv', delimiter=',')
Ytemp = np.loadtxt('logisticY.csv', delimiter=',')[np.newaxis]
Y = Ytemp.T

#normalizing the data
mean_X = Xtemp.mean(axis=0)
std_X = Xtemp.mean(axis=0)
Xtemp[:, 0] = (Xtemp[:, 0] - mean_X[0])/std_X[0]
Xtemp[:, 1] = (Xtemp[:, 1] - mean_X[1])/std_X[1]

m = Xtemp.shape[0]
n = Xtemp.shape[1] + 1
#appending ones to Xtemp matrix
X = np.hstack((np.ones((m, 1)), Xtemp))
theta_intial = np.zeros((n, 1))

theta_optimal = newton(-X, Y, theta_intial)
print('Optimal Value of theta is \n', theta_optimal)

#Plotting the data
prob1_matrix = sigmoid(X, theta_optimal)
positive_indices = np.argwhere(prob1_matrix >= 0.5)
zero_indices = np.argwhere(prob1_matrix < 0.5)
X1 = np.empty(shape=[0, 2])
X0 = np.empty(shape=[0, 2])
for row in positive_indices:
	X1 = np.vstack((X1, Xtemp[row[0]]))
for row in zero_indices:
	X0 = np.vstack((X0, Xtemp[row[0]]))

X1_plot_0 = X1[:, 0]
X1_plot_1 = X1[:, 1]
X0_plot_0 = X0[:, 0]
X0_plot_1 = X0[:, 1]
plt.plot(X1_plot_0, X1_plot_1, '+')
plt.plot(X0_plot_0, X0_plot_1, 'o')
prediction = np.zeros((m, 1))
prediction[positive_indices] = 1

#Boundary
min_X = np.min(Xtemp)
max_X = np.max(Xtemp)
X_plot =np.linspace(min_X, max_X, num_points)
X1 = Xtemp[:, 0]
X2 = Xtemp[:, 1]
plt.plot(X_plot, boundary(X_plot, theta_optimal))

#Labels
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic regression')
plt.legend()
plt.show()

