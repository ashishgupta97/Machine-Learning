import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import time

#Global variables
tau=10
num_points = 100

def Jtheta(X, W, Y, theta):
	return ((Y- X*theta).T * W * (Y - X*theta))/2

def grad_Jtheta(X, W, Y, theta)
	return 

def plot_learned_hypothesis(theta, Xtemp, X, Y):
	plt.plot(Xtemp, Y, '*', markersize=2)
	plt.plot(Xtemp, np.dot(X, theta), markersize=5)
	plt.show()

def predictor(x_predic, theta):
	return theta[0] + theta[1]*x_predic


def unweighted_normal(X, Y):
	return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))


def weighted_normal(W, X, Y):
	return np.dot(np.linalg.inv(np.dot(np.dot(X.T, W), X)), np.dot(np.dot(X.T, W), Y))

def create_W(X, curr_x, tau):
	#create m*m  matrix
	m = X.shape[0]
	W = np.zeros((m, m))
	for i in range(0, m):
		W[i][i] = math.exp(-((X[i] - curr_x)**2)/(2 * (tau**2)))
	return W


Xtemp = np.loadtxt('weightedX.csv', delimiter=',')[np.newaxis]
Ytemp = np.loadtxt('weightedY.csv', delimiter=',')[np.newaxis]	
Y = Ytemp.T
m = Xtemp.shape[0]
#Normalizing the data
mean = np.mean(Xtemp)
sigma = np.std(Xtemp)
Xtemp = (Xtemp - mean)/sigma
Xtemp = Xtemp.T

#Append ones to X
X=np.hstack((np.ones((Xtemp.shape[0], 1)), Xtemp))

#Part a 
analytical_theta=unweighted_normal(X, Y)
print('Analytical Solution is ', analytical_theta)

#Plot the learned hypothesis line and data
plot_learned_hypothesis(analytical_theta, Xtemp, X, Y)


#Part b
#construction of the diagonal weight matrix on the dataset and plotting 
plt.plot(Xtemp, Y, '*', markersize=2)

min_X = Xtemp.min()
max_X = Xtemp.max()
X_plot = np.linspace(min_X, max_X, num=num_points)
prediction = np.zeros((num_points, 1))
for i in range(0, num_points):
	W = create_W(Xtemp, X_plot[i], tau)
	theta_for_x = weighted_normal(W, X, Y)
	prediction[i] = theta_for_x[0] + theta_for_x[1]*X_plot[i]

plt.plot(X_plot, prediction, markersize=2)
plt.ylabel('X')
plt.xlabel('Y')
plt.title('Data and Learned curve')
plt.show() 
print(prediction)
