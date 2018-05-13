import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import time
from scipy.special import expit

#Global Variables
num_intervals = 50

def LinearDecisionBoundary(phi, mu0, mu1, sigma):
	theta = np.dot(np.linalg.inv(sigma), (mu1 - mu0))
	theta0 = np.dot(np.dot(mu0.T, np.linalg.inv(sigma)), mu0) - np.dot(np.dot(mu1.T, np.linalg.inv(sigma)), mu1) + np.log(phi/(1-phi))
	theta = np.vstack((theta0, theta))
	return theta

def linear_boundary(X, theta):
	return -(theta[0] + theta[1]*X)/theta[2]

Xtemp = np.genfromtxt('q4x.dat', delimiter=' ', dtype=float)[:, [0, 2]]
Y = np.loadtxt('q4y.dat', delimiter= ' ', dtype=str)[np.newaxis]
Y = Y.T
m = Xtemp.shape[0]
n = Xtemp.shape[1]

#Normalize data
X_mean = np.mean(Xtemp, axis=0)
X_std = np.std(Xtemp , axis = 0)
X = (Xtemp - X_mean)/X_std

#Create separate X for class1(Canada) and class0(Alaska)
X1 = np.empty(shape= [0, 2]) #Canada
X0 = np.empty(shape= [0, 2]) #Alaska

for i in range(0, m):
	if(Y[i] =='Canada'):
		X1 = np.vstack((X1, X[i]))
	else:
		X0 = np.vstack((X0, X[i]))

#Compute the parameters from closed form solutions of GDA
phi = X1.shape[0]/m

mu1 = np.sum(X1, axis=0)/X1.shape[0]
mu1 = mu1[np.newaxis]
mu1 = mu1.T
mu0 = np.sum(X0, axis=0)/X0.shape[0]
mu0 = mu0[np.newaxis]
mu0 = mu0.T 

sigma1 = np.dot((X1 - mu1.T).T, (X1 - mu1.T))/X1.shape[0]
sigma0 = np.dot((X0 - mu0.T).T, (X0 - mu0.T))/X0.shape[0]
sigma = (sigma1 + sigma0)/2
print ('phi = ', phi)
print ('mu0 = ', mu0)
print ('mu1 = ', mu1)
print ('Sigma = ', sigma)
print ('Sigma0 = ', sigma0)
print ('Sigma1 = ', sigma1)
#Plot data with its linear separator
X1_plot0 = X1[:, 0]
X1_plot1 = X1[:, 1]
X0_plot0 = X0[:, 0]
X0_plot1 = X0[:, 1]
plt.plot(X1_plot0, X1_plot1, '+', label='Canada')
plt.plot(X0_plot0, X0_plot1, 'o', label='Alaska')
theta = LinearDecisionBoundary(phi, mu0, mu1, sigma)
#Plot the boundary
min_X = np.min(X, axis=0)[0]
max_X = np.max(X, axis=0)[0]
X_plot = np.linspace(min_X, max_X, num_intervals)
plt.plot(X_plot, linear_boundary(X_plot, theta))


plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear decision boundary')
plt.legend(loc=1)
plt.show()




