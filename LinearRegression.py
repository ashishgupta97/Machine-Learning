import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
#Global Variables
num_iter=0
saved_theta=np.zeros((2, 1))
cost=np.zeros(1)
def h(X, theta):
	return np.dot(X, theta)

def Jtheta(X, Y, theta):
	return 1/2.0*np.sum((h(X, theta)-Y)**2)

def linear_prediction(x, theta):
	return theta[0] + theta[1]*x

def normal(X, Y):
	return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

def gradient_descent(X, Y, alpha, theta_in):
	i=0
	epsilon=1e-15
	theta=theta_in;
	global saved_theta
	global num_iter
	global cost
	saved_theta=theta_in
	
	J=Jtheta(X, Y, theta)
	cost=J
	while (True):
		Jprev=Jtheta(X, Y, theta)
		theta_new= theta + (alpha)*np.dot(X.T, (Y-h(X, theta)))
		Jnew=Jtheta(X, Y, theta_new)
		if math.fabs(Jnew-Jprev)<epsilon:
			break
		theta=theta_new
		saved_theta=np.hstack((saved_theta, theta))
		cost=np.vstack((cost, Jnew))
		i=i+1
	num_iter=i
	print('Number of iterations', i)
	return theta


Xtemp = np.loadtxt('linearX.csv', delimiter=',')				#array of X
ytemp = np.loadtxt('linearY.csv', delimiter=',')[np.newaxis]	#converting 1d array into 2d matrix using np.newaxis
ones = np.ones(len(Xtemp))

#Normalizing the data
mean = np.mean(Xtemp)
sigma = np.std(Xtemp)
Xtemp = (Xtemp - mean)/sigma

Xtemp1 = np.vstack((ones, Xtemp))
X=Xtemp1.T.copy()												#taking transpose of X
Y=ytemp.T.copy()												#taking transpose of Y
alpha=0.0001							
theta=[[0.], [0.]]

#part a
theta_optimal=gradient_descent(X, Y, alpha, theta)
print('Optimal value of theta', theta_optimal)
print('Analytical solution is', normal(X, Y))


#part b
plt.plot(Xtemp, Y, 'ro')
plt.plot(Xtemp, np.dot(X, theta_optimal))
plt.xlabel('Aciditiy')
plt.ylabel('Density')
plt.show()

def createJ_plot(Theta_0, Theta_1):
	Theta = np.matrix([[Theta_0], [Theta_1]])
	return ((Y - X * Theta).T * (Y - X * Theta) / (2*X.shape[0])).item(0)

#part c
#3D mesh
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta_0_plot=np.arange(-4, 4, 0.05)[np.newaxis]
theta_1_plot=np.arange(-1, 1, 0.002)[np.newaxis]  #Make it 4
theta_0_plot, theta_1_plot=np.meshgrid(theta_0_plot, theta_1_plot)

J_plot=np.vectorize(createJ_plot)(theta_0_plot, theta_1_plot)
ax.plot_surface(theta_0_plot, theta_1_plot, J_plot, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
plt.show()
