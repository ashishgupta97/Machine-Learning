
# coding: utf-8

# In[3]:


import numpy as np
import csv
import math
import svmutil as svm
import matplotlib.pyplot as plt
from svm import *
#global variables
batch_size =100
num_iterations = 4000


# In[4]:


def pegasos(x_low, x_high):
    w = np.zeros(784)
    b = 0
    i = 1
    while(i<num_iterations):
        random_num = np.random.randint(batch_size)
        low_indices = np.random.choice(len(x_low), random_num, replace=False)
        high_indices = np.random.choice(len(x_high), (batch_size-random_num), replace=False)
        temp_w = [0]*784
        temp_b = 0
        #low
        temp_vector = np.zeros(784)
        temp = 0
        for j in range(random_num):
            check = False
            if (-1)*(np.dot(x_low[low_indices[j]], w)+b)<1:
                check = True
            if(check):
                temp_vector = temp_vector + (-1)*x_low[low_indices[j]]
                temp+=-1
        
        for j in range((batch_size-random_num)):
            check = False
            if((np.dot(x_high[high_indices[j]], w)+b)<1):
                check = True;
            if(check):

                temp_vector = temp_vector + x_high[high_indices[j]]
                temp+=1

        alpha = 1/i
        w = w*(1-alpha) + (alpha*temp_vector)/batch_size
        b = b + alpha*temp/batch_size
        i+=1
    return w, b


# In[5]:


data = np.loadtxt('train.csv', delimiter=',', dtype=int)
x_cols = data.shape[1]-1
X_temp = data[:, 0:x_cols]
Y = data[:, x_cols:x_cols+1]


# In[6]:


data_test = np.loadtxt('test.csv', delimiter=',', dtype=int)
x_cols_test = data_test.shape[1]-1
X_temp_test = data_test[:, 0:x_cols_test]
Y_test = data_test[:, x_cols_test:x_cols_test+1]


# In[7]:


X_temp.astype(float)
X = X_temp/255
X_temp_test.astype(float)
X_test = X_temp_test/255
#w, b = pegasos(X, Y, batch_size, num_iterations)


# In[8]:


#separate the data into buckets w.r.t their labels
x_classes = [[] for i in range(10)] 
for i in range (X.shape[0]):
    x_classes[Y[i][0]].append(X[i])
#parameter_w = [[]*10 for i in range(10)]
#parameter_b = [[]*10 for i in range(10)]
#print(parameter_w)


# In[9]:


lookup = []
for i in range(10):
    for j in range(i+1, 10):
        lookup.append(((i, j), (pegasos(x_classes[i], x_classes[j]))))


# In[10]:


#train accuracy
y_train_pred = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    count = np.zeros(10)
    for classifier in lookup:
        w = classifier[1][0]
        b = classifier[1][1]
        if(np.dot(X[i], w) + b>=0):
            count[classifier[0][1]]+=1
        else:
            count[classifier[0][0]]+=1
    y_train_pred[i] = np.argmax(count)


# In[11]:


total = Y.shape[0]
correct=0
for i in range(X.shape[0]):
        if(Y[i][0]==y_train_pred[i]):
            correct+=1
print('accuracy is', correct/total*100)


# In[24]:


#test accuracy
y_test_pred = np.zeros(X_test.shape[0])
for i in range(X_test.shape[0]):
    count = np.zeros(10)
    for classifier in lookup:
        w = classifier[1][0]
        b = classifier[1][1]
        if(np.dot(X_test[i], w) + b>=0):
            count[classifier[0][1]]+=1
        else:
            count[classifier[0][0]]+=1
    y_test_pred[i] = np.argmax(count)

    


# In[26]:


total = Y_test.shape[0]
correct=0
for i in range(X_test.shape[0]):
        if(Y_test[i][0]==y_test_pred[i]):
            correct+=1
print('accuracy is', correct/total*100)


# In[5]:


#LIBSVM Model
train_labels = []
param = []
for i in range(0, Y.shape[0]):
    train_labels.append(Y[i][0])
for i in range(0, X.shape[0]):
    param.append(X[i].tolist())


test_labels = []
param_test = []
for i in range(0, Y_test.shape[0]):
    test_labels.append(Y_test[i][0])
for i in range(0, X_test.shape[0]):
    param_test.append(X_test[i].tolist())


# In[6]:


linear_model = svm.svm_train(train_labels, param, '-t 0 -c 1.0')
p_labs, p_acc, p_vals = svm.svm_predict(test_labels, param_test, linear_model)


# In[7]:


gaussian_model = svm.svm_train(train_labels, param, '-g 0.05 -c 1.0')
gp_labs, gp_acc, gp_vals = svm.svm_predict(test_labels, param_test, gaussian_model)


# In[13]:


C = [0.00001, 0.001, 1, 5, 10]
cv_accuracy = np.zeros(5)
test_accuracy = np.zeros(5)
for i in range(5):
    model = svm.svm_train(train_labels, param, '-g 0.05 -c '+str(C[3]))
    [labels, accuracy, decisions] = svm.svm_predict(test_labels, param_test, model)
    test_accuracy[3] = accuracy[1]
    cv_accuracy[i] = svm.svm_train(train_labels, param, '-g 0.05 -v 10 -c '+str(C[i]))


# In[20]:


test_accuracy[0] = 72.1
test_accuracy[1] = 72.1
test_accuracy[2] = 97.23
test_accuracy[3] = 97.29
test_accuracy[4] = 97.29
plt.plot(np.log10(C), test_accuracy)
plt.show()


# In[38]:


p_labs = list(map(int, p_labs))
confusion_matrix = np.zeros((10, 10), dtype=int)
for i in range(Y_test.shape[0]):
    confusion_matrix[Y_test[i][0]][p_labs[i]]+=1
print(confusion_matrix)

