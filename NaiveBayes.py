
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import string
import random
from random import randint
from collections import Counter
#Global Variables


# In[3]:


def readFile(fileName):
    fileRead =[]
    file_reader = open(fileName)
    i = 0
    for line in file_reader:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        l = [x for x in line.translate(translator).strip().split()]
        fileRead.append(l)
    return fileRead
        
    


# In[4]:


def removeDuplicates(X):
    temp = [] #Extend temp to contain words in all the reviews
    for review in X:
        temp.extend(review)
    all_words = list(set(temp))
    return all_words


# In[79]:


def train_Naive_Bayes(X_train, Y_train, word_positions):
    y = len(word_positions)
    lookup = [[0]* y for i in range(11)]
    for i1 in range (0, len(X_train)):
        i = Y_train[i1][0]
        for i2 in range(0, len(X_train[i1])):
            j = word_positions.get(X_train[i1][i2])
            lookup[i][j]+=1
    #Esmtimating the parameters of the model
    phi = [0]*11 
    for i in range(1, 11):
        phi[i] = (np.count_nonzero(Y_train==i) + 1)/(Y_train.shape[0]+10)
    theta = [[0]* y for i in range(11)] 
    num_words_i_class = [0]*11
    for i in range(1, 11):
        num_words_i_class[i] = sum(lookup[i])
    for i in range(1, 11):
        for j in range(0, y):
            theta[i][j] = (lookup[i][j] + 1)/(num_words_i_class[i] + y)
    return phi, theta
    


# In[17]:


def predict_class(phi, theta, x, word_positions):
    prob = [1]*11
    for i in range(1, 11):
        prob[i] = np.log10(np.abs(phi[i]))
        for j in range(0, len(x)):
            if x[j] in word_positions:
                prob[i]+= np.log10(np.abs(theta[i][word_positions.get(x[j])]))
            else:
                prob[i]+=np.log10(1/len(word_positions))
    return np.argmax(prob[1:11])+1
    
    
    
                
    


# In[18]:


#Reading files
Y_test = np.loadtxt('imdb_test_labels.txt')[np.newaxis]
Y_train = np.loadtxt('imdb_train_labels.txt')[np.newaxis]
Y_test = Y_test.T
Y_train = Y_train.T
X_test = readFile('imdb_test_text2.txt')
X_train = readFile('imdb_train_text2.txt')


# In[20]:


Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)
#remove duplicates to create the dictionary
all_train_words = removeDuplicates(X_train)
all_test_words = removeDuplicates(X_test)


# In[21]:


#Mapping words to their positions in a dictionary
word_positions = dict(zip(all_train_words, np.arange(0, len(all_train_words))))
print('size of dictionary is, ', len(word_positions))


# In[80]:


phi, theta = train_Naive_Bayes(X_train, Y_train, word_positions)


# In[81]:


#Training data accuracy
correct = 0
total = Y_train.shape[0]
for i in range(0, total):
    y_predicted = predict_class(phi, theta, X_train[i], word_positions)
    if(y_predicted == Y_train[i]):
        correct+=1
accuracy = (correct/total)*100
print('Training data accuracy is', accuracy)





# In[68]:


#Test set accuracy
correct = 0
total = Y_test.shape[0]
confusion_matrix = [[0] * 11 for i in range(11)]
for i in range(0, total):
    y_predicted = predict_class(phi, theta, X_test[i], word_positions)
    if(y_predicted == Y_test[i]):
        correct+=1
    pred = y_predicted
    true = Y_test[i][0]
    confusion_matrix[pred][true]=confusion_matrix[pred][true]+1
accuracy = (correct/total)*100
print('Test data accuracy is', accuracy)
print('Confusion Matrix \n', confusion_matrix)


# In[73]:


#Random prediction accuracy
correct = 0
total = Y_test.shape[0]
for i in range(0, total):
    y_predicted = randint(1,10)
    if(y_predicted == Y_test[i]):
        correct+=1
accuracy = (correct/total)*100
print('Random prediction accuracy is', accuracy)



# In[70]:


#Max ocuurences prediction accuracy
counts = np.bincount(Y_train[:, 0])
freq_class = np.argmax(counts)
correct = 0
total = Y_test.shape[0]
for i in range(0, total):
    y_predicted = freq_class
    if(y_predicted == Y_test[i]):
        correct+=1
accuracy = (correct/total)*100
print('Max occurence prediction accuracy is', accuracy)


