# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:06:19 2022

@author: Lenovo
"""
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

###############################################################################
# Helper functions
        
# sigmoid function
def sigmoid(z):
    
    sig = 1.0/(1.0 + np.exp(-1.0 * z))
    
    return sig

# accuracy
def cal_accuracy(w,x,y):
    size = y.shape[0]
    correct = 0.
    z = np.matmul(w.T,x)
    y_pred = sigmoid(z).T
    
    for i in range(size):
        if(y_pred[i] > 0.5):
            Y = 1
        else:
            Y = 0
        if (Y == y[i]):
            correct = correct + 1
    return correct/size
            
# print training loss vs. the number of iterations
def plot_loss(loss):
    plt.plot(loss)
    plt.ylabel('training loss')
    plt.xlabel('iteration')
    plt.title("training loss vs. the number of iterations")        
    
# print the test accuracy vs. the number of iterations
def plot_accuracy(accuracy):
    plt.figure()
    plt.plot(accuracy)
    plt.ylabel('test accuracy')
    plt.xlabel('iteration')
    plt.title("the test accuracy vs. the number of iterations")  
##############################################################################
# read data in
train_set = scio.loadmat("train_data.mat")
test_set = scio.loadmat("test_data.mat")
train_data = train_set['X']
train_label = train_set['Y']
test_data = test_set['X']
test_label = test_set['Y']

# initialization
w = np.zeros(shape=(784, 1), dtype=np.float32)
lrate = 0.1
x = train_data.T
y = train_label.T
m = 60000

# set result holder
accuracy = []
loss = []

for t in range(100):
    # forward pass: compute predicted y
    z= np.matmul(w.T, x)
    y_pred = sigmoid(z)
        
    # compute and print loss
    # loss is the average of their negative log-likelihood values
    cost = -np.mean(np.multiply(y, np.log(y_pred)) + np.multiply(1.0-y, np.log(1.0 - y_pred)), axis=1)
    loss.append(cost)
    #print("iteration: ",t, "    loss:",cost)
        
    # back-propagate to compute gradients
    dw = np.matmul(x, (y_pred - y).T)/m
    # take gradient descent algorithm
    w = w - lrate * dw
        
    # predict accuracy
    accuracy_piece = cal_accuracy(w,test_data.T,test_label)
    accuracy.append(accuracy_piece)
    print("iteration: ",t+1, "    loss:",cost,"    accuracy:",accuracy_piece)
    
# print images
plot_loss(loss)
plot_accuracy(accuracy)

        
    

