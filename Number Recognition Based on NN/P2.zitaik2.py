# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:51:40 2022

@author: Lenovo
Citation: modified from CS440 AI machine problem 2 by Zitai Kong
"""
import math
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
###############################################################################
# helper functions
def find_label(y):
    result = []
    y_list = y.tolist()
    for item in y_list:
        length = len(item)
        max_prob = -math.inf
        max_index = None
        for i in range(length):
            if item[i]> max_prob:
                max_prob = item[i]
                max_index = i
        result.append(max_index)
    return result

def cal_accuracy(y_pred,y_truth):
    # convert y to list
    y = y_truth.tolist()
    # storage
    right_num = 0.
    total = 0.
    for i in range(len(y_pred)):
        total += 1
        if (y_pred[i]==y[i]):
            right_num += 1
    return right_num/total
###############################################################################
# read data in
train_set = scio.loadmat("train_data.mat")
test_set = scio.loadmat("test_data.mat")
train_data = train_set['X']
train_label = train_set['Y'].T
test_data = test_set['X']
test_label = test_set['Y'].T

# convert to tensor
'''
    @param train_set: an (N, out_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
'''
# change from (1,N) to (N,) list
temp_train_label = []
for label in train_label[0]:
    temp_train_label.append(label)
# change from (1,M) to (M,) list
temp_test_label = []
for label in test_label[0]:
    temp_test_label.append(label)
# convert to tensor and set data type
train_data = torch.tensor(train_data,dtype=torch.float32)
train_label = torch.tensor(temp_train_label,dtype=torch.uint8)
test_data = torch.tensor(test_data,dtype=torch.float32)
test_label = torch.tensor(temp_test_label,dtype=torch.uint8)
# record data size
train_num = train_data.shape[0]
test_num = test_data.shape[0]

# Set hyperparameters
in_size = 28*28
hidden_size = 32
out_size = 10
lrate = 0.001
train_batch_size = 256
test_batch_size = 1000
# storage
train_losses3 = []
test_losses3 = []
train_accuracies3 = []
test_accuracies3 = []
domain = []
for i in range(40):
    domain.append(i)
    
# set seed for initialization
torch.manual_seed(42)
random.seed(42)

# run for 3 times
for run in range(3):
    print("******************************************************")
    print("                 Run#",run+1," is running             ")
    print("******************************************************")
    # storage
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    # Define the neural network structure
    model = nn.Sequential(
            nn.Linear(in_size,32),
            nn.ReLU(),
            nn.Linear(32, out_size)
        )
    # initialize weights
    if (run == 1):
        nn.init.normal_(model[0].weight, mean=0.0, std=1.0)
        nn.init.normal_(model[2].weight, mean=0.0, std=1.0)
    if (run == 2):
        nn.init.zeros_(model[0].weight)
        nn.init.zeros_(model[2].weight)
    
    #Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    #Set up the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Train
    for epoch in range(40):
        i = 0
        for batch in range(int(train_num/train_batch_size)):
            # take one step training
            if ( (i+train_batch_size) > train_num ):
                i = 0
        
            x = train_data[i:i+train_batch_size]
            y = train_label[i:i+train_batch_size]
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
        
            # Compute and print loss
            loss = loss_fn(y_pred, y)
            #if t % 100 == 99:
            #    print(t, loss.item())
        
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            L = loss.item()
            
            i = i + train_batch_size
        
        # calculate losses at each epoch
        y_train = model(train_data)
        y_test = model(test_data)
        #i = 0
        #y_test = torch.tensor([],dtype = torch.uint8)
        #for batch in range(int(test_num/test_batch_size)):
        #    if ( (i+test_batch_size) > test_num ):
        #        i = 0
        #    x_test = test_data[i:i+test_batch_size]
        #    y_batch = model(x_test)
        #    y_test = torch.cat((y_test, y_batch), 0)
        train_loss = loss_fn(y_train, train_label)
        train_losses.append(train_loss.item())
        test_loss = loss_fn(y_test, test_label)
        test_losses.append(test_loss.item())
        print("epoch: ",epoch," train_loss: ",train_loss.item()," test_loss: ",test_loss.item())
        # calculate training and testing accuracy
        # first convert prob to label
        train_predict = find_label(y_train)
        test_predict = find_label(y_test)
        train_accuracies.append(cal_accuracy(train_predict, train_label))
        test_accuracies.append(cal_accuracy(test_predict, test_label))
        
    train_losses3.append(train_losses)
    test_losses3.append(test_losses)
    train_accuracies3.append(train_accuracies)
    test_accuracies3.append(test_accuracies)
    
train_accuracy_mean = np.mean(train_accuracies3,axis=0)
train_accuracy_std = np.std(train_accuracies3,axis=0)
test_accuracy_mean = np.mean(test_accuracies3,axis=0)
test_accuracy_std = np.std(test_accuracies3,axis=0)
    
print("******************************************************")
print("                 Results are printing                 ")
print("******************************************************")
# plot the training and testing loss curves w.r.t. the training epochs
# run 1
loss1, = plt.plot(train_losses3[0])
loss2, = plt.plot(test_losses3[0])
plt.ylabel('loss')
plt.xlabel('training epoch')
plt.title("training and testing loss w.r.t. the training epochs, init = random")
plt.legend(handles=[loss1,loss2],labels=['training loss','testing loss'],loc='best') 
# run 2
plt.figure()
loss1, = plt.plot(train_losses3[1])
loss2, = plt.plot(test_losses3[1])
plt.ylabel('loss')
plt.xlabel('training epoch')
plt.title("training and testing loss w.r.t. the training epochs, init = normal")
plt.legend(handles=[loss1,loss2],labels=['training loss','testing loss'],loc='best') 
# run 3
plt.figure()
loss1, = plt.plot(train_losses3[2])
loss2, = plt.plot(test_losses3[2])
plt.ylabel('loss')
plt.xlabel('training epoch')
plt.title("training and testing loss w.r.t. the training epochs, init = zero")
plt.legend(handles=[loss1,loss2],labels=['training loss','testing loss'],loc='best') 
# all
plt.figure()
loss1, = plt.plot(train_losses3[0])
loss2, = plt.plot(train_losses3[1])
loss3, = plt.plot(train_losses3[2])
plt.ylabel('loss')
plt.xlabel('training epoch')
plt.title("training loss w.r.t. the training epochs")
plt.legend(handles=[loss1,loss2,loss3],labels=['random init','normal init','zero init'],loc='best') 

plt.figure()
loss1, = plt.plot(test_losses3[0])
loss2, = plt.plot(test_losses3[1])
loss3, = plt.plot(test_losses3[2])
plt.ylabel('loss')
plt.xlabel('training epoch')
plt.title("testing loss w.r.t. the training epochs")
plt.legend(handles=[loss1,loss2,loss3],labels=['random init','normal init','zero init'],loc='best') 

# plot the training and testing accuracy w.r.t. the training epochs
plt.figure()
plt.errorbar(domain, train_accuracy_mean, train_accuracy_std,ms=5, capsize=2,label ='training accuracy')
plt.errorbar(domain, test_accuracy_mean, test_accuracy_std,ms=5, capsize=2,label ='testing accuracy')
plt.ylabel('accuracy')
plt.xlabel('training epoch')
plt.title("training and testing accuracy w.r.t. the training epochs, all")
plt.legend(loc='best')
# run 0
plt.figure()
accuracy1, = plt.plot(train_accuracies3[0])
accuracy2, = plt.plot(test_accuracies3[0])
plt.ylabel('accuracy')
plt.xlabel('training epoch')
plt.title("training and testing accuracy w.r.t. the training epochs, init = random")
plt.legend(handles=[accuracy1,accuracy2],labels=['training accuracy','testing accuracy'],loc='best')  
# run 1
plt.figure()
accuracy1, = plt.plot(train_accuracies3[1])
accuracy2, = plt.plot(test_accuracies3[1])
plt.ylabel('accuracy')
plt.xlabel('training epoch')
plt.title("training and testing accuracy w.r.t. the training epochs, init = normal")
plt.legend(handles=[accuracy1,accuracy2],labels=['training accuracy','testing accuracy'],loc='best')    
# run 2
plt.figure()
accuracy1, = plt.plot(train_accuracies3[2])
accuracy2, = plt.plot(test_accuracies3[2])
plt.ylabel('accuracy')
plt.xlabel('training epoch')
plt.title("training and testing accuracy w.r.t. the training epochs, init = zero")
plt.legend(handles=[accuracy1,accuracy2],labels=['training accuracy','testing accuracy'],loc='best')   
# all
plt.figure()
accuracy1, = plt.plot(train_accuracies3[0])
accuracy2, = plt.plot(train_accuracies3[1])
accuracy3, = plt.plot(train_accuracies3[2])
plt.ylabel('accuracy')
plt.xlabel('training epoch')
plt.title("training accuracy w.r.t. the training epochs")
plt.legend(handles=[accuracy1,loss2,loss3],labels=['random init','normal init','zero init'],loc='best') 

plt.figure()
accuracy1, = plt.plot(test_accuracies3[0])
accuracy2, = plt.plot(test_accuracies3[1])
accuracy3, = plt.plot(test_accuracies3[2])
plt.ylabel('accuracy')
plt.xlabel('training epoch')
plt.title("testing accuracy w.r.t. the training epochs")
plt.legend(handles=[accuracy1,accuracy2,accuracy3],labels=['random init','normal init','zero init'],loc='best') 

print(train_accuracy_mean)
print(train_accuracy_std)
print(test_accuracy_mean)
print(test_accuracy_std)
print(np.mean(train_accuracies3,axis=1))
print(np.mean(test_accuracies3,axis=1))
print(np.std(train_accuracies3,axis=1))
print(np.std(test_accuracies3,axis=1))
       
# ... after which, you should save it as "model.pkl":
torch.save(model, 'model.pkl')        

