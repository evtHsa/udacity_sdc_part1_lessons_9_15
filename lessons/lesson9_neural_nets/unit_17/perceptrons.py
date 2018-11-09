#!/usr/bin/env python3
#
import pdb
import numpy as np
print("eh?")
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)
def stepFunction(t):
   if t >= 0:
      return 1
   return 0

def prediction(X, W, b):
   return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep_mine(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    global cnt
    y_hat = prediction(X, W, b)
 
    #print("FIXME: cnt = %d" % cnt)
    cnt += 1

    for pt_ix in range(len(X)):
        x = X[pt_ix]
        delta = y[pt_ix] - y_hat
        adj = np.array([[X[pt_ix][i] * learn_rate] for i in range(W.shape[0])])
        #print("FIXME: W.shape = " + str(W.shape))
        #print("FIXME: type(W) = " + str(type(W)))
        #print("FIXME: W = " + str(W))
        #print("FIXME: adj.shape = " + str(adj.shape))
        #print("FIXME: type(adj) = " + str(type(adj)))
        #print("FIXME: adj = " + str(adj))
        if delta == 1:
            W += adj
            b += learn_rate
        if delta == -1:
            W -= adj
            b -= learn_rate
    return W, b
    
def perceptronStep_theirs(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    W_m, b_m = perceptronStep_mine(X, y, W, b, learn_rate)
    W_t, b_t = perceptronStep_theirs(X, y, W, b, learn_rate)
    if W_m.all() != W_t.all() or b_m != b_t:
        print("bad juju")
    return W_t, b_t
  
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
   x_min, x_max = min(X.T[0]), max(X.T[0])
   y_min, y_max = min(X.T[1]), max(X.T[1])
   W = np.array(np.random.rand(2,1))
   b = np.random.rand(1)[0] + x_max
   # These are the solution lines that get plotted below.
   boundary_lines = []
   for i in range(num_epochs):
      # In each epoch, we apply the perceptron step.
      W, b = perceptronStep(X, y, W, b, learn_rate)
