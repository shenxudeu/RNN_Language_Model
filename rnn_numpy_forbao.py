'''
Simple RNN in Numpy.
model structure:
    s_t = tanh(Ux_t + Ws_{t-1})
    o_t = softmax(Vs_t)

Derived from Denny Britz's NLP code https://github.com/dennybritz/rnn-tutorial-rnnlm.
'''

import os
import sys
import numpy as np
from IPython import embed
import operator
import cPickle
import timeit
import datetime as dttm
from reddit_comment_data import get_data
from utils import *

class RNN:
    def __init__(self, word_dim, hidden_dim, bptt_truncate=4):
        '''
        RNN Model
        inputs
        ------
            word_dim: int, size of input
            hidden_dim: int, size of hidden layer
            bptt_truncate: int

        weights
        ------
            U: np.array, (hidden_dim, word_dim)
            W: np.array, (hidden_dim, hidden_dim)
            V: np.array, (word_dim,hidden_dim)
        '''
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize weights
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim), (word_dim,hidden_dim))

    def forward_propagation(self, x):
        '''
        inputs
        ------
            x: np.array, (batch_size, word_dim)

        outputs
        ------
            s: np.array, (batch_size, hidden_dim)
            o: np.array, (batch_size, word_dim)

        Note: batch_size is also the number of time steps
        '''
        T = len(x)
        assert (x.shape[1] == self.word_dim), 'forward_propagation, input shape not match network size'
        assert (x.sum() == T), 'forward_propagation, input must be a one-hot matrix'
        
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T+1, self.hidden_dim))
        spx = np.zeros((T+1, self.hidden_dim)) # spx_t = W.S_{t-1} + U.X_t
        p = np.zeros((T, self.word_dim))
        o = np.zeros((T, self.word_dim))
        # For each time step ...
        for t in np.arange(T):
            spx[t] = self.U.dot(x[t]) + self.W.dot(s[t-1])
            s[t] = np.tanh(spx[t])
            o[t] = self.V.dot(s[t])
            p[t] = softmax(o[t])
        return [o, p, s, spx]

    def predict(self, x):
        '''Perform forward propagation and return index of the highest score'''
        o, p, s, spx = self.forward_propagation(x)
        return p.argmax(p, axis=1)

    def compute_total_loss(self, x, y):
        L = 0.
        # Forward-propagation for each sequence
        for i in range(len(y)):
            o, p, s, spx = self.forward_propagation(one_hot(x[i],self.word_dim))
            correct_predictions = p[np.arange(len(y[i])),y[i]]
            # Add to the loss L
            L += -1. * np.sum(np.log(correct_predictions))
        return L

    def get_loss(self, x, y):
        '''cross-entropy loss L = -1/N \sum_{n=0-N}{y_n \log{o_n}}'''
        L = self.compute_total_loss(x, y)
        # Divide the total loss by the number of training examples
        N = np.sum(len(y[i]) for i in range(len(y)))
        return L / N

    def bptt(self, x, y):
        '''
        Back-propagation through time
        gradient gates:
        dO = P - Y # given P = cross_entropy(Y, softmax(O))
        dspx = 1 - s**2 # given S = tanh(spx)
        '''
        T = len(x)
        ###### FORWARD PASS
        o, p, s, spx = self.forward_propagation(one_hot(x, self.word_dim))
        x = one_hot(x, self.word_dim)

        # Initialize the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        
        ###### BACKWARD PASS
        dLdST_ = np.zeros(self.hidden_dim)
        for t in reversed(range(T)):
            # time T
            dLdPT = np.zeros((self.word_dim))
            dLdPT[y[t]] = -1./p[t][y[t]]           # (w,)
            dPTdOT = -np.outer(p[t],p[t]) * (1. - np.identity(p.shape[1])) + np.diag(p[t]*(1. - p[t]))

            dLdOT = dPTdOT.dot(dLdPT)       # (w,)
            
            dLdV += np.outer(dLdOT, s[t])    # (w, h)

            dLdST = dLdOT.dot(self.V) + dLdST_           # (h,)
            dLdXPS = (1. - s[t]**2) * dLdST

            dLdW += np.outer(dLdXPS, s[t-1])      # (h,) x (h,) = (h,h)
            dLdU += np.outer(dLdXPS, x[t])      # (h,) x (w,) = (h,w)
            dLdST_ = np.dot(dLdXPS, self.W)

            ## time T-1
            #for dt in reversed(np.arange(-1,t)):
            #    dLdW += np.outer(dLdXPS, s[dt])      # (h,) x (h,) = (h,h)
            #    dLdU += np.outer(dLdXPS, x[dt+1])      # (h,) x (w,) = (h,w)

            #    dLdST = np.dot(dLdXPS, self.W)
            #    dLdXPS = (1. - s[dt]**2) * dLdST

        return [dLdU, dLdV, dLdW]

    def sgd_step(self, x, y, learning_rate):
        '''update model weights according to gradients.
           inputs
           ------
           x: list, dense presentation of one sentence.
           y: list, dense presentation of one sentence.
           learning_rate: double, the learning rate for SGD
        '''
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Compute the gradient using backpropagation
        bptt_gradients = self.bptt(x[0],y[0])

        # List of all weights we want to check
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the model, eg. self.W
            parameter = operator.attrgetter(pname)(self)
            print 'Performing gradient check for parameter %s with size %d'%(pname, np.prod(parameter.shape))
            # Iterate over each element of parameter matrix
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h)) / (2*h)
                parameter[ix] = original_value + h
                gradplus = self.compute_total_loss(x,y)
                parameter[ix] = original_value - h
                gradminus = self.compute_total_loss(x,y)
                estimated_gradient = (gradplus - gradminus) / (2*h)
                # Reset the parameter to original value
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                
                # Compute the relative error: (|x - y|)/(|x| + |y|)
                relative_error = np.abs(backprop_gradient-estimated_gradient) / (
                        np.abs(backprop_gradient)+np.abs(estimated_gradient))

                if relative_error > error_threshold:
                    print 'Gradient Check ERROR: parameter=%s, ix=%s'%(pname, ix)
                    print '+h Loss: %f'% gradplus
                    print '-h Loss: %f'% gradminus
                    print 'Estimated_gradient: %f'%estimated_gradient   
                    print 'Backpropagation gradient: %f'% backprop_gradient
                    print 'Relative Error: %f'%relative_error

                it.iternext()
            print 'Gradient Check for parameter %s passed. '% pname

def train(train_x, train_y, vocabulary_size, learning_rate=1e-3, nepoch=100, evaluate_loss_after=5):
    '''
    Train a RNN language model
    inputs
    ------
    train_x: array of lists, training inputs
    train_y: array of lists, training targets
    vocabulary_size: int, vocabulary size for the input data
    learning_rate: double, learning rate
    nepoch: int, number of epochs for training
    evaluate_loss_after: int, how many epochs to evaluate the training loss
    '''
    # build RNN model
    model = RNN(vocabulary_size,hidden_dim=100)

    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        if epoch % evaluate_loss_after ==0:
            loss = model.get_loss(train_x, train_y)
            losses.append((num_examples_seen, loss))
            time = dttm.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print '%s Loss after seen %d examples, epoch = %d: %f'%(time, num_examples_seen, epoch, loss)

        for i in range(len(train_y)):
            model.sgd_step(train_x[i],train_y[i],learning_rate)
            num_examples_seen += 1

    
if __name__ == '__main__':
    vocabulary_size=8000
    #(train_x, train_y, vocab, index_to_word, word_to_index) = get_data(vocabulary_size)
    #np.random.seed(10)
    model = RNN(vocabulary_size,hidden_dim=100)
    
    ## Test one sentence foward-propagation.
    #o,p, s, spx = model.forward_propagation(one_hot(train_x[10],vocabulary_size))
    #print 'Testing one sentence forward-propagation.'
    #print '\tInput sentense has %d words'%len(train_x[10])
    #print '\tOutput prediction has shape = (%d, %d)\n'%(p.shape[0],p.shape[1])

    ## Test loss of all training examples
    #loss = model.get_loss(train_x, train_y)
    ## Random guess loss
    #random_loss = -1. * np.log(1./vocabulary_size)
    #print 'Testing loss on all training examples.'
    #print '\tExpected loss for random prediction: %.4f'%random_loss
    #print '\tActual loss: %.4f\n'%loss

    
    # Gradient check
    grad_check_vocabulary_size=100
    grad_check_model = RNN(grad_check_vocabulary_size, hidden_dim=10,bptt_truncate=1000)
    grad_check_model.gradient_check(np.array([[0,1,2,3,4,5]]), np.array([[1,2,3,4,5,6]]))

    ## Time test for one step training
    #print '\nTesting one step training.'
    #with time_it():
    #    model.sgd_step(train_x[10], train_y[10],1e-3)
    #
    ## Start training
    #print '\n\n==================== START Training ==================='
    #train(train_x, train_y, vocabulary_size, learning_rate=5e-3, nepoch=100, evaluate_loss_after=1)
    







