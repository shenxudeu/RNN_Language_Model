'''
Batched LSTM in Numpy.
model structure:

    forget gate: 
        f_t = \sigma(W^f [h_{t-1}, x_t] + b^f)

    input gate: 
        i_t = \sigma(W^i [h_{t-1}, x_t] + b^i)
        c_t* = \tanh(W^c [h_{t-1}, x_t] + b^c)

    State:
        c_t = f_t * c_{t-1} + i_t * c_t*

    output gate:
        o_t = \sigma(W^o [h_{t-1}, x_t] + b^o)
        h_t = o_t * tanh(c_t)

    Loss:
        L_t = cross_entropy(y_t, softmax(h_t))

Notations are based on Colah's blog http://colah.github.io/posts/2015-08-Understanding-LSTMs/
Code is derived from Andrej Karpathy's code https://gist.github.com/karpathy/587454dc0146a6ae21fc
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


class LSTM:
    def __init__(self, word_dim, hidden_dim, bptt_truncate=4, seed=1984):
        '''
        LSTM Model
        inputs
        ------
            word_dim: int, input size
            hidden_dim: int, hidden node size
            bptt_truncate: int
        weights (with bias)
        ------
            WF: np.array, (word_dim+hidden_dim+1, hidden_dim)
            WC: np.array, (word_dim+hidden_dim+1, hidden_dim)
            WI: np.array, (word_dim+hidden_dim+1, hidden_dim)
            WO: np.array, (word_dim+hidden_dim+1, hidden_dim)
        '''
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize weights, first row is bias, initialize bias to zero
        np.random.seed(seed)
        fan_in = word_dim + hidden_dim + 1
        self.WF = np.random.randn(fan_in, hidden_dim) / np.sqrt(fan_in)
        self.WF[0,:] = 0.
        self.WC = np.random.randn(fan_in, hidden_dim) / np.sqrt(fan_in)
        self.WC[0,:] = 0.
        self.WI = np.random.randn(fan_in, hidden_dim) / np.sqrt(fan_in)
        self.WI[0,:] = 0.
        self.WO = np.random.randn(fan_in, hidden_dim) / np.sqrt(fan_in)
        self.WO[0,:] = 0.
        self.V = np.random.randn(self.hidden_dim, self.word_dim) / np.sqrt(self.hidden_dim)
        
    
    def forward_propagation(self, x):
        '''
        LSTM forward pass 
        inputs
        ------
            x: np.array, (n_sequence, batch_size, word_dim), one-hot presentation

        outputs
        ------
            c: np.array, state
            h: np.array, log-probability
            cache: list
        '''
        (T, batch_size, word_dim) = x.shape# sequence length
        assert (word_dim == self.word_dim), 'forward_propagation, input must be one-hot.'
        
        # We add one additional element for the initial output and state, which set to 0
        h = np.zeros((T+1, batch_size, self.hidden_dim)) # hidden state
        c = np.zeros((T+1, batch_size, self.hidden_dim)) # memory
        cf = np.zeros((T+1, batch_size, self.hidden_dim)) # memory activation


        # Intimidate tensors
        oG = np.zeros((T, batch_size, self.hidden_dim)) # output gate
        oGf = np.zeros(oG.shape) # output gate activation
        fG = np.zeros((T, batch_size, self.hidden_dim)) # forget gate
        fGf = np.zeros(fG.shape) # forget gate activation
        iG = np.zeros((T, batch_size, self.hidden_dim)) # input gate
        iGf = np.zeros(iG.shape) # input gate activation
        icG = np.zeros((T, batch_size, self.hidden_dim)) # input C gate
        icGf = np.zeros(icG.shape) # input C gate activation

        hcx = np.zeros((T, batch_size, self.word_dim +self. hidden_dim + 1)) # h concat x

        # Output tensor, log prob and prob
        y = np.zeros((T, batch_size, self.word_dim))   # log prob
        p = np.zeros((T, batch_size, self.word_dim)) # prob

        # For each time step ...
        for t in np.arange(T):
            # concat h and x
            hcx[t,:,0] = 1.
            hcx[t,:,1:self.word_dim+1] = x[t,:,:]
            hcx[t,:,self.word_dim+1:] = h[t-1,:,:]
            
            # Compute gates
            oG[t] = hcx[t].dot(self.WO) 
            oGf[t] = 1./ (1. + np.exp(-oG[t]))
            fG[t] = hcx[t].dot(self.WF) 
            fGf[t] = 1./ (1. + np.exp(-fG[t]))
            iG[t] = hcx[t].dot(self.WI) 
            iGf[t] = 1./ (1. + np.exp(-iG[t]))
            icG[t] = hcx[t].dot(self.WC) 
            icGf[t] = np.tanh(icG[t])

            # Compute state 'c' and output 'h' by connecting tensor graph
            c[t] = fGf[t] * c[t-1] + iGf[t] * icGf[t]
            cf[t] = np.tanh(c[t])
            h[t] = oGf[t] * cf[t]
            
            # Compute output log prob and prob
            y[t] = h[t].dot(self.V)
            p[t] = softmax(y[t])

        # caching
        cache = [y, h, c, cf, hcx, oG, oGf, fG, fGf, iG, iGf, icG, icGf]

        return p, cache

    def predict(self, x):
        '''Perform forward propagation and return index of the highest score'''
        p, _ = self.forward_propagation(x)
        embed()
        HERE
        return np.argmax(p, axis=1)

    def compute_total_loss(self, x, y):
        L = 0.
        # Forward-propagation for each sequence
        for i in range(len(x)):
            x_sent = one_hot(x[i], self.word_dim)
            x_sent = x_sent.reshape(x_sent.shape[0], 1, x_sent.shape[1])
            p, cache = self.forward_propagation(x_sent)
            correct_predictions = p[np.arange(len(y[i])),:,y[i]]
            L += -1. * np.sum(np.log(correct_predictions))
        return L

    def get_loss(self, x, y):
        L = self.compute_total_loss(x, y)
        # Divide the total loss by the number of training examples
        N = np.sum(len(y[i]) for i in range(len(y)))
        return L / N
    def bptt(self, x, y):
        '''
        Back-propagation through time
        Support batch backprop
        inputs
        ------
        x: np.array, (T, batch_size, word_dim)
        y: np.array, (T, batch_size, 1)
        '''
        (T, batch_size, word_dim) = x.shape
        ########## FORWARD PASS
        p, [ys, h, c, cf, hcx, oG, oGf, fG, fGf, iG, iGf, icG, icGf] = self.forward_propagation(x)

        # Initialize gradients in weights
        dWF = np.zeros(self.WF.shape)
        dWC = np.zeros(self.WC.shape)
        dWI = np.zeros(self.WI.shape)
        dWO = np.zeros(self.WO.shape)
        dV  = np.zeros(self.V.shape)

        # Initialize gradients in intimidate tensors
        dys = np.zeros(ys.shape)

        ########## BACKWARD PASS - Unfolding the network
        dL = 1.
        for t in np.arange(T)[::-1]:
            # backward L = cross_entropy(y, softmax(ys))
            dys[t] = p[t] * dL
            dys[t][np.arange(batch_size),y[t]] -= 1.
            
            # backward ys_t = h_t . V
            dV += h[t].T.dot(dys[t])
            dh_t = dys[t].dot(self.V.T)

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0,t-self.bptt_truncate), t+1)[::-1]:
                # backward h_t = oGf_t * cf_t
                doGf_t = dh_t * cf[bptt_step]
                dcf_t = dh_t * oGf[bptt_step]
                
                # backward oGf_t = sigma(oG)
                doG_t = (1. - oGf[bptt_step]) * oGf[bptt_step] * doGf_t
                
                # backward OG_t = hcx_t . WO
                dWO += hcx[bptt_step].T.dot(doG_t)
                dhcx_t = doG_t.dot(self.WO.T)

                # backward cf_t = tanh(c_t)
                dc_t = (1. - cf[bptt_step]**2) * dcf_t
     
                # backward c_t = fGf_t * c_tm1 + iGf_t * icGf_t
                dfGf_t = c[bptt_step-1] * dc_t
                dc_t_minus_1 = fGf[bptt_step] * dc_t
                diGf_t = icGf[bptt_step] * dc_t
                dicGf_t = iGf[bptt_step] * dc_t

                # backward fGf_t = sigma(fG_t)
                dfG_t = (1. - fGf[bptt_step]) * fGf[bptt_step] * dfGf_t
                # backward fG_t = hcx_t . WF
                dWF += hcx[bptt_step].T.dot(dfG_t)
                dhcx_t += dfG_t.dot(self.WF.T)

                # backward iGf_t = sigma(iG_t)
                diG_t = (1. - iGf[bptt_step]) * iGf[bptt_step] * diGf_t
                # backward iG_t = hcx_t . WI
                dWI += hcx[bptt_step].T.dot(diG_t)
                dhcx_t += diG_t.dot(self.WI.T)

                # backward icGf_t = tanh(icG_t)
                dicG_t = (1. - icGf[bptt_step]**2) * dicGf_t
                # backward cG_t = hcx_t . WC
                dWC += hcx[bptt_step].T.dot(dicG_t)
                dhcx_t += dicG_t.dot(self.WC.T)
                
                dh_t_minus_1 = dhcx_t[:,self.word_dim+1:]
                dh_t = dh_t_minus_1
                dc_t = dc_t_minus_1
            
        return [dWF, dWC, dWI, dWO, dV]


def gradient_check(h=0.001, error_threshold=0.01):
    '''
    Check gradient computation with small test data
    '''
    # Build model and generate simple data
    word_dim, hidden_dim = 10, 5
    model = LSTM(word_dim, hidden_dim, bptt_truncate=1000)
    x = [[0,1,2,3,4,5,6]]
    y = [[1,2,3,4,5,6,7]]

    # Compute the gradients using backpropagation
    #total_loss = model.compute_total_loss(x, y)
    x_ = one_hot(x[0], word_dim) 
    x_ = x_.reshape(x_.shape[0],1,x_.shape[1])
    y_ = y[0]
    bptt_gradients = model.bptt(x_, y_)

    # List of all weights we want to check
    model_parameters = ['WF','WC','WI','WO','V']

    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the model, eg. self.WF
        parameter = operator.attrgetter(pname)(model)
        print 'Performing gradient check for parameter %s with size %d'%(pname, np.prod(parameter.shape))
        # Iterate over each element of parameter matrix
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - h(x-h)) / (2*h)
            parameter[ix] = original_value + h
            gradplus = model.compute_total_loss(x,y)
            parameter[ix] = original_value - h
            gradminus = model.compute_total_loss(x,y)
            estimated_gradient = (gradplus - gradminus) / (2.*h)
            # Reset the parameter to original value
            parameter[ix] = original_value
            backprop_gradient = bptt_gradients[pidx][ix]

            # Compute relative error: (|x - y|) / (|x| + |y|)
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                    np.abs(backprop_gradient) + np.abs(estimated_gradient))

            if relative_error > error_threshold:
                print 'Gradient Check Error: parameter = %s, ix = %s'%(pname, ix)
                print '+h Loss: %f'%gradplus
                print '-h Loss: %f'%gradminus
                print 'Estimated gradient: %f'% estimated_gradient
                print 'Backpropagation gradient: %f'% backprop_gradient
                print 'Relative Error: %f'% relative_error

            it.iternext()
        print 'Gradient Check for parameter %s passed. '% pname



if __name__ == '__main__':
    #vocabulary_size=8000
    #(train_x, train_y, vocab, index_to_word, word_to_index) = get_data(vocabulary_size)
    #np.random.seed(10)
    #model = LSTM(vocabulary_size,hidden_dim=100)
    #
    ## Test one sentence foward-propagation.
    #x = one_hot(train_x[10], vocabulary_size)
    #x = x.reshape(x.shape[0], 1, x.shape[1])
    #p,cache = model.forward_propagation(x)
    #print 'Testing one sentence forward-propagation.'
    #print '\tInput sentense has %d words'%len(x)
    #print '\tOutput prediction has shape = (%d, %d, %d)\n'%(p.shape[0],p.shape[1], p.shape[2])

    ## Test loss of all training examples
    #loss = model.get_loss(train_x, train_y)
    ## Random guess loss
    #random_loss = -1. * np.log(1./vocabulary_size)
    #print 'Testing loss on all training examples.'
    #print '\tExpected loss for random prediction: %f'%random_loss
    #print '\tActual loss: %f\n'%loss
 
    
    # gradient check
    gradient_check()

    embed()
    HERE





