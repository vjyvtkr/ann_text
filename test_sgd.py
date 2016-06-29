# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:49:01 2016
@author: vjyvtkr
"""

import numpy as np
def two_layers(X,s0):
    #alphas = [0.001]
    
    # compute sigmoid nonlinearity
    def sigmoid(x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(output):
        return output*(1-output)
    ''' 
    X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
                    
    y = np.array([[0],
    			[1],
    			[1],
    			[0]])
    '''
    synapse_0 = s0
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))
    return layer_1