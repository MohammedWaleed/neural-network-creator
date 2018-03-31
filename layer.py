import math
import random 
import numpy as np


class Layer:
    
    def __init__(self,neurons_cnt,weights_dim,activation_fn_type = 'sigmoid',with_bias = True):
        """
            number of neurons
            weights matrix dimensions
            activation function [sigmoid,tanh]
        """
        self.with_bias = with_bias
        self.bias = 0 + with_bias
        self.neurons = np.zeros(neurons_cnt)
        self.acctual_val = self.neurons
        self.error = self.neurons
        self.activ_fn_type = activation_fn_type
        self.weights = self.init_weights(weights_dim)
    
    def evaluate(self,x):
        w = np.transpose(self.weights)
        
        self.acctual_val = w * x + self.bias

        for i in range(0,len(self.acctual_val)):
            self.neurons[i] = self.activation_fn(self.acctual_val[i])

        return self.neurons

    def init_weights(self,w_dim):
        w = np.ones([w_dim]*random.uniform(-1e-9, 1e-9))
        return w

    def activation_fn(self,val):
        if(self.activ_fn_type=='sigmoid'):
            return self.sigmoid(val)
        else:
            return self.tanh(val)
    
    def activation_fn_deriv(self,val):
        if(self.activ_fn_type=='sigmoid'):
            return self.sigmoid_deriv(val)
        else:
            return self.tanh_deriv(val)
    
    def sigmoid(self,val):
        return 1 / (1 + math.exp(-val))
    
    def tanh(self,val):
        return np.tanh(val)
    
    def sigmoid_deriv(self,val):
        f = self.sigmoid(val)
        f = f * (1-f)
        return f
    
    def tanh_deriv(self,val):
        f = self.tanh(val)
        f = 1 - (f ** 2)
        return f
    def calc_error_as_output(self,desired):
        """
            deal with this layer as an output layer, 
            error is equal to (desired_output - actual_output)
        """
        self.error = desired-self.neurons
        return self.error
    
    def calc_error(self,next_layer):
        """
            calculate the error from the next layer (back-propagate the error)
        """
        sw = next_layer.weights * next_layer.error
        for i in range(0,len(self.neurons)):
            self.error[i]= self.activation_fn_deriv(self.acctual_val[i])*sw[i]
        
        return self.error

        

            

        