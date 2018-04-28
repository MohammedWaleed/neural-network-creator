import numpy as np
import random
import math
class Neoron:
    def __init__(self,weights_len,activ_fn):
        
        self.weights_len = weights_len
        
        self.weights = np.zeros(weights_len)
        for w in self.weights:
            w += random.uniform(-1e-4, 1e-4) 
        
        self.activ_fn = activ_fn
        self.error = 0.0
        self.value = 0.0
    
    def evaluate(self,X,bias):

        val = 0.0
        for i in range(self.weights_len):
            val += self.weights[i] * X[i]
        
        val += bias
        self.value = self.activation_fn(val)
        
        return self

    def activation_fn(self,val):
        
        if(self.activ_fn=='sigmoid'):
            return self.sigmoid(val)
        else:
            return self.tanh(val)
    
    def activation_fn_deriv(self,val):
        
        if(self.activ_fn=='sigmoid'):
            return self.sigmoid_deriv(val)
        else:
            return self.tanh_deriv(val)
    
    def sigmoid(self,val):
        return 1.0 / (1.0 + math.exp(-val))
    
    def tanh(self,val):
        return np.tanh(val)
    
    def sigmoid_deriv(self,val):
       
        f = self.sigmoid(val)
        f = f * (1.0-f)
        return f
    
    def tanh_deriv(self,val):
        
        f = self.tanh(val)
        f = 1.0 - (f ** 2.0)
        return f
    
    def calc_error(self,indx,next_layer_neurons,is_last_layer=False):
        
        if is_last_layer:
            self.error = (next_layer_neurons-self.value) * self.activation_fn_deriv(self.value)
        else:
            e = 0.0
            for i in range(len(next_layer_neurons)):
                e += next_layer_neurons[i].error * next_layer_neurons[i].weights[indx]

            self.error = e * self.activation_fn_deriv(self.value)
            
        return self.error

    def update_weights(self,X,eta):
        
        for i in range(self.weights_len):
            self.weights[i] += eta * self.error * X[i]
        