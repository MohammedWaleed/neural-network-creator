from neoron import Neoron
import numpy as np
import random
class Layer:
    
    def __init__(self,neurons_cnt,weights_len,activation_fn_type = 'sigmoid',with_bias = True):
        """
            number of neurons
            weights matrix dimensions
            activation function [sigmoid,tanh]
            train with bias
        """
        self.neurons_cnt = neurons_cnt
        self.bias = np.zeros(neurons_cnt)
        self.with_bias = with_bias
        if with_bias:
            for b in self.bias:
                b += random.uniform(-1e-4, 1e-4)
        self.neurons = []
        for _ in range(neurons_cnt):
            neu = Neoron(weights_len,activation_fn_type)
            self.neurons.append(neu)
    
    def set_neurons_value(self,X):
        for i in range(self.neurons_cnt):
            self.neurons[i].value = X[i]
        return self.neurons

    def update_weights(self,prev_layer_neurons,eta):
        X = self.layer_as_array(prev_layer_neurons)
        for neu in self.neurons:
            neu.update_weights(X,eta)

        if self.with_bias:
            for i in range(self.neurons_cnt):
                self.bias[i] += eta*self.neurons[i].error

        return self.neurons

    def calc_error(self,next_layer_neurons,is_last_layer=False):
        if is_last_layer:
            for i in range(self.neurons_cnt):
                self.neurons[i].calc_error(0,next_layer_neurons[i],is_last_layer)
            
        else:
            for i in range(self.neurons_cnt):
                self.neurons[i].calc_error(i,next_layer_neurons,is_last_layer)
            
        return self.neurons

    def evaluate(self,prev_layer_neurons):
        X = self.layer_as_array(prev_layer_neurons)

        for i in range(self.neurons_cnt):
            self.neurons[i].evaluate(X,self.bias[i])
        
        return self.neurons
    
    def layer_as_array(self,layer_neurons):
        x = []
        for neu in layer_neurons:
            x.append(neu.value)
        
        return np.array(x)