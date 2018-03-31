from layer import Layer
import numpy as np
class NeuralNetwork:
    def __init__(self,features_cnt,layers_cnt,layers_neu_cnt,layers_actv_fn,output_cnt,out_actv_fn,with_bias = True):
        """
            features_cnt: number of input features(aka input layer)
            layers_cnt: number of hidden layers
            layers_neu_cnt: a list contains the number of neurons in each layer
            output_cnt: number of neurons in the output layer
        """
        self.with_bias = with_bias

        nn = []
        l = Layer(features_cnt,1,None,with_bias)
        nn.append(l)

        for i in range(0,layers_cnt):
            x = len(nn[i].neurons)
            y = layers_neu_cnt[i]
            l = Layer(y,[x,y],layers_actv_fn[i],with_bias)
            nn.append(l)
        
        x = len(nn[i].neurons)
        l = Layer(output_cnt,[x,output_cnt],out_actv_fn,with_bias)
        self.nn = nn
    
    def train(self,label_features_map,eta,num_epochs,mse_threshold):
        
        for epoch in range(0, num_epochs):
            loss = 0 # should be removed
            # iterate over each class
            for class_num,class_label in zip(range(0,len(label_features_map)),label_features_map):
                
                for features in label_features_map[class_label]: # iterate over class samples feature
                    self.nn[0].neurons = features
                    
                    for i in range(1,len(self.nn)) : # train the network by evaluation the layers
                        self.nn[i].evaluate(self.nn[i-0].neurons)
                    
                    desired = np.zeros(self.nn[-1])
                    desired[class_num]=1

                    self.nn[-1].calc_error_as_output(desired)
                    for i in range (len(self.nn.count)-2,0,-1):
                        self.nn[i].calc_error(self.nn[i+1])
                    





            print("Epoch %d-> learning_rate: %d, trainin_loss: %d\n"%(epoch,eta,loss))
        
        return 0
