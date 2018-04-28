from layer import Layer
import numpy as np
import time
class NeuralNetwork:
    def __init__(self,features_cnt,layers_cnt,layers_neu_cnt,layers_actv_fn,output_cnt,out_actv_fn,with_bias = True):
        """
            features_cnt: number of input features(aka input layer)
            layers_cnt: number of hidden layers
            layers_neu_cnt: a list contains the number of neurons in each layer
            output_cnt: number of neurons in the output layer
        """
        self.layers = []

        l = Layer(features_cnt,1,'',with_bias)
        self.layers.append(l)

        for i in range(layers_cnt):
            neu_cnt = layers_neu_cnt[i]
            weights_len = self.layers[i].neurons_cnt
            activ_fn = layers_actv_fn[i]
            l = Layer(neu_cnt,weights_len,activ_fn,with_bias)
            self.layers.append(l)
        
        l = Layer(output_cnt,self.layers[-1].neurons_cnt,out_actv_fn,with_bias)
        self.layers.append(l)


    
    def train(self,label_features_map,eta,num_epochs,with_mse,mse_threshold):
        loss_curve = []
        then = time.time()
        for epoch in range(0, num_epochs):
            # iterate over each class
            for class_num,class_label in zip(range(0,len(label_features_map)),label_features_map):
                
                for features in label_features_map[class_label]: # iterate over class samples feature
                    
                    self.evaluate(features)
                    
                    desired = np.zeros(self.layers[-1].neurons_cnt)
                    desired[class_num]=1.0

                    X = self.layers[-1].calc_error(desired,True)# calculate the error in the output layer
                    
                    for i in range (len(self.layers)-2,0,-1):# propagate the error throw the network
                        X = self.layers[i].calc_error(X)

                    X = self.layers[0].neurons
                    for i in range(1,len(self.layers)):# update the weights
                        X = self.layers[i].update_weights(X,eta)

                    X = self.layers[0].neurons
                    for i in range(1,len(self.layers)):# re-evaluate the network layers
                        X = self.layers[i].evaluate(X)
                    
                                        
            
           
            error_list =[]
            # iterate over each class
            for class_num,class_label in zip(range(0,len(label_features_map)),label_features_map):
                for features in label_features_map[class_label]: # iterate over class samples feature
                    self.evaluate(features)
                    desired = np.zeros(self.layers[-1].neurons_cnt)
                    desired[class_num]=1.0
                    output = self.layers[-1].layer_as_array(self.layers[-1].neurons)
                    error_list.append(self.calc_mse(desired,output))

            mse = 0 
            for i in error_list:
                mse+=i
            
            mse/=len(error_list)
            loss_curve.append(mse)
            now = time.time()

            tm = now - then
            print("Epoch %d-> time: %f sec, trainin_loss: %f, learning_rate%f\n"%(epoch,tm,mse,eta))
            if with_mse and np.isclose(mse,mse_threshold):
                break

        return self, loss_curve

    def evaluate(self,x):
        X = self.layers[0].set_neurons_value(x)
        for i in range(1,len(self.layers)):# evaluate the network layers
             X = self.layers[i].evaluate(X)
        
        output = self.layers[-1].layer_as_array(self.layers[-1].neurons)
        
        return np.argmax(output)

    def test(self,labels_features_map,class_cnt):
        co_mat = np.zeros([class_cnt,class_cnt])

        for i,label in zip(range(0,class_cnt),labels_features_map):
            for features in labels_features_map[label]:
                j = self.evaluate(features)
                co_mat[i,j]+=1
        total = 0
        diagonal = 0
        for i in range(0,class_cnt):
            for j in range(0,class_cnt):
                if i==j:
                    diagonal+= co_mat[i,j]
                total+=co_mat[i,j]
        print co_mat
        return diagonal/total 

    def calc_mse(self,desired,output):
             
        mse = 0
        for i in range(0,len(desired)):
            mse += (desired[i]-output[i])*(desired[i]-output[i])
        mse /=len(desired)
        return mse
