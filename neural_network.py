from layer import Layer
import numpy as np
import math
class NeuralNetwork:
    def __init__(self,features_cnt,layers_cnt,layers_neu_cnt,layers_actv_fn,output_cnt,out_actv_fn,with_bias = True):
        """
            features_cnt: number of input features(aka input layer)
            layers_cnt: number of hidden layers
            layers_neu_cnt: a list contains the number of neurons in each layer
            output_cnt: number of neurons in the output layer
        """
        self.with_bias = with_bias

        netrowk = []
        l = Layer(features_cnt,1,None,with_bias)
        netrowk.append(l)

        for i in range(0,layers_cnt):
            x = len(netrowk[i].neurons)
            y = layers_neu_cnt[i]
            l = Layer(y,[x,y],layers_actv_fn[i],with_bias)
            netrowk.append(l)
        
        x = len(netrowk[-1].neurons)
        l = Layer(output_cnt,[x,output_cnt],out_actv_fn,with_bias)
        netrowk.append(l)

        self.netrowk = netrowk
        

    
    def train(self,label_features_map,eta,num_epochs,with_mse,mse_threshold):
        loss_curve = []
        for epoch in range(0, num_epochs):
            loss = 0.0 # should be removed
            l_cnt = 0
            # iterate over each class
            for class_num,class_label in zip(range(0,len(label_features_map)),label_features_map):
                error = 0.0
                e_cnt = 0 
                for features in label_features_map[class_label]: # iterate over class samples feature
                    
                    self.netrowk[0].neurons = features

                    for i in range(1,len(self.netrowk)):# train the network by evaluation the layers
                        self.netrowk[i].evaluate(self.netrowk[i-1].neurons)
                    
                    desired = np.zeros(len(self.netrowk[-1].neurons))
                    desired[class_num]=1.0

                    self.netrowk[-1].calc_error_as_output(desired) # calculate the error in the output layer
                    for i in range (len(self.netrowk)-2,0,-1):
                        self.netrowk[i].calc_error(self.netrowk[i+1])
                    
                    for i in range(1,len(self.netrowk)): # propagate the error throw the network
                        self.netrowk[i].update(eta,self.netrowk[i-1])
                    
                    for i in range(1,len(self.netrowk)) : # feed the input to the network after updating 
                                                          # the weights
                        self.netrowk[i].evaluate(self.netrowk[i-1].neurons)
                    
                    error += self.calc_mse(desired,self.netrowk[-1].neurons)
                    e_cnt+=1
                    l_cnt +=1
                loss += error
                #print("Epoch %d-> learning_rate: %f, trainin_loss: %f\n"%(epoch,eta,error/e_cnt))
            loss_curve.append(loss/l_cnt)
            #print("Epoch %d-> learning_rate: %f, trainin_loss: %f\n"%(epoch,eta,loss/l_cnt))
            if with_mse and np.isclose(loss/l_cnt,mse_threshold):
                break

        return self.netrowk, loss_curve

    def evaluate(self,x):
        self.netrowk[0].neurons = x
        for i in range(1,len(self.netrowk)) : # train the network by evaluation the layers
            self.netrowk[i].evaluate(self.netrowk[i-1].neurons)
        
        output = self.netrowk[-1].neurons
        
        return np.argmax(output)

    def test(self,labels_features_map):
        sz = len(labels_features_map)
        co_mat = np.zeros([sz,sz])

        for i,label in zip(range(0,sz),labels_features_map):
            for features in labels_features_map[label]:
                j = self.evaluate(features)
                co_mat[i,j]+=1
        total = 0
        diagonal = 0
        for i in range(0,sz):
            for j in range(0,sz):
                if i==j:
                    diagonal+= co_mat[i,j]
                total+=co_mat[i,j]
        print co_mat
        return diagonal/total 

    def calc_mse(self,desierd,output):
             
        mse = 0
        for i in range(0,len(desierd)):
            mse += (desierd[i]-output[i])*(desierd[i]-output[i])
        mse /=2
        return mse
