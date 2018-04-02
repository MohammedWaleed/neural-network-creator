import sys
import numpy as np

def load_features(filename):
    dataset = open(filename,"r")

    features=[]
    labels=[]
    label_features = {}
    features_cnt =0
    for data in dataset: # load features and labels to the memory
        temp=data.split(',')
        features_cnt = len(temp)-1
        label_features.update({temp[-1][0:-2]:[]})
        features.append(temp[:4])
        labels.append(temp[-1][0:-2])
        
    features = np.array(features).astype(np.float)

    for i in range(0,len(labels)):
        label_features[labels[i]].append(features[i])

    
    return features_cnt, label_features
