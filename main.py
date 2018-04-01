import load_features
import draw_result
import neural_network

def main():
    label_features = load_features.load_features("IrisData.txt")

    training_data = {}
    for label in label_features:
        training_data.update({label:label_features[label][0:30]})

    testing_data = {}
    for label in label_features:
        testing_data.update({label:label_features[label][30:]})
    
    

