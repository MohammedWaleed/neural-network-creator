import load_features
import draw_result
from neural_network import NeuralNetwork
import json
import random

    
def main():

    data = load_data()
    features_cnt,label_features = load_features.load_features(data["inputFile"])
    #for label in label_features:
     #   label_features[label] = random.shuffle(label_features[label])
        
    training_data = {}
    for label in label_features:
        training_data.update({label:label_features[label][:data["traininSamplesCnt"]]})

    testing_data = {}
    for label in label_features:
        testing_data.update({label:label_features[label][data["traininSamplesCnt"]:]})
    
    network = NeuralNetwork(features_cnt,
                            data["hiddenLayers"]["cnt"],
                            data["hiddenLayers"]["layersNeuronsCnt"],
                            data["hiddenLayers"]["layersActivFns"],
                            data["outputLayer"]["neorunsCnt"],
                            data["outputLayer"]["activFn"],
                            data["withBias"])

    model,loss_curve = network.train(training_data,
                                data["eta"],
                                data["epochsNo"],
                                data["stopMSE"],
                                data["MSE"])

    

    draw_result.draw_training(loss_curve)

    x = network.test(testing_data,data["outputLayer"]["neorunsCnt"])*100
    print "Accuracy: %f%%"%x

def load_data():
    with open('hyperparameters.json') as data_file:
        data = json.load(data_file)
    return data
if __name__ == '__main__':
    main()

