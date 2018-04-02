import load_features
import draw_result
from neural_network import NeuralNetwork

def main():
    features_cnt,label_features = load_features.load_features("Iris Data.txt")

    training_data = {}
    for label in label_features:
        training_data.update({label:label_features[label][0:30]})

    testing_data = {}
    for label in label_features:
        testing_data.update({label:label_features[label][30:]})
    
    network = NeuralNetwork(features_cnt,4,[4,8,6,5],['tanh','tanh','tanh','tanh'],3,'tanh')

    _,loss_curve = network.train(training_data,0.005,500,0.005)
    draw_result.draw_training(loss_curve)

    x = network.test(testing_data)*100
    print "Accuracy: %f%%"%x

if __name__ == '__main__':
    main()

