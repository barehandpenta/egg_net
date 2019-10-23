from nn import NeuralNetwork
import numpy as np
import time
import sys

sys.path.append('..')

from utils.data_utils import open_csv_file, preProcessing

def preProcessing(filename):
    train_examples = open_csv_file(filename, 'r')
    inputs = [None]*len(train_examples)
    targets = [None]*len(train_examples)
    for i in range(len(train_examples)):
        inputs[i] = np.asfarray(train_examples[i][1:])
        targets[i] = np.zeros(3) + 0.01
        targets[i][int(train_examples[i][0])] = 0.99
    return inputs, targets




EPOCH = 1000
check_point = "new" # last 

def main():
    inputs, targets = preProcessing('konel_egg_train.csv')

    if check_point == "new":
        network = NeuralNetwork(34, 2, 15, 3, 0.01)
    elif check_point == "last":
        network = NeuralNetwork(34, 2, 15, 3, 0.01)
        network.load_model("../data/egg_model_weights.csv")


    print("Starting training eggnet:")
    time.sleep(1)
    for e in range(EPOCH):
        print("====================================")
        print("Epoch:", e, '/', EPOCH)
        for i in range(len(targets)):
            cost = network.train(inputs[i], targets[i])
    print("Training done!")
    print("Saving weights file to ../data/egg_model_weights.csv")
    
    network.save_model('../data/egg_model_weightsf.csv')

    print("Done!")
    _test = input("Press Y for testing model / N to quit: ")
    if _test == "Y" or _test == "y":
        network = NeuralNetwork(34, 2, 15, 3)
        network.load_model('../data/egg_model_weights.csv')
        guess = [None]*len(inputs)
        score = 0
        for i in range(len(inputs)):
            result = network.feedFoward(inputs[i]).T
            index = np.where(result == np.amax(result))
            result = np.zeros(3) + 0.01
            result[index[1]] = 0.99
            guess[i] = result
        for i in range(len(targets)):
            if np.array_equal(guess[i], targets[i]):
                score += 1

        print("Testing done:", "Score: ", score, " ", score*100/len(targets))

if __name__ == "__main__":
    main()