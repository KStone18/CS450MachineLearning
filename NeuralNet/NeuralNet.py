from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp
from sklearn import preprocessing
from random import triangular as tri

import math
import pandas as pds
import numpy as np
import sys

# Create the class Neuron
class Neuron:

    # Declare variables for the class
    def __init__(self, attribute_num):
        self.set_of_weights = [tri(-1.0, 1.0) for att in range(attribute_num + 1)]
        self.bias = -1
        self.threshold = 0
        self.activate_val = 0


    # Output function - Multiplies weights and inputs then sums them together.
    # This is our hj values
    def output(self, act_inputs):
        act_inputs = np.append(act_inputs, self.bias)
        sum = 0;
        for w, input in enumerate(act_inputs):
            sum += (self.set_of_weights[w] * input)

        sig_val = self.sigmoidal(sum)

        #print(sig_val)
        return sig_val
        #return 0 if sum < self.threshold else 1
        #return 1 if sum([self.set_of_weights[i] * x_val for i, x_val in enumerate(act_inputs)]) >= self.threshold else 0

    def sigmoidal(self, sum_hj):
        activate_val = 1/(1 + math.exp(-(sum_hj)))
        return activate_val

# class layer:
#     def __init__(self):
#         self.num_of_nodes = 0

class Network:
    def __init__(self, num_layers, data, num_of_targets):
        self.layers = []
        self.total_results = []


        # A for loop looping from 0 - number of layers
        # We then create a neuron and if x is greater than 0 we create the
        # network with 1 - the amount of layers. Otherwise we
        # for x in range(0, num_of_layers):
        #     self.layers.append([Neuron(len(self.layers[x - 1]) if x > 0 else data.shape[1])
        #                                for _ in range(int(input("How many Neurons for layer " + str(x) + "?
        for i in range(0, num_layers):
            self.layers.append(self.create_layers(num_layers, i, data, num_of_targets))


    def create_layers(self, num_layers, layer_num, data, num_tar):
        # Hidden Layer
        if layer_num > 0 and layer_num < num_layers - 1:
            #print("hidden layer")
            return [Neuron(len(self.layers[layer_num - 1]))
                    for _ in range(int(input("How many Neuron in layer " + str(layer_num) + "? ")))]

        # first or input layer
        elif layer_num == 0:
            #print("first layer")
            return [Neuron(data.shape[1])
                    for _ in range(int(input("How many Neuron in layer " + str(layer_num) + "? ")))]

        # last or output layer
        else:
           # print("last layer")
            return [Neuron(len(self.layers[layer_num - 1])) for _ in range(num_tar)]


    def results(self, actual_inputs):
        results = []

        for i, layer in enumerate(self.layers):
            results.append([node.output(results[i - 1] if i > 0 else actual_inputs) for node in layer])

        return results

    def print_outputs(self):
      for row in self.total_results:
          print(row[-1])

    def train(self, train_data):
        for a_inputs in train_data:
            activation_num = self.results(a_inputs)
            #print(activation_num[-1])
            self.total_results.append(activation_num)

    def predict(self, test_data):
        pre = []
        for t in test_data:
            pre.append(self.results(t))

        #print("lenght of pre: ", len(pre))
        # for p in pre:
        #     print(p[-1])
        return pre


# Get the size of the test
def get_test_size():
    test_size = float(input("Choose a test size between 0.1 - 0.5: "))

    while test_size <= 0.0 or test_size > 0.5:
        test_size = float(input("Please choose a test size between 0.1 and 0.5: "))

    return test_size


# Get the number of how many times to shuffle
def get_status():
    state = int(input("Please choose how many times to shuffle: "))

    while state <= 0:
        state = int(input("Please choose a valid number times. Valid numbers are those greater than Zero: "))

    return state




# Calculates the accuracy
def get_accuracy(results_of_predict, test_targets):
    value_correct = 0

    for rop, test_t in zip(results_of_predict, test_targets):
        if (test_t == rop.index(max(rop))):
            value_correct += 1

    print("\nCorrectly foretold ", value_correct, " of ", test_targets.size, "\nI was correct ",
          "{0:.2f}% of the time!!!".format(100 * (value_correct / test_targets.size)), sep="")





# Normalizes the data to prevent skewness
def standardize(x_data, y_data):
    std_scale = preprocessing.StandardScaler().fit(x_data)
    train_data = std_scale.transform(x_data)
    test_data = std_scale.transform(y_data)

    return (train_data, test_data)

# Process the datasets and splits the data.
def process_info(data, target, classifier):
    test = float(get_test_size())
    state = int(get_status())


    train_data, test_data, train_target, test_target = tsp(data, target, test_size=test, random_state=state)

    train_data_std, test_data_std = standardize(train_data, test_data)

    classifier.train(train_data_std)
    get_accuracy(classifier.predict(test_data), test_target)


    # predict_results = classifier.predict(test_data_std)
    # get_neural_accuracy(predict_results, target)

    #get_accuracy()


def load_file(filename):
    df = pds.read_csv(filename)

    data = df.ix[:, df.columns != "target"]
    target = df.ix[:, df.columns == "target"]

    #labels = df.columns
    #nlabels = labels[:-1]

    return data.values, target.values

def load_dataset(data_set):
    return data_set.data, data_set.target

def target_num(targets):
    targ = []
    for t in targets:
        if t not in targ:
            targ.append(t)


    return len(targ)


# Driver of the program - will load data sets and call appropriate
# functions to process the data.
def main(argv):

    # Iris dataset
    data, targets = load_dataset(datasets.load_iris())
    #print(targets)
    # Pima Dataset
    #data, targets = load_file("pima.csv")

    # Get number of neurons and columns
    #num_layers = int(input("How many layers do you want? "))
    num_cols = len(data[0])

    # get the num_of_targets to force number of last layer neurons
    num_targets = target_num(targets)
    #print("num targets", num_targets)


    num_layers = 0
    while num_layers < 1:
        num_layers = int(input("How many layers do you want? "))


    # Make the Network
    neural_net = Network(num_layers, data, num_targets)

    process_info(data, targets, neural_net)

    # for i in neural_net.total_results:
    #     print(i)
    # #


   # train_data, test_data = process_info(data, targets, neural_net)

    # Train the algorithm
    #neural_net.train(train_data)

    # Print the output of the neuron
    #neural_net.print_outputs()







    #get_accuracy(classifier.predict(train_data_std, train_target, test_data_std), test_target)

    #print(neural_net.total_results)


    # while number != 1 or number != 2 or number != 0:
    #     print("Please choose one of the following")
    #     print("\t0. Type 0 to quit")
    #     print("\t1. Load from a CSV file")
    #     print("\t2. Load from the Iris study")
    #     #print("\t3. Load from the Breast Cancer Study")
    #     #print("\t4. Use the KnnClassifier built into sklearn")
    #
    #
    #     number = int(input("> "))
    #
    #     if (number == 1):
    #         data, targets, names = load_file("pima.csv")
    #         process_info(data.values, target.values, classif)
    #
    #     if (number == 2):
    #         dataset2 = datasets.load_iris()
    #         data = dataset2.data
    #         target = dataset2.target
    #         process_info(data, target, classif)
    #
    #     if (number == 3):
    #         dataset2 = datasets.load_breast_cancer()
    #         data = dataset2.data
    #         target = dataset2.target
    #         process_info(data, target, classif)
    #
    #
    #
    #     if (number == 0):
    #         return



if __name__ == "__main__":
    main(sys.argv)