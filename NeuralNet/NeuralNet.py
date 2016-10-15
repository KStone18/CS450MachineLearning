from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp
from sklearn import preprocessing
from random import triangular as tri


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

    # Output function - Multiplies weights and inputs then sums them together.
    #
    def output(self, act_inputs):
        act_inputs = np.append(act_inputs, self.bias)
        return 1 if sum([self.set_of_weights[i] * x_val for i, x_val in enumerate(act_inputs)]) >= self.threshold else 0



class Network:
    def __init__(self, num_of_neuron, num_of_attributes):
        self.nodes = [Neuron(num_of_attributes) for _ in range(num_of_neuron)]

    def results(self, actual_inputs):
        return [node.output(actual_inputs) for node in self.nodes]

    def print_outputs(self, data):
        for a_inputs in data:
            print(self.results(a_inputs))

    def train(self):
        pass

    def predict(self):
        pass


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
    for i in range(test_targets.size):
        value_correct += results_of_predict[i] == test_targets[i]

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

    return train_data_std, test_data_std

    #classifier.train(train_data_std, test_data_std)
    #get_accuracy(classifier.predict(train_data_std, train_target, test_data_std), test_target)
    print ("\n")


def load_file(filename):
    df = pds.read_csv(filename)

    data = df.ix[:, df.columns != "target"]
    target = df.ix[:, df.columns == "target"]

    #labels = df.columns
    #nlabels = labels[:-1]

    return data.values, target.values

def load_dataset(data_set):

    return data_set.data, data_set.target


# Driver of the program - will load data sets and call appropriate
# functions to process the data.
def main(argv):

    # Iris dataset
    data, targets = load_dataset(datasets.load_iris())

    # Pima Dataset
    #data, targets = load_file("pima.csv")

    # Get number of neurons and columns
    num_neurons = int(input("How many Neurons do you want? "))
    num_cols = len(data[0])


    # Make the Network
    neural_net = Network(num_neurons, num_cols)

    train_data, test_data = process_info(data, targets, neural_net)

    # Print the output of the neuron
    neural_net.print_outputs(train_data)



    # for i in range(len(data)):
    #     print(neural_net.nodes[i].set_of_weights)
    #



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