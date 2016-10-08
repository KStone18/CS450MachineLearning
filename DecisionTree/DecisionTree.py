from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp
from sklearn import preprocessing
from statistics import mode
from scipy import stats

import pandas as pds
import numpy as np
import sys


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


def get_Values(info, col):
    values = []
    for data_point in info:
        if data_point[col] not in values:
            values.append(data_point[col])

    return values


def load_file(filename):
    df = pds.read_csv(filename)

    data = df.ix[:, df.columns != "target"]
    target = df.ix[:, df.columns == "target"]

    labels = df.columns
    nlabels = labels[:-1]

    return data.values, target.values, nlabels


def process_info(data, target, classifier, labels):
    test = float(get_test_size())
    state = int(get_status())

    train_data, test_data, train_target, test_target = tsp(data, target, test_size=test, random_state=state)

    classifier.train(train_data, train_target, labels)

    # get_accuracy(classifier.predict(train_data, train_target, test_data), test_target)
    print("\n")


def find_entropy(data):
    if data != 0:
        return -data * np.log2(data)
    else:
        return 0


def all_same(items):
    return all(x == items[0] for x in items)


def cal_weighted_average(data, class_targ, feature):
    # List of values for a given feature or label
    counted_data = len(data)

    feature_values = get_Values(data, feature)

    attribute_values_count = np.zeros(len(feature_values))
    my_entropy = np.zeros(len(feature_values))

    index_of_val = 0

    for v in feature_values:
        data_index = 0
        new_Classes = []

        for point in data:
            if point[feature] == v:
                attribute_values_count[index_of_val] += 1
                new_Classes.append(class_targ[data_index])

            data_index += 1

        class_val = []
        for nClass in new_Classes:

            if class_val.count(nClass) == 0:
                class_val.append(nClass)

        num_class_values = np.zeros(len(class_val))

        clas_Index = 0

        for cla_val in class_val:
            for nClass in new_Classes:
                if nClass == cla_val:
                    num_class_values[clas_Index] += 1
            clas_Index += 1

        for j in range(len(class_val)):
            my_entropy[index_of_val] += find_entropy(float(num_class_values[j]) / sum(num_class_values))

        weight_of_feature = attribute_values_count[index_of_val] / counted_data
        my_entropy[index_of_val] = (my_entropy[index_of_val] * weight_of_feature)
        index_of_val += 1

        return sum(my_entropy)


# Makes the tree structure
def make_tree(data, classes, col_names):
    num_Of_Col_Labels = len(col_names)
    data_length = len(data)

    class_mode = stats.mode(classes)
    class_mode = class_mode[0]

    if data_length == 0 or num_Of_Col_Labels == 0:
        return class_mode

    elif len(classes[0]) == data_length:
        return classes[0]

    else:
        total_info_gain = np.zeros(num_Of_Col_Labels)

        for name_of_col in range(num_Of_Col_Labels):
            total_info_gain[name_of_col] = cal_weighted_average(data, classes, name_of_col)

        #total_info_gain = np.zeros(num_Of_Col_Labels)

        best_feat = np.argmin(total_info_gain)

        val = get_Values(data, best_feat)

        tree = {col_names[best_feat]: {}}

        n_data = []
        n_class = []

        for my_val in val:
            index = 0
            for d_point in data:
                if d_point[best_feat] == my_val:
                    if best_feat == 0:
                        dat_point = d_point[1:]
                        new_col_names = col_names[1:]
                    elif best_feat == num_Of_Col_Labels:
                        dat_point = d_point[:-1]
                        new_col_names = col_names[:-1]
                    else:
                        dat_point = d_point[:best_feat]
                        if isinstance(dat_point, np.ndarray):
                            dat_point = dat_point.tolist()
                        dat_point.extend(d_point[best_feat + 1:])

                        new_col_names = col_names[:best_feat]
                        new_col_names.append(col_names[best_feat + 1:])

                    n_data.append(dat_point)
                    n_class.append(classes[index])

                index += 1
            subtree = make_tree(n_data, n_class, new_col_names)

            tree[col_names[best_feat]][my_val] = subtree
        return tree
    # print (num_Of_Col_Labels)
    # if all_same(classes):
    #     n = Node(classes[0])
    #     # n.name = classes[0]
    #     return n








# Create the class for node
class Node:
    def __init__(self, name="", child_Node={}):
        self.label = name
        self.childNode = child_Node


# Create the ID3 class
class ID3Classifier:
    # def __init__(self):


    def predict(self, train_data, train_target, test_data):
        pass

    def train(selfs, data_set, train_target, col_names):
        a = make_tree(data_set, train_target, col_names)
        # print("tree")
        print(a)


# Main method: Driver of the program
def main(argv):
    number = -1
    classif = ID3Classifier()
    while number != 1 or number != 2 or number != 3 or number != 0:
        print("Please choose one of the following")
        print("\t0. Type 0 to quit")
        print("\t1. Load from loan.csv file")
        print("\t2. Load from the Iris study")
        # print("\t3. Load from the Breast Cancer Study")

        number = int(input("> "))

        if (number == 1):
            filename = "loan.csv"
            data, target, labels = load_file(filename)

            process_info(data, target, classif, labels)

            # if (number == 2):
            #     dataset2 = datasets.load_iris()
            #     data = dataset2.data
            #     target = dataset2.target

            # process_info(data, target, classif)


if __name__ == "__main__":
    main(sys.argv)
