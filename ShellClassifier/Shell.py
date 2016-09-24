from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp
from sklearn.neighbors import KNeighborsClassifier as knclass
from sklearn import preprocessing

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




#classifier = knclass(n_neighbors=3)
#classifier.fit(train_data, train_target)
#predictions = classifier.predict(test_data)
# train_data, test_data, train_target, test_target = tsp(iris.data, iris.target, test_size, random_state=17)

# CLASS: HardCoded
class HardCoded:


    def train(self, data_set, target_set):
        print("\nTraining... ")
        print("Training is now complete, initiate phase 2 of prediction cycle.")

        #Standardizes each value and stores it


    def predict(self, data_set):
        x = []
        for i in data_set:
            x.append(0)

        return x

#CLASS: KnnClassifier
class KnnClassifier:
    def __init__(self):
        self.k = None
        self.myData = None
        self.myTData = None

    def predict(self, k, train_data, train_target, test_data):

        nInputs = np.shape(test_data)[0]
        neighbors = np.zeros(nInputs)

        for n in range(nInputs):
            # Compute the difference
            distances = np.sum((train_data-test_data[n,:])**2, axis=1)

            #Indentify the nearest neighbours
            indices = np.argsort(distances,axis=0)

            classes = np.unique(train_target[indices[:k]])

            if len(classes) == 1:
                neighbors[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[train_target[indices[i]]] += 1
                neighbors[n] = np.max(counts)

        return neighbors



    # Train Function
    def train(self, data_set, test_data):
        self.myData = np.asarray(data_set)
        self.myTData = np.asarray(test_data)


# Calculates the accuracy
def get_accuracy(results_of_predict, test_targets):
    value_correct = 0
    for i in range(test_targets.size):
        value_correct += results_of_predict[i] == test_targets[i]

    print("\nCorrectly foretold ", value_correct, " of ", test_targets.size, "\nI was correct ",
          "{0:.2f}% of the time!!!".format(100 * (value_correct / test_targets.size)), sep="")

# Gets the number of neighbors from user
def get_k_value():
    k = 0
    while k <= 0:
        k = int(input("Please enter how many neighbors to find: "))

    return k

# Normalizes the data to prevent skewness
def standardize(x_data, y_data):
    std_scale = preprocessing.StandardScaler().fit(x_data)
    train_data = std_scale.transform(x_data)
    test_data = std_scale.transform(y_data)

    return (train_data, test_data)

# Process the datasets and splits the data.
def process_info(dataset, classifier):
    test = float(get_test_size())
    state = int(get_status())
    k = int(get_k_value())
    train_data, test_data, train_target, test_target = tsp(dataset.data, dataset.target, test_size=test,
                                                           random_state=state)
    #print ("\n\n", standardize(train_data, test_data))

    train_data_std, test_data_std = standardize(train_data, test_data)
    classifier.train(train_data_std, test_data_std)
    get_accuracy(classifier.predict(k, train_data_std, train_target, test_data_std), test_target)


# Driver of the program - will load data sets and call appropriate
# functions to process the data.
def main(argv):
    #dataset = pds.read_csv("cars3.csv")
    dataset = datasets.load_iris()

    classif = KnnClassifier()

    #print (dataset)

    process_info(dataset, classif)



#get_accuracy(predictions, test_target)

if __name__ == "__main__":
    main(sys.argv)