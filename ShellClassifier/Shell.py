from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp

iris = datasets.load_iris()
train_data, test_data, train_target, test_target = tsp(iris.data, iris.target, test_size=.3, random_state=17)

#print(test_data)
#print("end")
#print(test_target)


class HardCoded:
    def train(self, data_set, target_set):
        print("Training...")
        print("Training is now complete, initiate phase 2 of prediction cycle")

    def predict(self, data_set):
        x = []
        for i in data_set:
            x.append(0)

        return x


def get_accuracy(results_of_predict, test_targets):
    value_correct = 0
    for i in range(test_targets.size):
        value_correct += results_of_predict[i] == test_targets[i]

    print("\nPhase 2 completed. Your results are...")
    print("\nCorrectly foretold ", value_correct, " of ", test_targets.size, "\nI was correct ",
          "{0:.2f}% of the time!!!".format(100 * (value_correct / test_targets.size)), sep="")

y = HardCoded()
y.train(train_data, train_target)
get_accuracy(y.predict(test_data), test_target)


