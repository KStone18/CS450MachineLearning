from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp

iris = datasets.load_iris()


def get_test_size():
    test_size = float(input("Choose a test size between 0.1 - 0.5: "))

    while test_size <= 0.0 or test_size > 0.5:
        test_size = float(input("Please choose a test size between 0.1 and 0.5: "))

    return test_size


def get_status():
    state = int(input("Please choose how many times to shuffle: "))

    while state <= 0:
        state = int(input("Please choose a valid number times. Valid numbers are those greater than Zero: "))

    return state

# train_data, test_data, train_target, test_target = tsp(iris.data, iris.target, test_size, random_state=17)


class HardCoded:


    def train(self, data_set, target_set):
        print("\nTraining... ")
        print("Training is now complete, initiate phase 2 of prediction cycle.")


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


def play_game():
    test = float(get_test_size())
    state = int(get_status())
    train_data, test_data, train_target, test_target = tsp(iris.data, iris.target, test_size = test,
                                                           random_state = state)

    y = HardCoded()

    y.train(train_data, train_target)
    get_accuracy(y.predict(test_data), test_target)


play_game()