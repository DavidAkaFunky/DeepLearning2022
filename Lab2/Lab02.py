from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# PART 1

def part1():
    data = load_digits()

    print(data.DESCR)
    plt.gray()
    for i in range(10):
        plt.matshow(data.images[i])
        plt.show()

    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# PART 2

def part2(X_train, X_test, y_train, y_test):
    pass

# PART 3

def part3(X_train, X_test, y_train, y_test):
    clf = Perceptron(fit_intercept=False, shuffle=False)
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = part1()
    part3(X_train, X_test, y_train, y_test)