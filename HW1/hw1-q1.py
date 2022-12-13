#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def relu(x):
	return (np.maximum(0, x))


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        pass

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            #print(x_i.shape, y_i.shape)
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        eta = kwargs.get("learning_rate", 1)
        # print(x_i.shape, y_i.shape, self.W.dot(x_i).shape, self.W[y_i, :].shape)
        y_i_hat = np.argmax(self.W.dot(x_i))
        if y_i_hat != y_i:
            # Perceptron update.
            self.W[y_i, :] += eta * x_i
            self.W[y_i_hat, :] -= eta * x_i


class LogisticRegression(LinearModel):
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        y_hat_i = self.softmax(self.W.dot(x_i))
        y_i_one_hot = np.zeros(y_hat_i.shape)
        y_i_one_hot[y_i] = 1
        self.W += learning_rate * np.multiply(x_i, np.atleast_2d(y_i_one_hot - y_hat_i).T)


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        units = [n_features, hidden_size, n_classes]
        self.weights = [np.random.normal(0.1, 0.01, (units[1], units[0])), 
                        np.random.normal(0.1, 0.01, (units[2], units[1]))]
        self.biases = [np.zeros(units[1]), np.zeros(units[2])]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        z1 = self.weights[0].dot(X) + self.biases[0]
        h1 = relu(z1) #hidden layer 
        # Assume the output layer has no activation.
        z2 = self.weights[1].dot(h1) + self.biases[1]

        return z2, h1

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        gold_labels = y
        accuracy = 0
        for x, y in zip(X, y):
            predicted_labels = self.predict(x)
            for predicted_label in predicted_labels:
                accuracy += np.mean(np.argmax(predicted_label, axis=0) == np.argmax(gold_labels, axis=0))
        print(accuracy/(10000*10000))
        print(accuracy)
        return accuracy/(10000*10000)
    
    def compute_label_probabilities(self, output):
        # softmax transformation.
        probs = np.exp(output) / np.sum(np.exp(output))
        return probs
    
    def backward(self, x, y, output, hiddens, loss_function='cross_entropy'):
        g = relu
        z = output
        if loss_function == 'squared':
            grad_z = z - y  # Grad of loss wrt last z.
        elif loss_function == 'cross_entropy':
        # softmax transformation.
            probs = self.compute_label_probabilities(output)
            grad_z = probs - y  # Grad of loss wrt last z.
        grad_weights = []
        grad_biases = []
        for i in range(1, -1, -1):
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)

            # Gradient of hidden layer below.
            grad_h = self.weights[i].T.dot(grad_z)

            # Gradient of hidden layer below before activation.
            assert(g == relu)
            grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases, learning_rate): 
        for i in range(2):
            self.weights[i] -= learning_rate*grad_weights[i]
            self.biases[i] -= learning_rate*grad_biases[i]

    def train_epoch(self, X, y, learning_rate=0.001, loss_function='cross_entropy'):
        for x, y in zip(X, y):
            output, hiddens = self.predict(x)
            grad_weights, grad_biases = self.backward(x, y, output, hiddens, loss_function=loss_function)
            self.update_parameters(grad_weights, grad_biases, learning_rate=learning_rate)


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
