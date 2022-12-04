from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Compute accuracy of predicted labels.
def evaluate(predicted_labels, gold_labels):
    accuracy = np.mean(predicted_labels == gold_labels)
    return accuracy

# One epoch of the multi-class perceptron algorithm.
def multi_class_perceptron_epoch(inputs, labels, W, eta=1):
    mistakes = 0
    for x, y in zip(inputs, labels):
        # Sign function.
        y_hat = np.argmax(W.dot(x))
        if y_hat != y:
            mistakes += 1
            # Perceptron update.
            W[y, :] += eta * x
            W[y_hat, :] -= eta * x
    print("Mistakes: %d" % mistakes)

def multi_class_classify(inputs, W):
    predicted_labels = []
    for x in inputs:
        y_hat = np.argmax(W.dot(x))
        predicted_labels.append(y_hat)
    predicted_labels = np.array(predicted_labels)
    return predicted_labels

data = load_digits()

inputs = data.data  # num_examples x num_features
labels = data.target  # num_examples x num_labels

num_examples, num_features = np.shape(inputs)
num_labels = np.max(labels)+1  # labels are 0, 1, ..., num_labels-1

# Augment points with a dimension for the bias.
inputs = np.concatenate([np.ones((num_examples, 1)), inputs], axis=1)

print(inputs)
print(labels)

print(data.DESCR)

plt.gray()
for i in range(10):
    plt.matshow(data.images[i])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Initialize all weights to 0 (including the bias)
W = np.zeros((num_labels, num_features+1))  # num_labels x (num_features + 1)

# Learning rate.
eta = 1  
    
# Run 100 epochs of perceptron.
train_accuracies = []
test_accuracies = []
for epoch in range(100):
    print("Epoch %d" % (epoch + 1))

    # Run 1 epoch of training.
    multi_class_perceptron_epoch(X_train, y_train, W, eta)
    
    # Predict on training set and evaluate.
    predicted_labels = multi_class_classify(X_train, W)
    accuracy = evaluate(predicted_labels, y_train)
    print("Accuracy (training set): %f" % accuracy)
    train_accuracies.append(accuracy)
    
    # Predict on test set and evaluate.
    predicted_labels = multi_class_classify(X_test, W)
    accuracy = evaluate(predicted_labels, y_test)
    print("Accuracy (test set): %f\n" % accuracy)
    test_accuracies.append(accuracy)
    
# Plot train and test accuracies as a function of number of epochs.
plt.plot(range(100), train_accuracies, 'b-', label='train acc')
plt.plot(range(100), test_accuracies, 'r-', label='test acc')
plt.legend()
plt.savefig('ola.png')


clf = Perceptron(fit_intercept=False, shuffle=False)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
