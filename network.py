# import modules
import numpy as np
import random


class Network(object):
    """
        Use the np.random.randn function to generate Gaussian distributions with mean 0 and
        standard deviation 1. This random initialization gives our stochastic gradient descent
        algorithm a place to start from.
    """

    def __init__(self, sizes):  # 'sizes' contains the number of neurons in the respective layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 'randn': samples from the standard normal
        # distribution
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivative C_x / partial a for the output activations."""
        return output_activations - y

    def feedforward(self, a):
        """Return the output of the network if 'a' is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        :param training_data: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param epochs: the number of epochs to train for
        :param mini_batch_size: the size of mini-batches to use when sampling
        :param eta: the learning rate
        :param test_data: if this option is provided, the network will be evaluated against test data after each epoch,
        and partial progress printed out.
        :return:
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch {}: {}/{}'.format(j, self.evaluate_test(test_data), n_test))
            else:
                print('Epoch {} complete, and training accuracy: {}/{}'.format(j, self.evaluate_train(training_data),
                                                                               len(training_data)))
                print('The Error at epoch {} is {}'.format(j, self.evaluate_cost(training_data)))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using
        back-propagation to a single mini batch.
        :param mini_batch: mini_batch to train for
        :param eta: learning rate
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nable_b, delta_nable_w = self.backprop(x, y)  # gradients
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nable_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nable_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedback
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # delta L
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)  # gradient W_L (the last one between the last hidden layer and
        # output layer)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def evaluate_test(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_train(self, train_data):
        train_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in train_data]
        return sum(int(x == y) for (x, y) in train_results)

    def evaluate_cost(self, train_data):
        cost = np.mean([sum((self.feedforward(x) - y) ** 2) for (x, y) in train_data])
        return cost


# Miscellaneous functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data = list(training_data)
# test_data = list(test_data)
# net = Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
