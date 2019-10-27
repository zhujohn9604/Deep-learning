import numpy as np


# Miscellaneous
def softmax(x):
    return np.exp(x) / sum(np.exp(x))


class RNN(object):

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """
        :param word_dim: the number of units in input layer
        :param hidden_dim: the number of units in hidden layer
        :param bptt_truncate: bptt params
        Notice: for language model, the input_dim = output_dim
        """
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.randn(hidden_dim, hidden_dim)
        self.W = np.random.randn(hidden_dim, word_dim)
        self.V = np.random.randn(word_dim, hidden_dim)

    def forward_propagation(self, x):
        T = len(x)  # the total number of time steps
        h = np.zeros((T + 1, self.hidden_dim))  # hidden layer
        h[-1] = np.zeros(self.hidden_dim)
        y_hat = np.zeros((T, self.word_dim))  # T output layers
        for t in np.arange(T):
            h[t] = np.tanh(self.W[:, x[t]] + self.U.dot(h[t - 1]))
            y_hat[t] = softmax(self.V.dot(h[t]))
        return [h, y_hat]

    def predict(self, x):
        h, y_hat = self.forward_propagation(x)
        return np.argmax(y_hat, axis=1)  # horizontal

    def calculate_loss(self, x, y):
        L = 0 # total cost
        for i in np.arange(len(y)):  # each sentence
            h, y_hat = self.forward_propagation(x[i])
            correct_word_predictions = y_hat[np.arange(len(y[i])), y[i]]
            L += -1 * sum(np.log(correct_word_predictions))
        N = sum([len(y_i) for y_i in y])
        return L / N

    def bptt(self, x, y):
        """
        Perform Back-propagation through time: all parameters (V, W, U) are shared by all time steps
        :param x: input x
        :param y: label y
        :return: dLdU, dLdV, dLdW
        Comments:
        1. gradient of W: caused by the flow of information through the hidden layers (indirectly)
        2. gradient of U: caused by the flow of information through the hideen layers (directly)
        """
        T = len(y)
        h, y_hat = self.forward_propagation(x)  # h: sent_dim(T) * hidden_dim; y_hat: sent_dim / #timesteps(T) *
        # word_dim(8000)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_y_hat = y_hat  # #timesteps * word_dim(8000)
        delta_y_hat[np.arange(len(y)), y] -= 1  # dL/dy_hat * dy_hat/dz = y_hat - 1
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_y_hat[t], h[t]) # word_dim * hidden_dim
            delta_t = self.V.T.dot(delta_y_hat[t]) * (1 - (h[t] ** 2))  # hidden_dim * 1
            for i in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
                dLdU += np.outer(delta_t, h[i-1])
                dLdW[:, x[i]] += delta_t  # delta_t * x^t.T & hidden_dim * word_dim
                delta_t = self.U.T.dot(delta_t) * (1 - (h[i] ** 2))
        return [dLdU, dLdV, dLdW]

    def SGD_step(self, x, y, eta):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= eta * dLdU
        self.V -= eta * dLdV
        self.W -= eta * dLdW

    def update_SGD(self, X_train, y_train, eta=0.005, n_epoch=100):
        for epoch in range(n_epoch):
            data_train_zip = zip(X_train, y_train)
            data_train = [(x, y) for x, y in data_train_zip]
            np.random.shuffle(data_train)
            X_train_shuffled = [x for x, y in data_train]
            y_train_shuffled = [y for x, y in data_train]
            for i in range(len(y_train_shuffled)):
                self.SGD_step(X_train_shuffled[i], y_train_shuffled[i], eta)
            print('Epoch {} complete, and cost is {}.'.format(epoch, self.calculate_loss(X_train_shuffled, y_train_shuffled)))


"""
    def gradient_check(self, x, y, h = 0,001, error_threshold = 0,01):
        
        When implement back-propagation, we use gradient checking to help us make sure that what we are
        doing is correct. The idea behind this is that the derivative of a parameter is equal to the slope
        at the point (definition of derivative). One thing is this checking is very expensive, so we had
        better to first test on a sample set.
        :param h: h in definition of derivative
        :param error_threshold: self-explanatory
        :return:
        
        btpp_gradients = self.bptt(x, y)
        model
"""