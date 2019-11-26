import numpy as np
import network


# one-hot code the results / labels
def one_hot_code(j):
    lable = np.zeros((2, 1))
    lable[j] = 1.0
    return lable


x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
x = [np.reshape(i, [2, 1]) for i in x]
y = [0, 1, 1, 1]
y = [one_hot_code(j) for j in y]


data = zip(x, y)

data = list(data)
net = network.Network([2, 2, 2])
net.SGD(data, 1000, 4, 1)
