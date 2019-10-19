# Deep-learning

# Reference List:

1. Neural networks

URL: http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits

This is a good source for learning neural betworks, and strongly recommended. And, for python 3.X users, load the mnist dataset using mnist_loader.py (modified).

A few important points when running the network.py program:

1. The label of training data has to be pre-processed with one hot encoding.

2. The label of test data should keep in the original format. 

(e.g., for a multi-classification problem with 9 classes, say, the 1st data instance belongs to class 5. Then, if it is in the training set, the label should be encoded into [0, 0, 0, 0, 1, 0, 0, 0, 0]). If it is in the test set, we just keep it as 5.
