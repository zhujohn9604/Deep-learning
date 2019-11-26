import csv
import nltk
import itertools
import operator
import sys
import pandas as pd
import numpy as np
import RNN

unknown_token = "UNKNOWN_TOKEN"

data = pd.read_csv('reddit-comments-2015-08.csv')
sentences = data['body']
vocabulary_size = 8000
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
# count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


model = RNN.RNN(8000)
model.update_SGD(X_train[:10], y_train[:10], n_epoch=10)
