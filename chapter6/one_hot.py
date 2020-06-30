import numpy as np

from typing import List, Dict

import string

from keras.preprocessing.text import Tokenizer
'''
Word-level one-hot encoding (toy example)
'''
samples: List[str] = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index: Dict[str, int] = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
print(token_index)

max_length = 10     # only consdier the fisrt max_length words in each sample
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1

results


'''
Character-level one-hot encoding (toy example)
Fixed some issues with the printed code
'''
characters = string.printable   # all printable ASCII characters
token_index = dict(zip(characters, range(1, len(characters) + 1)))
print(token_index)

max_length = 50
results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1

results

'''
Using Keras for word-level one-hot encoding
'''
tokenizer = Tokenizer(num_words=1000)   # only consider the 1000 most frequent words
tokenizer.fit_on_texts(samples)     # build the word index
sequences = tokenizer.texts_to_sequences(samples)   # turn strings into lists of integer indices
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# we can recover the word index that was computed
word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens')

'''
Word-level one-hot encoding with hashing trick.
Instead of explicitly assigning an index to each word and keeping a reference of these indices
in a dictionary, we can has words into vectors of fixed size.
We will typically use this trick when the number of unique tokens in our vocabulary is too
large to handle explicitly.
'''
dimensionality = 1000   # stores the words as vectors of size 1000. If we have close to 1000
                        # words (or more) we'll see many collisions (load factor closer to 1)
                        # load factor = num of words / num of buckets (i.e., dimensionlaity)
max_length = 10

results = np.zeros(shape=(len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1