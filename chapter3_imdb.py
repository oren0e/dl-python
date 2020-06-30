from keras.datasets import imdb
import numpy as np

from typing import Dict

from collections import Counter

from keras import models, layers, Input

import matplotlib.pyplot as plt

# read data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

Counter(train_labels)
max([max(sequence) for sequence in train_data])     # no integer is bigger than 10000

word_index: Dict = imdb.get_word_index()  # a dictionary mapping words to an integer index
{key: value for key, value in sorted(word_index.items(), key=lambda x: x[1])}   # sorted by value

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# the indices are offset by 3 because 0, 1, and 2 are reserved indices for "padding",
# "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i -3, '?') for i in train_data[0]])
decoded_review

# encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# understanding the function above. sequence is like a mask
temp_mat = np.zeros((10, 5))
seq0 = [1, 2, 4]
temp_mat[0, seq0] = 1
temp_mat

# vectorize labels as well
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## Building the Network ## (using Functional API)
input_tensor = Input(shape=(10000,))    # the first dim is omitted because it is the batch size
                                        # and the model should be able to deal with any batch size
                                        # so in this case the second dimension is the number of columns
                                        # (the first here, i.e., 10000)
model = layers.Dense(16, activation='relu')(input_tensor)
model = layers.Dense(16, activation='relu')(model)
output_tensor = layers.Dense(1, activation='sigmoid')(model)

final_model = models.Model(input_tensor, output_tensor)
final_model.summary()

# setting aside a validation set
x_val = x_train[:10000] # out of 25000
partial_x_train = x_train[10000:]   # notice how it does not change the configuration of our
                                    # model since it can deal with any batch size!
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

final_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = final_model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                          validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# plot the loss and accuracy (learning curves)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Retrain the model from scratch with 4 epochs
final_model.fit(x_train, y_train, epochs=4, batch_size=512)
results = final_model.evaluate(x_test, y_test)
results

# Using a trained network to make predictions on new data
probs = final_model.predict(x_test)
plt.hist(probs, bins=30, color='brown', edgecolor='black')
plt.show()