import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models, layers, Input

from collections import Counter

import copy

# read data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Decoding newswires back to text
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# indices are offset because 0,1,2 are reserved for "padding", "start of sequence" and "unknown"
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 45 classes ("topics")
{key: value for key, value in sorted(Counter(train_labels).items(), key= lambda x: x[0])}

# Preparing the data - vectorizing
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# there is built-in way to do that in Keras
one_hot_train_labels_keras = to_categorical(train_labels)
assert one_hot_train_labels.all() == one_hot_train_labels_keras.all()
del one_hot_train_labels_keras

# Building the Network
input_tensor = Input((10000,))
k_model = layers.Dense(64, activation='relu')(input_tensor)
k_model = layers.Dense(64, activation='relu')(k_model)
output_tensor = layers.Dense(46, activation='softmax')(k_model)
model = models.Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# set aside validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# train the network for 20 epochs
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))

# plot learning curves
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Retrain for 9 epochs
model.fit(partial_x_train,
          partial_y_train, epochs=9, batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
results

# compare to a random baseline
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
float(np.sum(hits_array)) / len(test_labels)
# results are ~80% and random is 19% which is pretty good.

# generate predictions for new data
predictions = model.predict(x_test)
predictions[0].shape    # each entry in predictions is a vector of length 46
np.sum(predictions[0])  # sums to 1

np.argmax(predictions[0])   # the largest entry is the predicted class
                            # (entry with the highest probability)

'''
Note about encoding the labels:
If we would have encoded the labels with casting them as integer tensros
and not one-hot encode them, like:
y_train = np.array(train_labels)
then we had to use the 'sparse_categorical_crossentropy' loss
in model.compile(). Mathematically it is the same as 'categorical_crossentropy'
it just has a different interface
'''

# If we had used less hidden units than the 46 categories we have,
# it would have resulted in learning bottlenecks (harm performance):
input_tensor = Input((10000,))
k_model = layers.Dense(64, activation='relu')(input_tensor)
k_model = layers.Dense(4, activation='relu')(k_model)
output_tensor = layers.Dense(46, activation='softmax')(k_model)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=128,
          validation_data=(x_val, y_val))

# trying to represent 46 classes with projecting it to 4-dimensional space is
# just too small of a dimension to do that.

