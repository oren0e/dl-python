'''
In Keras, you use a 1D convnet via the Conv1D layer.
It takes as input 3D tensors with shape (samples, time, features) and returns
similarly shaped 3D tensors. The convolution window is a 1D window on the
temporal axis: axis 1 in the input tensor.

Let's build a simple two-layer 1D convnet and apply it to the IMDB sentiment
classification task.
'''
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt

import os

import numpy as np

from typing import Tuple, Optional


max_features = 10000
max_len = 500

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

input_tensor = layers.Input((max_len,))
'''
in the Embedding layer:
input_dim = num of categories = max_features in our case (the distinct number of
words that we are examining in this problem).
output_dim = embedding_size = 128.
'''
kmodel = layers.Embedding(max_features, 128)(input_tensor)
kmodel = layers.Conv1D(32, 7, activation='relu')(kmodel)
kmodel = layers.MaxPooling1D(5)(kmodel)
kmodel = layers.Conv1D(32, 7, activation='relu')(kmodel)
kmodel = layers.GlobalMaxPooling1D()(kmodel)  # ends with either this or Flatten() to turn the 3D
                                              # inputs into 2D outputs, allowing us to add one or
                                              # more Dense layers to the model for classification
                                              # or regression.
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# plot results
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
print(max(history.history['val_acc']))

# Combining CNNs and RNNs to process long sequences
'''
Using the 1D convnets on the weather forecasting problem won't yield good results
(I didn't include the code here) because the convnet looks for patterns aynwhere
in the input timeseries and has no knowledge of the temporal position of a pattern 
it sees (toward the beginning, toward the end, and so on). Because more recent data
points should be interpreted differently from older data points in the case of this
specific forecasting problem, the convnet fails at producing meaningful results.
This limitation isn't an issue with the IMDB data, because patterns of keywords
associated with a positive or negative sentiment are informative independently
of where they're found in the input sentences.

One strategy to combine the speed and lightness of convnets with the order
sensitivity of RNNs is to use 1D convnet as a preprocessing step before an RNN.
This is especially beneficial when you're dealing with sequences that are so
long they can't realistically be processed with RNNs, such as sequences with 
thousands of steps.

We will use part of the Jena code here.
Because the combination of the convnet and the RNN allows us to manipulate
much longer sequences, we can either look at data from longer ago (by increasing
the `lookback` parameter of the data generator) or look at high-resolution timeseries
(by decreasing the `step` parameter of the generator).
Here, somewhat arbitrarily, we'll use `step` that's half as large, resulting in a
timeseries twice as long, where the temperature data is sampled at a rate of 1 point
per 30 minutes.
'''
data_dir = '/home/corel/python_projects/dl_keras_book/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

# convert the data into numpy array
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]    # drop the timestamp
    float_data[i, :] = values

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data: np.ndarray, lookback: int, delay: int, min_index: int, max_index: Optional[int],
              shuffle: bool = False, batch_size: int = 128, step: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    '''
    :param data: The original array of floating-point data, which we normalized.
    :param lookback: How many timesteps back the input data should go.
    :param delay: How many timesteps in the future the target should go.
    :param min_index: Indices in the data array that delimit which timesteps to draw from.
                      This is useful for keeping a segment of the data for validation and
                      another for testing.
    :param max_index: See min_index.
    :param shuffle: Wether to shuffle the samples or draw them in chronological order.
    :param batch_size: The number of samples per batch.
    :param step: The period, in timesteps, at which you sample the data
    :return: Tuple (samples, targets) where samples is one batch of input data and targets is
             the corresoinding array of target temperatures.
    '''
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 720
step = 3
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000,
                      shuffle=True, step=step)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
                      shuffle=True, step=step)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None,
                      shuffle=True, step=step)

val_steps = (300000 - 200001 - lookback) // batch_size

test_steps = (len(float_data) - 300001 - lookback) // batch_size

input_tensor = layers.Input((None, float_data.shape[-1]))
kmodel = layers.Conv1D(32, 5, activation='relu')(input_tensor)
kmodel = layers.MaxPooling1D(3)(kmodel)
kmodel = layers.Conv1D(32, 5, activation='relu')(kmodel)
kmodel = layers.GRU(32, dropout=0.1, recurrent_dropout=0.5)(kmodel)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)