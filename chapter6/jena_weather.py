import os

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

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

temp = float_data[:, 1]  # temperature (in degrees celsius)
plt.plot(range(len(temp)), temp)
plt.show()

# first 10 days (temp is recorded every 10 minutes))
plt.plot(range(1440), temp[:1440])
plt.show()

'''
The formulation of the problem is this:
Given data going as far back as _lookback_ timesteps (a timestep is 10 minutes) and 
sampled every _steps_ timesteps, can we predict the temperature in _delay_ timesteps?
We will use these values:
_lookback_ = 720 (observations will go back 5 days)
_steps_ = 6 (observations will be sampled at one data point per hour)
_delay_ = 144 (targets will be 24 hours in the future)

We will need to:
1. Normalize the data because each variable has different scale
2. Write a python generator that takes the current array of float data and
yields batches of data from the recent past, along with a target temperature in
the future. Because the samples in the dataset are highly redundant (sample N and sample
N+1 will have most of their timesteps in common because the temperatures do not differ that much),
it would be wasteful to explicitly allocate every sample. Instead, we'll generate the samples on the
fly using the original data (that's the purpose of the generator).

First 200000 timesteps are the training data  
'''
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


def generator(data: np.ndarray, lookback: int, delay: int, min_index: int, max_index: int,
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

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
                      shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None,
                      shuffle=True, step=step, batch_size=batch_size)

# these are the forecasted parts that we will want to look at
val_steps = (300000 - 200001 - lookback) // batch_size   # how many steps to draw from val_gen in order to see
                                            # the entire validation set
test_steps = (len(float_data) - 300001 - lookback) // batch_size     # " " for test

'''
Naive baseline non-ml model to beat:
Always predict that the temperature 24 hours from now will be equal to the temperature right now.
We will evaluate this approach using the MAE (mean absolute error) metric
'''
def evaluate_naive_method() -> None:
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

#evaluate_naive_method()
# convert the MAE back to Celsius error (we need to do that because we had normalized all variables before)
celsius_mae = 0.29 * std[1]     # that's a fairly large average absolute error. Now it is up to us
                                # to do better!

'''
In the same way that it's useful to establish a common-sense non-ml baseline before
trying ml approaches, it's useful to try simple, cheap machine learning models
(such as small, densely connected networks) before looking into complicated and
computationally expensive models such as RNNs. This is the best way to make sure any
further complexity we throw at the problem is legitimate and delivers real benefits.

The following shows a fully connected model that starts by flattening the data and then 
runs it through two Dense layers. Note the lack of activation function on the last Dense
layer, which is typical for a regression problem.
'''
input_tensor = layers.Input((lookback // step, float_data.shape[-1]))
kmodel = layers.Flatten()(input_tensor)
kmodel = layers.Dense(32, activation='relu')(kmodel)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
                              epochs=20, validation_data=val_gen,
                              validation_steps=val_steps)

# plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validatio Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# A first recurrent baseline
'''
The first fully connected approach didn't do well. It flattened the timeseires
which removed the notion of time from the input data. Let's instead look at the data as
a sequence where we can exploit the temporal ordernig of data points.
We will usr a GRU layer, which is similar to LSTM but is cheaper to run.
'''
input_tensor = layers.Input((None, float_data.shape[-1]))
#kmodel = layers.GRU(32)(input_tensor)
kmodel = layers.GRU(32)(input_tensor)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)

# plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
min(val_loss) * std[1]  # = 2.3, where the non-ml baseline was 2.56. Nice improvement due to ML model.

# Using recurrent dropout to fight overfitting
'''
Let's add dropout and recurrent_dropout.
dropout = a float specifying the dropout rate for input units of the layer.
recurrent_dropout = specifying the dropout rate of the recurrent units.
Because networks being regularized with dropout always take longer to fully converege,
we'll train the network for twice as many epochs.
'''
input_tensor = layers.Input((None, float_data.shape[-1]))
kmodel = layers.GRU(32, recurrent_dropout=0.2, dropout=0.2)(input_tensor)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
'''
Note: Because recurrent_dropout does not seem to be implemented in the cuda version of the layers
the performance is very slow thus I won't train it here. For further details see these issues:
https://github.com/tensorflow/tensorflow/issues/40944
https://github.com/keras-team/keras/issues/8935
'''

# Stacking recurrent layers
'''
Because we are no longer overfitting (as a result of using the dropout) but seem to have hit
a performance bottleneck, we should consider increasing the capacity of the network. Recall
the universal description of the machine learning workflow: It's generally a good idea to increase
the capacity of your network until overfitting becomes the primary obstacle. As long as you aren't
overfitting too badly, you're likely under capacity.

Increasing the network capacity is typically done either by increasing the number of units
in the layers or adding more layers.
To stack recurrent layers on top of each other in Keras, all intermediate layers should
return their full sequence of outputs (a 3D tensor) rather than their output at the last
timestep. This is done by specifying `return_sequence` = True.  
'''
input_tensor = layers.Input((None, float_data.shape[-1]))
kmodel = layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)(input_tensor)
kmodel = layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5)(kmodel)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)
'''
The model won't be trained here for the exact same reasons as the last version - 
lack of support of cuda version for recurrent_dropout in GRU

But from the results (in the book) we can draw 2 conclusions:
1. Because we are still not overfitting too badly, we could safely increase the size
of the layers in a quest for better validation-loss improvement. This has a non-negligible
computational cost, though.
2. Adding a layer didn't help by a significant factor, so you may be seeing diminishing returns
from increasing network capacity at this point.
'''

# Using bidirectional RNNs
'''
A bidirectional RNN exploits the order sensitivity of RNNs: It consists of using two regular
RNNs, such as GRU and LSTM layers, each of which processes the input sequence in one direction
(chronologically and antichronologically), and then merging their representations.
Could the RNNs have performed well enough if they processed input sequences in antichronological
order (newer timesteps first)? Let's try this in practice and see what happens.

All we need to do is write a variant of the data generator where the input sequences are reverted
along the time dimension (replace the last line with `yield samples[:, ::-1, :], targets)
'''
def generator_backwards(data: np.ndarray, lookback: int, delay: int, min_index: int, max_index: int,
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
        yield samples[:, ::-1, :], targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator_backwards(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator_backwards(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
                      shuffle=True, step=step, batch_size=batch_size)
test_gen = generator_backwards(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None,
                      shuffle=True, step=step, batch_size=batch_size)

# these are the forecasted parts that we will want to look at
val_steps = (300000 - 200001 - lookback) // batch_size   # how many steps to draw from val_gen in order to see
                                            # the entire validation set
test_steps = (len(float_data) - 300001 - lookback) // batch_size     # " " for test

input_tensor = layers.Input((None, float_data.shape[-1]))
#kmodel = layers.GRU(32)(input_tensor)
kmodel = layers.GRU(32)(input_tensor)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)

# plot results
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
min(history.history['val_loss'])

'''
The reversed GRU strongly underperforms even the common-sense baseline, indicating that in this
case, chronological processing is important to the success of our approach. This make perfect sense:
The GRU layer will typically be better at remembering the recent past than the distant past, and naturally
the more recent weather data points are more predictive than older data points for the problem (that's what
makes our common-sense baseline fairly strong). This isn't always the case in other problems, including 
natural language: Intuitively, the importance of a word in understanding a sentence isn't usually 
dependent on its position in the sentence.
Let's try the same trick on the LSTM IMDB example
'''
max_features = 10000    # number of words to consider as features
maxlen = 500    # cuts off texts after this number of words (among the max_features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# reverse sequences
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

input_tensor = layers.Input((maxlen,))
kmodel = layers.Embedding(max_features, 128)(input_tensor)
kmodel = layers.LSTM(32)(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
max(history.history['val_acc'])
'''
We get nearly the same performance as when we used the non-reversed order, thus
validating our assumption that in natural language - the order does matter but
which order - does not matter.

To instantiate a bidirectional RNN in Keras we use the `bidirectional` layer,
which takes as its first argument a recurrent layer instance. It creates a second,
separate instance of this recurrent layer and uses one instance for processing the input
sequences in chronological order and the other instance for processing the input sequences
in reversed order. Let's try that on the IMDB sentiment analysis task:
'''
max_features = 10000    # number of words to consider as features
maxlen = 500    # cuts off texts after this number of words (among the max_features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

input_tensor = layers.Input((maxlen,))
kmodel = layers.Embedding(max_features, 32)(input_tensor)
kmodel = layers.Bidirectional(layers.LSTM(32))(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
max(history.history['val_acc'])

# plot
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

'''
We see that it performs slightly better than the regular LSTM we have tried in the 
previous section. It also seems to overfit more quickly, which is unsurprising because
bidirectional layer has twice as many parameters as a chronological LSTM. With some
regularization, the bidirectional approach would likely be a strong performer on this task.

Now let's try the same approach on the temperature task:
'''
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
                      shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None,
                      shuffle=True, step=step, batch_size=batch_size)

# these are the forecasted parts that we will want to look at
val_steps = (300000 - 200001 - lookback) // batch_size   # how many steps to draw from val_gen in order to see
                                            # the entire validation set
test_steps = (len(float_data) - 300001 - lookback) // batch_size     # " " for test

input_tensor = layers.Input((None, float_data.shape[-1]))
kmodel = layers.Bidirectional(layers.GRU(32))(input_tensor)
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)
min(history.history['val_loss']) * std[1]


'''
This performs about as well as the regular GRU. It's easy to understand why:
All the predictive power must come from the chronological half of the network,
because the antichronological half is known to be severely underperforming on this
task (again, because the recent past matters much more than the distant past in this case).

There are more things we could try in order to improve the performance on the
temperature-forecasting task:
- Adjust the number of units in each recurrent layer in the stacked setup. The current choices
are largely arbitrary and thus probably suboptimal.
- Adjust the learning rate for the RMSprop() optimizer.
- Try using LSTM layers instead of GRU layers.
- Try using a bigger densely connected regressor on top of the recurrent layers:
that is, a bigger Dense layer or even a stack of Dense layers.
- Don't forget to eventually run the best performing models (in terms of validation MAE)
on the test set! Otherwise, you'll develop architectures that are overfitting to the 
validation set.
'''