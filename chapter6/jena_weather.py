import os

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

#from keras import models, layers
#from keras.optimizers import RMSprop
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop


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
