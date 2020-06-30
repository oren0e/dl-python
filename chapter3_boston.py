from keras.datasets import boston_housing
from keras import models, layers, Input

import numpy as np
import matplotlib.pyplot as plt

from keras_tqdm import TQDMCallback
from tqdm import tqdm

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Preparing the data
# Normalizing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Building the Network
def build_model() -> models.Model:
    input_tensor = Input((train_data.shape[1],))
    kmodel = layers.Dense(64, activation='relu')(input_tensor)
    kmodel = layers.Dense(64, activation='relu')(kmodel)
    output_tensor = layers.Dense(1)(kmodel)

    model = models.Model(input_tensor, output_tensor)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K-Fold CV
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

for i in tqdm(range(k)):
    print(f'processing fold {i}')
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i+1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i+1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
              validation_data=(val_data, val_targets),
              epochs=num_epochs, batch_size=1, verbose=0, callbacks=[TQDMCallback()])
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

# compute average per-epoch MAE
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

'''
To improve plot readability we will:
1. Omit the first 10 data points which are on a different scale
2. Replace each point with exponential moving average of the
previous points, to obtain a smooth curve.
'''
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# lowest MAE
np.argmin(smooth_mae_history)   # 37

# Build final model on all train data and compare with test
model = build_model()
model.fit(train_data, train_targets, epochs=37, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score  # I am off by $3,067!
