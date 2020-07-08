import tensorflow.keras as keras
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 2000     # number of words to consider as features
max_len = 500           # cuts of texts after this number of words
                        # (among max_features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

input_tensor = layers.Input((max_len,))
kmodel = layers.Embedding(max_features, 128, name='embed')(input_tensor)
kmodel = layers.Conv1D(32, 7, activation='relu')(kmodel)
kmodel = layers.MaxPooling1D(5)(kmodel)
kmodel = layers.Conv1D(32, 7, activation='relu')(kmodel)
kmodel = layers.GlobalMaxPooling1D()(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
'''
We will launch the training with a Tensorboard callback instance. This callback
will write log events to disk at the specified location.

histogram_freq=1 => records activation histograms every 1 epoch
embeddings_freq=1 => records embedding data every 1 epoch  
'''
callbacks = [
    keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1)
]
