'''
We try simple RNN model on the IMDB data
'''
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.preprocessing import sequence

from keras import layers, models

max_features = 10000    # number of words to consider as features
maxlen = 500    # cuts of texts after this many words (among the max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

'''
Note that we are providing maxlen to Input() as the input length (sequence length in this case)
because they have all been resized (using padding) to be of length 500.
When the sequences are not of the same length we can pass None to Input() i.e.
Input((None,))
'''

input_tensor = layers.Input((maxlen,))
kmodel = layers.Embedding(max_features, 32)(input_tensor)
kmodel = layers.SimpleRNN(32)(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)

model = models.Model(input_tensor, output_tensor)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# plot results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#### LSTM ####
input_tensor = layers.Input((maxlen,))
kmodel = layers.Embedding(max_features, 32)(input_tensor)
kmodel = layers.LSTM(32)(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)

model = models.Model(input_tensor, output_tensor)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# plot results - LSTM
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy (LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss (LSTM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()