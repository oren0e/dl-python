'''
In Keras, you use a 1D convnet via the Conv1D layer.
It takes as input 3D tensors with shape (samples, time, features) and returns
similarly shaped 3D tensors. The convolution window is a 1D window on the
temporal axis: axis 1 in the input tensor.

Let's build a simple two-layer 1D convnet and apply it to the IMDB sentiment
classification task.
'''
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers, models
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

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
kmodel = layers.Embedding(max_features, 128, input_length=max_len)(input_tensor)
kmodel = layers.Conv1D(32, 7, activation='relu')(kmodel)
kmodel = layers.MaxPooling1D(5)(kmodel)
kmodel = layers.Conv1D(32, 7, activation='relu')(kmodel)
kmodel = layers.GlobalMaxPooling1D()(kmodel)  # ends with either this or Flatten() to turn the 3D
                                              # inputs into 2D outputs, allowing us to add one or
                                              # more Dense layers to the model for classification
                                              # or regression.
output_tensor = layers.Dense(1)(kmodel)
model = models.Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# plot results
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