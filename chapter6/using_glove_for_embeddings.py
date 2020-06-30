import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers, models, Input

import numpy as np

import matplotlib.pyplot as plt

# reading the data
imdb_dir = '/home/corel/python_projects/dl_keras_book/imdb_data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# Tokenizing the data
'''
We will restrict the training size to be the first 200 samples to simulate
a situation where we have little training data available, because that's when 
using pre-trained word embeddings helps the most. In situations where we have enough
training data, learning the embeddings ourselves is a better option.
'''
maxlen = 100    # cuts off reviews after 100 words
training_samples = 200
validation_samples = 10000
max_words = 10000   # consider only the top 10000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens')

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
data.shape
labels.shape

# split to train and validation, but shuffle first
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# Parsing the GloVe word-embeddings file
glove_dir = '/home/corel/python_projects/dl_keras_book/glove_6b'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
del f
#print(f'Found {len(embeddings_index)} word vectors in GloVe')

'''
Next, we will build a matrix of (max_words x embedding_dim) that we can load 
into an Embedding layer. Each entry (row) i contains the embedding_dim-dimensional
vector for the word of index i in the reference word index (built during tokenization).
Note that index 0 isn't supposed to stand for any word or token - it's a placeholder.
'''
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Define a model
input_tensor = Input((maxlen,))
kmodel = layers.Embedding(max_words, embedding_dim, input_length=maxlen)(input_tensor)
kmodel = layers.Flatten()(kmodel)
kmodel = layers.Dense(32, activation='relu')(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)
model.summary()

# loading glove matrix into the model
model.layers[1].set_weights([embedding_matrix])
model.layers[1].trainable = False   # freeze the embedding layer from learning new weights

# compile and train the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# plots
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
plt.show()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

'''
The model quickly overfits given that we have so few training samples.
We can try and train the model with the embedding layer ourselves. This probably won't
help because we have so few training data points.
'''
input_tensor = Input((maxlen,))
kmodel = layers.Embedding(max_words, embedding_dim, input_length=maxlen)(input_tensor)
kmodel = layers.Flatten()(kmodel)
kmodel = layers.Dense(32, activation='relu')(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# plots
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
plt.show()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

'''
Validation accuracy stalls at the 50s so in this case the pre-trained approach was a little
better. If we will increase the number of training samples this will quickly stop being the case!
'''

# evaluate on test data, first tokenize the test data
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)