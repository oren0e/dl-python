'''
We will work with the IMDB movie reviews data. We will:
1. Prepare the data
2. restrict the movie reviews to the top 10000 most common words
3. Cut off the reviews after only 20 words.
4. The network will learn 8-dimensional embeddings for each of the 10000 words,
turn the input integer sequences (2D integer tensor) into embedded sequences
(3D float tensor of (samples, sequence_length, embedding_size)),
flatten the tensor to 2D, and train a single Dense layer on top for classification.
'''
from keras.datasets import imdb
from keras import preprocessing
from keras import models, layers, Input

max_features = 10000    # Number of words to consider as features
maxlen = 20             # Cuts off the text after this number of words
                        # (among the max_features most common words)

# read data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# turn the lists of integers into a 2D integer tensor of shape (samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

input_tensor = Input((maxlen,))
kmodel = layers.Embedding(10000, 8, input_length=maxlen)(input_tensor)
kmodel = layers.Flatten()(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)

model = models.Model(input_tensor, output_tensor)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

