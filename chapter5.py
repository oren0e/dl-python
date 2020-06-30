from keras import models, layers, Input
from keras.datasets import mnist
from keras.utils import to_categorical

# MNIST convnet example
'''
A convnet takes input shape (image height, image width, image channels).
This is not including the "first" batch size dimension.
'''
input_tensor = Input((28, 28, 1))
kmodel = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(64, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(64, (3, 3), activation='relu')(kmodel)
# now adding dense layers
kmodel = layers.Flatten()(kmodel)
kmodel = layers.Dense(64, activation='relu')(kmodel)
output_tensor = layers.Dense(10, activation='softmax')(kmodel)
'''
The Dense() layers process vectors which are 1D, whereas here, before
adding the dense layers, the current output is a 3D tensor.
So, first we have flattened the 3D outputs to 1D, and then added
a few dense layers on top.
'''

model = models.Model(input_tensor, output_tensor)
model.summary()

# Training the convnet on MNIST images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc