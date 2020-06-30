import os
import shutil

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from keras import layers, models, Input
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image   # module with image preprocessing utilities
from keras.applications import VGG16    # pre-trained model


original_dataset_dir = '/home/corel/python_projects/dl_keras_book/cats_dogs_original'
base_dir = '/home/corel/python_projects/dl_keras_book/cats_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

fnames = [f'cat.{i}.jpg' for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'dog.{i}.jpg' for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

# sanity check
len(os.listdir(train_cats_dir))
len(os.listdir(train_dogs_dir))
len(os.listdir(validation_cats_dir))
len(os.listdir(validation_dogs_dir))
len(os.listdir(test_cats_dir))
len(os.listdir(test_dogs_dir))

#### Modeling ####
input_tensor = Input((150, 150, 3))
kmodel = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(64, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(128, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(128, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Flatten()(kmodel)
kmodel = layers.Dense(512, activation='relu')(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)

model = models.Model(input_tensor, output_tensor)
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# pre-processing
train_datagen = ImageDataGenerator(rescale=1./255)  # rescale images by 1/255
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),  # resize all images to 150x150 pixels
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size=(150, 150),  # resize all images to 150x150 pixels
                                                    batch_size=20,
                                                    class_mode='binary')

# let's look at one of these generators
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

'''
steps per epoch tells the model when to stop taking data from the generator
which will yield data batches endlessly. if we have 20 samples in a batch then
it will take 100 steps to cover all of the 2000 training examples (cats and dogs).
If you pass generator as validation data then this generator is expected to yield
data endlessly. Thus you should also specify the validation_steps argument which
says how many batches to draw from the validation generator for evaluation.
In our case its 50 x 20 (batch_size=20) = 1000 (500 for each class)
'''
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)
model.save('cats_and_dogs_small_1.h5')

# plot the performance of the model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Data Augmentation (used a lot in computer-vision!)
'''
This technique augments existing training samples via a number of
random transformations that yields believable-looking images.
'''
# define number of random image transformations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]    # choose one image to augment
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)  # converts it to Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)   # reshapes it to (1, 150, 150, 3)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# adding a Dropout layer to further fight overfitting
input_tensor = Input((150, 150, 3))
kmodel = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(64, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(128, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Conv2D(128, (3, 3), activation='relu')(kmodel)
kmodel = layers.MaxPooling2D((2, 2))(kmodel)
kmodel = layers.Flatten()(kmodel)
kmodel = layers.Dropout(0.5)(kmodel)
kmodel = layers.Dense(512, activation='relu')(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)

model = models.Model(input_tensor, output_tensor)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# training the convnets using data-augmentation generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
# the validation data should not be augmented
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(150, 150),  # resizes all images to 150x150 pixels
    batch_size=32,
    class_mode='binary')     # because we use binary_crossentropy loss

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)
model.save('cats_and_dogs_small_2.h5')

# plot results again
loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

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

'''
Thanks to data augmentation and Dropout layer you are no longer
overfitting. The training curves are closely tracking the validation curves.
'''

## Using a pre-trained model ##
conv_base: models.Model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
'''
At this stage there are two ways you can proceed:
1. Running the convolutional base over your dataset, recording the output of this model
to a Numpy array on disk, and then using this data as input to a standalone, densely connected
classifier. This is cheap because it requires running the convolutional base only once and this is
the most expensive part of the model chain. On the other hand, for this exact reason it won't allow
you to use data augmentation.

2. Extending the model conv_base by adding Dense layers on top of it, and running the whole
thing end to end with your input data. This will allow to use data augmentation because every 
input image goes through the convolutional base every time it's seen by the model. But for this 
reason this option is much more expensive than the first one.
'''

### First option - fast feature extraction without data augmentation ###
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extract_features(directory: str, sample_count: int) -> Tuple[np.ndarray, np.ndarray]:
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break   # because it is a generator we must stop it manually
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# we'll feed these to a densely connected layer.
# first we must flatten those out to (samples, 4*4*512)
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

# at this point we can create our densely connected classifier
# and train it on the data and labels that we just recorded
input_tensor = Input((4*4*512,))
kmodel = layers.Dense(256, activation='relu')(input_tensor)
kmodel = layers.Dropout(0.5)(kmodel)
output_tensor = layers.Dense(1, activation='sigmoid')(kmodel)

model = models.Model(input_tensor, output_tensor)
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30, batch_size=20,
                    validation_data=(validation_features, validation_labels))

# plot performance
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
plt.title('Training and Validatoin Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

'''
We got an improved accuracy of about 90%. This is good but we are still
overfitting. That's because we are not using data augmentation, which is essential
for preventing overfitting with small image datasets.
'''

### Second option - feature extraction with data augmentation ###
# we will extend the conv_base model and will run it end to end on the inputs
# Because models behave just like layers we can add it just like we would have
# done with a layer
kmodel = layers.Flatten()(conv_base.output)
kmodel = layers.Dense(256, activation='relu')(kmodel)
kmodel_out = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(conv_base.input, kmodel_out)
model.summary()

'''
Before compiling and fitting the model its important to FREEZE the pre-trained
model (the convolutional base). Freezing a layer or a set of layers means preventing 
their weights from being updated during training.
'''
len(model.trainable_weights)
for layer in conv_base.layers:
    layer.trainable = False
kmodel = layers.Flatten()(conv_base.output)
kmodel = layers.Dense(256, activation='relu')(kmodel)
kmodel_out = layers.Dense(1, activation='sigmoid')(kmodel)
model = models.Model(conv_base.input, kmodel_out)
len(model.trainable_weights)
model.summary()
'''
4 weight tensors will be trained for the two layers we added
(main weight matrix + bias vector) for each layer = 4
Note that if you change the weight trainability you need to compile the model
first for these changes to take effect.
Now we can start training our model with the same data augmentation configuration
we have used before. 
'''

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)
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
plt.title('Training and Validatoin Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
max(history.history['val_acc'])

## Fine-Tuning (another technique for model reuse, complementary to feature extraction) ##
'''
For fine tuning we will fine tune only the last 3 convolutional layers from the pre-trained
model. Why? Because earlier layers in the convolutional base encode more generic, reusable
features whereas upper layers encode more specialized features. It is more useful to make use
of the more generic features and fine tune the specific, more detailed features. This happens because
of the max_pooling effect (like strides) - each layer has a smaller window in the early layers thus
allowing it to learn general features like edges curves, etc. And upper layers has larger windows
(think of bigger strides) so that it learns more specific features like ear of a cat or nose of a dog.

The steps for fine-tuning are:
1. Add your custom network on top of an already trained base network
2. Freeze the base network
3. Train the part you added
4. Unfreeze some layers in the base pre-trained network
5. Jointly train both these layers and the part you added.

Up to this point we already did steps 1-3. We proceed to step 4.
'''
for layer in conv_base.layers:
    layer.trainable = True
len(conv_base.trainable_weights)

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

len(conv_base.trainable_weights)    # 3*2 trainable weights (3 main matrices and 3 bias vectors)
len(model.trainable_weights)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_finetuned.h5')

# plot the results
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

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

# smoothing the curves
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed Training accuracy')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

max(history.history['val_acc'])

# evaluate on test data
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test accuracy:', test_acc)


#### Visualization of Convnets ####
'''
Here we will visualize intermediate activations (output of a layer is
called its activation) of different layers
'''
model: models.Model = models.load_model('cats_and_dogs_small_2.h5')
model.summary()

# get an image of a cat which was not part of training
img_path = '/home/corel/python_projects/dl_keras_book/cats_dogs_small/test/cats/cat.1700.jpg'

# preprocess the imgae into a 4D array
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()