import os
print(os)

base_dir = '/home/neerajm/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
#  data should be formatted into appropriately preprocessed floatingpoint tensors before being fed into the network
# 1 Read the picture files.
# 2 Decode the JPEG content to RGB grids of pixels.
# 3 Convert these into floating-point tensors.
# 4 Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know,
# neural networks prefer to deal with small input values).
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255) # rescale all images by 1/255
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150), # Resizes all images to 150 × 150
batch_size=20,
class_mode='binary') # as we have binary labels

validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(150, 150),
batch_size=20,
class_mode='binary')

for data_batch, labels_batch in train_generator:
     print('data batch shape:', data_batch.shape)
     print('labels batch shape:', labels_batch.shape)
     break

from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# As it is a binary-classification problem, we’ll end the network with a
# single unit (a Dense layer of size 1) and a sigmoid activation

print(model.summary())

# Configuring the model for training

from tensorflow.keras import optimizers

# Because we ended the network with a single sigmoid unit,
# we’ll use binary crossentropy as the loss
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])
# Let’s fit the model to the data using the generator. You do so using the fit_generator
# method, the equivalent of fit for data generators like this one. It expects as its first
# argument a Python generator that will yield batches of inputs and targets indefinitely,
# like this one does. Because the data is being generated endlessly, the Keras model
# needs to know how many samples to draw from the generator before declaring an
# epoch over. This is the role of the steps_per_epoch argument: after having drawn
# steps_per_epoch batches from the generator—that is, after having run for
# steps_per_epoch gradient descent steps—the fitting process will go to the next
# epoch. In this case, batches are 20 samples, so it will take 100 batches until you see
# your target of 2,000 samples

history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=30,
validation_data=validation_generator,
validation_steps=50)

model.save('cats_and_dogs_small_1.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# These plots are characteristic of overfitting. The training accuracy increases linearly
# over time, until it reaches nearly 100%, whereas the validation accuracy stalls at 70–72%.
# The validation loss reaches its minimum after only five epochs and then stalls, whereas
# the training loss keeps decreasing linearly until it reaches nearly 0.
