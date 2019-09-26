# Overfitting is caused by having too few samples to learn from, rendering you unable
# to train a model that can generalize to new data

# Data augmentation takes the approach of generating more training data
# from existing training samples, by augmenting the samples via a number of random
# transformations that yield believable-looking images.


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

# - rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
# - width_shift and height_shift are ranges (as a fraction of total width or
# height) within which to randomly translate pictures vertically or horizontally.
# - shear_range is for randomly applying shearing transformations.
# - zoom_range is for randomly zooming inside pictures.
# - horizontal_flip is for randomly flipping half the images horizontally—relevant
# when there are no assumptions of horizontal asymmetry (for example, real-world pictures).
# - fill_mode is the strategy used for filling in newly created pixels, which can
# appear after a rotation or a width/height shift.

import os
from tensorflow.keras.preprocessing import image

base_dir = '/home/neerajm/cats_and_dogs_filtered'
train_cats_dir = os.path.join(base_dir, 'train/cats')

fnames = [os.path.join(train_cats_dir, fname) for
fname in os.listdir(train_cats_dir)]

img_path = fnames[4]
img = image.load_img(img_path, target_size=(150, 150)) # reads and resize
x = image.img_to_array(img) # converts to numpy array (150,150,3)

x = x.reshape((1,) + x.shape)

import matplotlib.pyplot as plt
i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
