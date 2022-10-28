
from keras.layers import MaxPool2D, RandomFlip, RandomRotation, Rescaling, RandomZoom, Softmax
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional.conv2d import Conv2D
from keras.utils import img_to_array, load_img
from numpy.core.fromnumeric import resize
from keras.layers.activation import ReLU, elu
from keras.activations import swish
import tensorflow_datasets as tf_ds
from keras.models import Sequential
from random import seed, shuffle
import matplotlib_inline as plt2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k
import pathlib
import cv2
import os


batch_size = 32
img_height = 180
img_width = 180

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/mariofebo/Desktop/tesi/cod copia/dataset',
  labels = 'inferred',
  class_names = ['daisy', 'dandelion', 'hydrangea', 'rose', 'sunflower', 'tulip'],
  batch_size = batch_size,
  image_size = (img_height, img_width),
  shuffle = True,
  seed = 123,
  validation_split = 0.2,
  subset = "training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
  '/Users/mariofebo/Desktop/tesi/cod copia/dataset',
  labels = 'inferred',
  class_names = ['daisy', 'dandelion', 'hydrangea', 'rose', 'sunflower', 'tulip'],
  batch_size = batch_size,
  image_size = (img_height, img_width),
  shuffle = True,
  seed = 123,
  validation_split = 0.2,
  subset = "validation",
  )

class_names = ds_train.class_names
print(class_names)

for image_batch, labels_batch in ds_train:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = ds_train.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
num_classes

def data_aug():
  data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
  ])
  return data_augmentation


model = Sequential([
                    Rescaling(1./255, 
                    input_shape = (180, 180, 3)),
                    data_aug(),
                    Conv2D(8, 3, padding= 'same'), 
                    MaxPool2D(),
                    Conv2D(32, 3, padding = 'same'),
                    MaxPool2D(),
                    Conv2D(64, 3, padding= 'same'),
                    MaxPool2D(),
                    Flatten(),
                    Dropout(0.5),
                    Dense(18, activation= 'elu'), 
                    Dense(num_classes, activation='softmax')
                    ])


model.summary()


model.compile(
    loss ='sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    optimizer='adam',
)

report = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs = 25
)

accuracy = report.history['accuracy']
val_acc = report.history['val_accuracy']

loss = report.history['loss']
val_loss = report.history['val_loss']

epochs_range = range(25)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.evaluate(val_ds)
class_names


def load_image(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_image(raw)
  return tensor
load_image('girasole.jpg').shape

batch_img = tf.expand_dims('girasole.jpg', 0).shape
new_img = load_image('girasole.jpg')
new_img.shape

plt.imshow(new_img)

new_img = tf.expand_dims(new_img, 0)
new_img = resize(new_img, [1, 180, 180, 3])
new_img.shape
model.predict(new_img)

print(class_names[np.argmax(model.predict(new_img))])

def load_image(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_image(raw)
  return tensor
load_image('rosa.jpeg').shape

batch_img2 = tf.expand_dims('rosa.jpeg', 0).shape

new_img2 = load_image('rosa.jpeg')
new_img2.shape

plt.imshow(new_img2)

new_img2 = tf.expand_dims(new_img2, 0)
new_img2 = resize(new_img2, [1, 180, 180, 3])
new_img2.shape

model.predict(new_img2)

print(class_names[np.argmax(model.predict(new_img2))])

def load_image(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_image(raw)
  return tensor
load_image('tulipano.jpeg').shape

batch_img3 = tf.expand_dims('tulipano.jpeg', 0).shape

new_img3 = load_image('tulipano.jpeg')
new_img3.shape

plt.imshow(new_img3)

new_img3 = tf.expand_dims(new_img3, 0)
new_img3 = resize(new_img3, [1, 180, 180, 3])
new_img3.shape

model.predict(new_img3)
print(class_names[np.argmax(model.predict(new_img3))])

def load_image(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_image(raw)
  return tensor
load_image('margherita.jpeg').shape

batch_img4 = tf.expand_dims('margherita.jpeg', 0).shape

new_img4 = load_image('margherita.jpeg')
new_img4.shape

plt.imshow(new_img4)

new_img4 = tf.expand_dims(new_img4, 0)
new_img4 = resize(new_img4, [1, 180, 180, 3])
new_img4.shape

model.predict(new_img4)
print(class_names[np.argmax(model.predict(new_img4))])


def load_image(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_image(raw)
  return tensor
load_image('denteleo.jpeg').shape

batch_img6 = tf.expand_dims('denteleo.jpeg', 0).shape

new_img6 = load_image('denteleo.jpeg')
new_img6.shape

plt.imshow(new_img6)

new_img6 = tf.expand_dims(new_img6, 0)
new_img6 = resize(new_img6, [1, 180, 180, 3])
new_img6.shape

model.predict(new_img6)

print(class_names[np.argmax(model.predict(new_img6))])


def load_image(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_image(raw)
  return tensor
load_image('ortensia.jpeg').shape

batch_img = tf.expand_dims('ortensia.jpeg', 0).shape
new_img7 = load_image('ortensia.jpeg')
new_img7.shape

plt.imshow(new_img7)

new_img7 = tf.expand_dims(new_img7, 0)
new_img7 = resize(new_img7, [1, 180, 180, 3])
new_img7.shape
model.predict(new_img7)

print(class_names[np.argmax(model.predict(new_img7))])


cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret==True:
          cv2.imshow('webcam',frame)
          if cv2.waitKey(1) & 0xFF == ord('y'): 
            print("scatta la foto premendo il tasto 'y' ")
            img_cv = cv2.imwrite('/Users/mariofebo/Desktop/tesi/cod/img_cv.png', frame)
            cv2.destroyAllWindows()
            break
else:
  print("Error opening video stream or file")
cap.release()
print("Image written to file-system : ", img_cv)


def load_image_cv(file_name):
  raw = tf.io.read_file(file_name)
  tensor = tf.io.decode_png(raw)
  return tensor
load_image_cv('img_cv.png').shape


batch_img6 = tf.expand_dims('img_cv.png', 0).shape

img_cv = load_image_cv('img_cv.png')
img_cv.shape

plt.imshow(img_cv)

img_cv = tf.expand_dims(img_cv, 0)
img_cv = resize(img_cv, [1, 180, 180, 3])
img_cv.shape

model.predict(img_cv)

print(class_names[np.argmax(model.predict(img_cv))])
