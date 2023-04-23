import numpy as np

# Commented out IPython magic to ensure Python compatibility.
# import pertinent libraries
import os
import glob as glob
import numpy as np
# [Keras Models]
# import the Keras implementations of VGG16, VGG19, InceptionV3 and Xception models
# the model used here is VGG16
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
# %matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import load_img,img_to_array
import tensorflow_addons as tfa

import cv2

"""# Data

## Data Augmentation
"""

import Augmentor
p = Augmentor.Pipeline("archive\\train")
p.rotate90(probability=0.7)
#p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
p.flip_random(probability = 0.2)
#p.skew(0.5,magnitude=1)
p.random_brightness(0.3,0.75,1.25)
p.process()
p.sample(3000)

"""## Bilateral Filtering"""

def Bilateral_filtering(directory):
    for filename in os.listdir(directory):       
        # load the image
        img = cv2.imread(os.path.join(directory, filename))

        # apply bilateral filtering
        filtered_img = cv2.bilateralFilter(img, 6, 75,1475)

        # save the filtered image
        cv2.imwrite(os.path.join(directory,filename), filtered_img)

"""### Train"""

directory = "archive\\train\\output\\Positive"
Bilateral_filtering(directory)

directory = "archive\\train\\output\\Negative"
Bilateral_filtering(directory)

"""### Val and Test"""

directory = "archive\\valid\\Positive"
Bilateral_filtering(directory)
directory = "archive\\valid\\Negative"
Bilateral_filtering(directory)

directory = "archive\\test\\Positive"
Bilateral_filtering(directory)
directory = "archive\\test\\Negative"
Bilateral_filtering(directory)

"""## Grayscaling"""

def Grayscaling(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(filepath, gray_image)

"""### Train"""

directory = "archive\\train\\output\\Positive"
Grayscaling(directory)
directory = "archive\\train\\output\\Negative"
Grayscaling(directory)

"""### Val and Test"""

directory = "archive\\valid\\Positive"
Grayscaling(directory)
directory = "archive\\valid\\Negative"
Grayscaling(directory)

directory = "archive\\test\\Positive"
Grayscaling(directory)
directory = "archive\\test\\Negative"
Grayscaling(directory)





"""# Data Generators"""

negative_images = os.listdir("archive\\train\\output\\Negative")
positive_images = os.listdir("archive\\train\\output\\Positive")

num_classes = 2
image_resize = 224
batch_size_training = 50
batch_size_validation = 50

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = data_generator.flow_from_directory(
    "archive\\train\\output",
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    'archive\\valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')

"""# Model"""

model = Sequential()
model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
model.add(Dense(num_classes, activation='softmax'))

model.add(Dense(2, activation = "sigmoid"))

model.layers[0].trainable = False

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tfa.metrics.F1Score(num_classes=2, average="micro")])

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 20

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)

model.evaluate(train_generator,steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

model.save('resnet_model1.h5')

"""# Model 2"""

from keras.applications.vgg16 import VGG16

model_vgg = Sequential()
model_vgg.add(VGG16(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
model_vgg.add(Dense(num_classes, activation='softmax'))
model_vgg.add(Dense(num_classes, activation ='sigmoid'))

model_vgg.layers[0].trainable = False

model_vgg.summary()

model_vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tfa.metrics.F1Score(num_classes=2, average="micro")])

fit_history_vgg = model_vgg.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    
)

model_vgg.save('vgg16_model1.h5')

data_generator_test = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)
test_generator = data_generator_test.flow_from_directory(
    'archive\\test',
    target_size=(image_resize, image_resize),
    shuffle=False)

model_vgg.evaluate(test_generator,steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

"""## Final Metrics"""

from keras import backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1_val = 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))
    return f1_val

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',recall,precision,f1_score])

"""### An array of [Loss, Accuracy, Recall, Precision, F1 Score]"""

model.evaluate_generator(test_generator,steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

"""### Prediction for cases"""

directory = "archive\\predict\\Positive"
Bilateral_filtering(directory)
Grayscaling(directory)

from PIL import Image
Image(filename='archive\\predict\\IMG_1129.JPG')

img = load_img("archive\\predict\\IMG_1129.JPG", target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

model = keras.models.load_model('resnet_model1.h5')

# Make a prediction
predictions = model.predict(x)

# Decode the predictions
# Load the class labels
class_labels = [ "Negative","Positive"]

# Get the index of the class with the highest probability
top_class_index = np.argmax(predictions)

# Get the class label
top_class_label = class_labels[top_class_index]
print(top_class_label)











