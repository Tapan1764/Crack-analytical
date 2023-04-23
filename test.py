import glob as glob
import numpy as np
import tensorflow.keras as keras
import tensorflow_addons as tfa
import pandas as pd
import cv2

img = cv2.imread("D:\\ICT\\Sem 6\\HCD\\Project\\archive (2)\\Negative\\00041.jpg")
img1 = cv2.imread("D:\\ICT\\Sem 6\\HCD\\Project\\archive (2)\\Positive\\00021.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
x = np.expand_dims(img, axis=0)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (224, 224))
x1 = np.expand_dims(img1, axis=0)

# Define a custom metric function
def f1_score(y_true, y_pred):
    return tfa.metrics.F1Score(num_classes=2)(y_true, y_pred)['f1_score']

# Register the custom metric function
custom_objects = {'f1_score': f1_score}

# Load the Keras model, specifying the custom_objects parameter
with keras.utils.custom_object_scope(custom_objects):
    Model = keras.models.load_model('resnet_model1.h5')

# Make a prediction
predictions = Model.predict(x)

# Decode the predictions
# Load the class labels
class_labels = [ "Negative","Positive"]

# Get the index of the class with the highest probability
top_class_index = np.argmax(predictions)

# Get the class label
top_class_label = class_labels[top_class_index]
print("img : ",top_class_label)

# Make a prediction
predictions = Model.predict(x1)

# Decode the predictions
# Load the class labels
class_labels = [ "Negative","Positive"]

# Get the index of the class with the highest probability
top_class_index = np.argmax(predictions)

# Get the class label
top_class_label = class_labels[top_class_index]
print("img1 : ",top_class_label)

cv2.imshow("img",img)
cv2.imshow("img1",img1)

cv2.waitKey(0)