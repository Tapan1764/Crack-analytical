import glob as glob
import numpy as np

import pandas as pd
import keras
import cv2

img = cv2.imread("D:\\ICT\\Sem 6\\HCD\\Project\\archive (2)\\Positive\\00716.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
x = np.expand_dims(img, axis=0)

model = keras.models.load_model('resnet_model1.h5')

predictions = model.predict(x)
class_labels = [ "Negative","Positive"]
top_class_index = np.argmax(predictions)

top_class_label = class_labels[top_class_index]
print(top_class_label)