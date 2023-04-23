import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
#PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
#IMAGE PROCESS
from PIL import Image
import tensorflow
from tensorflow.keras.utils import load_img,img_to_array
#from tensorflow.keras.utils import utils as image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
# from keras.preprocessing import image
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import convolve
from skimage import data, io, filters
import skimage
from skimage.morphology import convex_hull_image, erosion
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#OPTIMIZER
from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
#MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D
from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import VGG16,VGG19,inception_v3
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
#SKLEARN CLASSIFIER
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)

"""# PATH,LABEL,TRANSFORMATION PROCESS

#### MAIN PATH
"""

Surface_Data = Path("archive (2)")

"""#### JPG PATH"""

Surface_JPG_Path = list(Surface_Data.glob(r"*/*.jpg"))

"""#### JPG LABELS"""

Surface_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],Surface_JPG_Path))

"""#### TO SERIES"""

Surface_JPG_Path_Series = pd.Series(Surface_JPG_Path,name="JPG").astype(str)
Surface_Labels_Series = pd.Series(Surface_Labels,name="CATEGORY")

"""#### TO DATAFRAME"""

Main_Surface_Data = pd.concat([Surface_JPG_Path_Series,Surface_Labels_Series],axis=1)

print(Main_Surface_Data.head(-1))

# from google.colab import drive
# drive.mount('/content/drive')

"""#### TO SHUFFLE"""

Main_Surface_Data = Main_Surface_Data.sample(frac=1).reset_index(drop=True)

print(Main_Surface_Data.head(-1))

"""# VISUALIZATION"""

plt.style.use("dark_background")

"""#### LABESL"""

Positive_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Positive"]
Negative_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Negative"]

Positive_Surface = Positive_Surface.reset_index()
Negative_Surface = Negative_Surface.reset_index()

"""#### VISION FUNCTION"""

def simple_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    plt.xlabel(Reading_Img.shape)
    plt.ylabel(Reading_Img.size)
    # plt.imshow(Reading_Img)

def canny_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    Canny_Img = cv2.Canny(Reading_Img,90,100)
    
    plt.xlabel(Canny_Img.shape)
    plt.ylabel(Canny_Img.size)
    # plt.imshow(Canny_Img)

def threshold_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    _,Threshold_Img = cv2.threshold(Reading_Img,130,255,cv2.THRESH_BINARY_INV)
    
    plt.xlabel(Threshold_Img.shape)
    plt.ylabel(Threshold_Img.size)
    # plt.imshow(Threshold_Img)

def threshold_canny(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    _,Threshold_Img = cv2.threshold(Reading_Img,130,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    
    plt.xlabel(Canny_Img.shape)
    plt.ylabel(Canny_Img.size)
    # plt.imshow(Canny_Img)

"""#### SIMPLE VISION"""

simple_vision(Main_Surface_Data["JPG"][4])

simple_vision(Main_Surface_Data["JPG"][2])

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    operations.set_xlabel(Reading_Img.shape)
    operations.set_ylabel(Reading_Img.size)
    operations.imshow(Reading_Img)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    operations.set_xlabel(Reading_Img.shape)
    operations.set_ylabel(Reading_Img.size)
    operations.imshow(Reading_Img)
    
# plt.tight_layout()
# plt.show()

"""#### CANNY VISION"""

canny_vision(Main_Surface_Data["JPG"][4])

canny_vision(Main_Surface_Data["JPG"][2])

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    Canny_Img = cv2.Canny(Reading_Img,90,100)
    
    operations.set_xlabel(Canny_Img.shape)
    operations.set_ylabel(Canny_Img.size)
    operations.imshow(Canny_Img)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    Canny_Img = cv2.Canny(Reading_Img,90,100)
    
    operations.set_xlabel(Canny_Img.shape)
    operations.set_ylabel(Canny_Img.size)
    operations.imshow(Canny_Img)
    
# plt.tight_layout()
# plt.show()

"""#### THRESHOLD VISION"""

threshold_vision(Main_Surface_Data["JPG"][4])

threshold_vision(Main_Surface_Data["JPG"][2])

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    
    operations.set_xlabel(Threshold_Img.shape)
    operations.set_ylabel(Threshold_Img.size)
    operations.imshow(Threshold_Img)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    
    operations.set_xlabel(Threshold_Img.shape)
    operations.set_ylabel(Threshold_Img.size)
    operations.imshow(Threshold_Img)
    
# plt.tight_layout()
# plt.show()

"""#### THRESHOLD-CANNY VISION"""

threshold_canny(Main_Surface_Data["JPG"][4])

threshold_canny(Main_Surface_Data["JPG"][2])

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    
    operations.set_xlabel(Canny_Img.shape)
    operations.set_ylabel(Canny_Img.size)
    operations.imshow(Canny_Img)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    
    operations.set_xlabel(Canny_Img.shape)
    operations.set_ylabel(Canny_Img.size)
    operations.imshow(Canny_Img)
    
# plt.tight_layout()
# plt.show()

"""#### DRAW CONTOURS"""

figure,axis = plt.subplots(nrows=1,ncols=3,figsize=(12,12))

Reading_Img = cv2.imread(Main_Surface_Data["JPG"][4])
Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)

_,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
Canny_Img = cv2.Canny(Threshold_Img,90,100)
contours,_ = cv2.findContours(Canny_Img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
Draw_Contours = cv2.drawContours(Reading_Img,contours,-1,(255,0,0),1)

axis[0].imshow(Threshold_Img)
axis[1].imshow(Canny_Img)
axis[2].imshow(Draw_Contours)

figure,axis = plt.subplots(nrows=1,ncols=3,figsize=(12,12))

Reading_Img = cv2.imread(Main_Surface_Data["JPG"][2])
Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)

_,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
Canny_Img = cv2.Canny(Threshold_Img,90,100)
contours,_ = cv2.findContours(Canny_Img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
Draw_Contours = cv2.drawContours(Reading_Img,contours,-1,(255,0,0),1)

axis[0].imshow(Threshold_Img)
axis[1].imshow(Canny_Img)
axis[2].imshow(Draw_Contours)

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    contours,_ = cv2.findContours(Canny_Img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    Draw_Contours_Positive = cv2.drawContours(Reading_Img,contours,-1,(255,0,0),1)
    
    operations.set_xlabel(Draw_Contours_Positive.shape)
    operations.set_ylabel(Draw_Contours_Positive.size)
    operations.imshow(Draw_Contours_Positive)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    contours,_ = cv2.findContours(Canny_Img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    Draw_Contours_Negative = cv2.drawContours(Reading_Img,contours,-1,(255,0,0),1)
    
    operations.set_xlabel(Draw_Contours_Negative.shape)
    operations.set_ylabel(Draw_Contours_Negative.size)
    operations.imshow(Draw_Contours_Negative)
    
# plt.tight_layout()
# plt.show()

"""#### HESSIAN MATRIX"""

figure,axis = plt.subplots(nrows=1,ncols=2,figsize=(12,12))

Reading_Img = cv2.imread(Negative_Surface["JPG"][2])
Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)

Hessian_Matrix_Img = hessian_matrix(Reading_Img,sigma=0.15,order="rc")
maxima_Img,minima_Img = hessian_matrix_eigvals(Hessian_Matrix_Img)

axis[0].imshow(maxima_Img,cmap="Greys_r")
axis[1].imshow(minima_Img,cmap="Greys_r")

figure,axis = plt.subplots(nrows=1,ncols=2,figsize=(12,12))

Reading_Img = cv2.imread(Positive_Surface["JPG"][2])
Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)

Hessian_Matrix_Img = hessian_matrix(Reading_Img,sigma=0.15,order="rc")
maxima_Img,minima_Img = hessian_matrix_eigvals(Hessian_Matrix_Img)

axis[0].imshow(maxima_Img,cmap="Greys_r")
axis[1].imshow(minima_Img,cmap="Greys_r")

"""#### THRESHOLD SKELETON MORPHOLOGY"""

figure,axis = plt.subplots(nrows=1,ncols=2,figsize=(12,12))

Reading_Img = cv2.imread(Positive_Surface["JPG"][2])
Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)
_,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)

Array_Img = np.array(Reading_Img > Threshold_Img).astype(int)
Skeleton_Morphology_Img = skimage.morphology.skeletonize(Array_Img)

axis[0].imshow(Reading_Img,cmap="Greys_r")
axis[1].imshow(Skeleton_Morphology_Img)

figure,axis = plt.subplots(nrows=1,ncols=2,figsize=(12,12))

Reading_Img = cv2.imread(Negative_Surface["JPG"][2])
Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)
_,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)

Array_Img = np.array(Reading_Img > Threshold_Img).astype(int)
Skeleton_Morphology_Img = skimage.morphology.skeletonize(Array_Img)

axis[0].imshow(Reading_Img,cmap="Greys_r")
axis[1].imshow(Skeleton_Morphology_Img)

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Array_Img = np.array(Reading_Img > Threshold_Img).astype(int)
    Skeleton_Morphology_Img = skimage.morphology.skeletonize(Array_Img)
    
    operations.set_xlabel(Skeleton_Morphology_Img.shape)
    operations.set_ylabel(Skeleton_Morphology_Img.size)
    operations.imshow(Skeleton_Morphology_Img)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Array_Img = np.array(Reading_Img > Threshold_Img).astype(int)
    Skeleton_Morphology_Img = skimage.morphology.skeletonize(Array_Img)
    
    operations.set_xlabel(Skeleton_Morphology_Img.shape)
    operations.set_ylabel(Skeleton_Morphology_Img.size)
    operations.imshow(Skeleton_Morphology_Img)
    
# plt.tight_layout()
# plt.show()

"""#### CANNY SKELETON MORPHOLOGY"""

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    Array_Img = np.array(Reading_Img > Canny_Img).astype(int)
    Skeleton_Morphology_Img = skimage.morphology.skeletonize(Array_Img)
    
    operations.set_xlabel(Skeleton_Morphology_Img.shape)
    operations.set_ylabel(Skeleton_Morphology_Img.size)
    operations.imshow(Skeleton_Morphology_Img)
    
# plt.tight_layout()
# plt.show()

figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2GRAY)
    
    _,Threshold_Img = cv2.threshold(Reading_Img,150,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    Array_Img = np.array(Reading_Img > Canny_Img).astype(int)
    Skeleton_Morphology_Img = skimage.morphology.skeletonize(Array_Img)
    
    operations.set_xlabel(Skeleton_Morphology_Img.shape)
    operations.set_ylabel(Skeleton_Morphology_Img.size)
    operations.imshow(Skeleton_Morphology_Img)
    
# plt.tight_layout()
# plt.show()

"""# SPLITTING TRAIN AND TEST"""

xTrain,xTest = train_test_split(Main_Surface_Data,train_size=0.8,shuffle=True,random_state=42)

print(xTrain.shape)
print(xTest.shape)

"""# IMAGE GENERATOR

#### STRUCTURE
"""

Train_IMG_Generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=25,
                                        shear_range=0.5,
                                        zoom_range=0.5,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        brightness_range=[0.6,0.9],
                                        vertical_flip=True,
                                        validation_split=0.1)

Test_IMG_Generator = ImageDataGenerator(rescale=1./255)

"""#### HOW TO LOOK BY GENERATOR"""

Example_Surface_Img = Main_Surface_Data["JPG"][444]
Loading_Img = load_img(Example_Surface_Img,target_size=(220,220),color_mode="rgb")
Array_Img = img_to_array(Loading_Img)
Array_Img = Array_Img.reshape((1,) + Array_Img.shape)

i = 0

for batch in Train_IMG_Generator.flow(Array_Img,batch_size=32):
    plt.figure(i)
    Image_Out = plt.imshow(tensorflow.keras.utils.img_to_array(batch[0]))
    i += 1
    
    if i % 6 == 0:
        break

# plt.show()

"""#### APPLYING"""

Train_Set = Train_IMG_Generator.flow_from_dataframe(dataframe=xTrain,
                                                   x_col="JPG",
                                                   y_col="CATEGORY",
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   target_size=(200,200),
                                                   subset="training",
                                                    batch_size=32,
                                                    seed=32)

Validation_Set = Train_IMG_Generator.flow_from_dataframe(dataframe=xTrain,
                                                   x_col="JPG",
                                                   y_col="CATEGORY",
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   target_size=(200,200),
                                                   subset="validation",
                                                    batch_size=32,
                                                    seed=32)

Test_Set = Test_IMG_Generator.flow_from_dataframe(dataframe=xTest,
                                                   x_col="JPG",
                                                   y_col="CATEGORY",
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   target_size=(200,200),
                                                    batch_size=32,
                                                    seed=32)

"""#### CHECKING"""

print("TRAIN: ")
print(Train_Set.class_indices)
print(Train_Set.classes[0:5])
print(Train_Set.image_shape)
print("---"*20)
print("VALIDATION: ")
print(Validation_Set.class_indices)
print(Validation_Set.classes[0:5])
print(Validation_Set.image_shape)
print("---"*20)
print("TEST: ")
print(Test_Set.class_indices)
print(Test_Set.classes[0:5])
print(Test_Set.image_shape)

"""# MODEL RNN-LSTM

#### PARAMETERS
"""

print(Train_Set.image_shape[0],Train_Set.image_shape[1],Train_Set.image_shape[2])

compile_optimizer = "adam"
compile_loss = "binary_crossentropy"
input_dim = (Train_Set.image_shape[0],Train_Set.image_shape[1],Train_Set.image_shape[2])
class_dim = 1

"""#### CALLBACKS"""

Early_Stopper = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")
Checkpoint_Model = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      filepath="model_save")

"""#### STRUCTURE"""

Model = Sequential()

Model.add(Conv2D(32,(3,3),activation="relu",input_shape=input_dim))
Model.add(BatchNormalization())
Model.add(MaxPooling2D((2,2)))

Model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
Model.add(Dropout(0.2))
Model.add(MaxPooling2D((2,2)))

Model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
Model.add(Dropout(0.2))
Model.add(MaxPooling2D((2,2)))

Model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
Model.add(Dropout(0.2))
Model.add(MaxPooling2D((2,2)))

Model.add(Flatten())
Model.add(Dense(512,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(class_dim,activation="sigmoid"))

Model.compile(optimizer=compile_optimizer,loss=compile_loss,metrics=["accuracy"])

CNN_Model = Model.fit(Train_Set,
                      validation_data=Validation_Set,
                      callbacks=[Early_Stopper,Checkpoint_Model],
                      epochs=5)

"""#### CHECKING MODEL"""

print(Model.summary())

Model.save("Model_Last_Prediction.h5")

Grap_Data = pd.DataFrame(CNN_Model.history)
Grap_Data.plot()

plt.plot(CNN_Model.history["accuracy"])
plt.plot(CNN_Model.history["val_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()

plt.plot(CNN_Model.history["loss"])
plt.plot(CNN_Model.history["val_loss"])
plt.ylabel("LOSS")
plt.legend()
plt.show()

"""#### PREDICTION PROCESS"""

Model_Results = Model.evaluate(Test_Set)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.2f" % Model_Results[1])

Model_Test_Prediction = Model.predict(Test_Set)
Model_Test_Prediction = Model_Test_Prediction.argmax(axis=-1)
print(Model_Test_Prediction)

Model_Test_Prediction_Classes = Model.predict_classes(Test_Set)
# Model_Test_Prediction_Classes = Model.predict_step(Test_Set)
print(Model_Test_Prediction_Classes)

fig, axes = plt.subplots(nrows=5,
                         ncols=5,
                         figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(xTest["JPG"].iloc[i]))
    ax.set_title(f"PREDICTION:{Model_Test_Prediction_Classes[i]}")
    ax.set_xlabel(xTest["CATEGORY"].iloc[i])
plt.tight_layout()
plt.show()