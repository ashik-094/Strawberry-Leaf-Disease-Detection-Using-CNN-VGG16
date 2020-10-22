import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import datetime
import time
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
#set your dataset directory in DATADIR
DATADIR = "DATASET"
CATEGORIES = ["Strawberry___healthy",
              "Strawberry___Leaf_scorch",
          ]
noOfClasses = len(CATEGORIES)
print('No of classes:%d' %noOfClasses)
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  #path to classes
    for img in os.listdir(path):
        #for gray IMREAD_GRAYSCALE
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        plt.imshow(img_array)
        plt.show()
        break
    break
print(img_array.shape)
IMG_SIZE = 224
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array)
plt.show()
training_data =[]
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)  #path to classes
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #for gray IMREAD_GRAYSCALE
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
create_training_data()
print(len(training_data))
import random
random.shuffle(training_data)
"""
for sample in training_data[:10]:
    print(sample[1])
"""
X = []
y = []
for features, label in training_data:
        X.append(features)
        y.append(label)   
X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 3) #for coloured  3 , for gray it will be 1 . here (3) 
y=  to_categorical(y,noOfClasses)
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X)
NAME ="Cats-Dogs-cnn-64-{}".format(int(time.time()))
tensorboard = TensorBoard (log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
))
checkpointer = ModelCheckpoint(filepath="best_weights.weights", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

X = X/255.0
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())  #converts 3D to 1D
model.add(Dense(noOfClasses))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
             optimizer ="adam",
             metrics =['accuracy'])

#fit model and save in history
history=model.fit(X,y, batch_size = 32, epochs =10 , validation_split = 0.1,shuffle =True, callbacks =[tensorboard,checkpointer])
















