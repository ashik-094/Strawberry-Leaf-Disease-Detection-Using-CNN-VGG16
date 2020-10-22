import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization
import pickle
from tensorflow.keras.callbacks import TensorBoard
import datetime
import time
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedKFold
np.random.seed(7)
#data directortory (set yourself)
DATADIR = "DATASET" 
#name of classes
CATEGORIES = ["Strawberry___healthy",
              "Strawberry___Leaf_scorch",
          ]
      
#print number of classes
noOfClasses = len(CATEGORIES)
print('No of classes:%d' %noOfClasses)

#load image
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

#resizing image . in this case 50 by 50
IMG_SIZE = 224
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
#show the resized image
plt.imshow(new_array)
plt.show()

#create training data
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
                
n_fold = 5
    #layers of cnn ->convo,activation,maxpool
model = Sequential()
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()
        
        
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
            
model.summary()
    
    #output layer
model.add(Dense(noOfClasses))
model.add(Activation("softmax"))
model.summary()
        
for i in range(n_fold):   
    #create training data
    training_data =[]
    
    create_training_data()
    print(len(training_data))
    
    #shuffle all images to avoid overfit
    import random
    random.shuffle(training_data)
    """
    for sample in training_data[:10]:
        print(sample[1])
    """
    #seperate class and features
    X = []
    y = []
    for features, label in training_data:
            X.append(features)
            y.append(label)
    #convert image into numpy array to rehsape      
    X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,3) #for coloured  3 , for gray it will be 1 . here (3) 
    #one hot encode to class
    y=  to_categorical(y,noOfClasses)
    #### IMAGE AUGMENTATION
    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.1,
                                 rotation_range=10)
    dataGen.fit(X)
    
    
    #before feeding data we need to normalize(keras.utils.normalize-- most probably) but 
    #as image data are min 0 and max 255(pixel data)
    X = X/255.0
    n_fold = 5
    history=[]
    print('Fold[%d]' %(i+1))

    #compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer ="adam",
                  metrics =['accuracy'])
        
    #fit model and save in history
    history=model.fit(X,y, batch_size = 32, epochs =10 , validation_split = 0.2,shuffle =True)
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
        # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



# =============================================================================
# print(len(training_data))
# print(len(classNo))
# X_train,X_test,y_train,y_test = train_test_split(training_data,classNo,test_size=0.2)
# X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2)
# print(len(X_train))
# print(len(y_train))
# print(len(X_validation))
# print(len(y_validation))
# print(len(X_test))
# print(len(y_test))
# 
# =============================================================================

















