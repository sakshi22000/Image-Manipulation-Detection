from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input,InputLayer, Conv2D,UpSampling2D , Flatten,MaxPooling2D,Conv2DTranspose
from tensorflow.keras.models import Model,Sequential
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from PIL import Image
import random
from math import ceil
import imageio
import cv2
from skimage.data import astronaut
from skimage import transform,io
import numpy as np
import cv2
from pprint import pprint as pp
from operator import itemgetter
import sys

os.chdir('E:\\Project image manipulation\\manipulation_data\\real_data\\original_data\\')
rootdir = os.getcwd()
filelist=[]
transformed_image=[]
#defining image arrays
tampered_data=[]
real_data=[]
#filling in original data
i=0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if (filepath.find('gt') >0):
            i=i+1
            print(filepath)
            width = 300
            height = 300
            image1 = cv2.imread(filepath,0)
            transformed_image = transform.resize(image1,(300,300), mode='symmetric', preserve_range=True)
            transformed_image=(transformed_image-transformed_image.mean())/255
            real_data.append(transformed_image)
os.chdir('E:\\Project image manipulation\\manipulation_data\\real_data\\manipulated_data\\') 
rootdir = os.getcwd()
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if (filepath.find('_gt') > 0):
            i=i+1
            print(filepath)
            im = Image.open(filepath)
            width = 300
            height = 300
            image1 = cv2.imread(filepath,0)
            transformed_image = transform.resize(image1,(300,300), mode='symmetric', preserve_range=True)
            transformed_image = (transformed_image-transformed_image.mean())/255
            tampered_data.append(transformed_image)
transformed_image=(transformed_image-transformed_image.mean())/255
transformed_image.max()
transformed_image.min()
real_data=np.asarray(real_data)
tampered_data=np.asarray(tampered_data)
tf.config.experimental_run_functions_eagerly(True)
real_data=tf.expand_dims(real_data, axis=-1)
tampered_data=tf.expand_dims(tampered_data, axis=-1)
combined_input=np.concatenate([real_data, tampered_data])
y_combined=np.zeros(real_data.shape[0]+tampered_data.shape[0])
y_combined[:real_data.shape[0]]=1
combined_input=combined_input.reshape(575,300,300,1)
trainsize= ceil(0.8 * combined_input.shape[0])
testsize= ceil(0.2 * combined_input.shape[0])+1
trainsel=np.random.randint(low=0,high=combined_input.shape[0],size=trainsize)
testsel=np.random.randint(low=0,high=combined_input.shape[0],size=testsize)
train_inp=combined_input[trainsel,]
test_inp=combined_input[testsel,]
train_out=y_combined[trainsel,]
test_out=y_combined[testsel,]
def adam_optimizer():
    return Adam(lr=0.001,beta_1=0.9)
model = Sequential()
model.add(InputLayer(input_shape=(300, 300, 1)))
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same',strides=2))
model.add(Conv2D(32, (3, 3), activation='tanh',padding='same'))
model.add(layers.LeakyReLU(0.6))
model.add(layers.Dropout(0.4))
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same',strides=2))
model.add(layers.LeakyReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same',strides=2))
model.add(layers.LeakyReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='tanh', padding='same',strides=2))
model.add(layers.LeakyReLU(0.3))
model.add(layers.Dropout(0.2))
model.add(Conv2D(1, (3, 3), activation='tanh',padding='same',strides=2))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
print(model.summary())
model.save("E:\model.h5")
model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
model.fit(x=train_inp, y=train_out, batch_size=100,epochs=3)
train_pred = model.predict(train_inp)
test_pred = model.predict(test_inp)
check=np.interp(train_pred, (train_pred.min(), train_pred.max()), (0,1))
check1=np.interp(test_pred, (test_pred.min(), test_pred.max()),(0,1))
model.save("E:\model.h5")
check1 = np.interp(test_pred, (test_pred.min(), test_pred.max()),(0,1))
plt.hist(check)
plt.hist(check1)
plt.xlabel('Probability of the model being accurate')
plt.ylabel('No.of Images')
train_check = np.concatenate((train_out.reshape(-1,1),check.reshape(-1,1)),axis=1)
test_check = np.concatenate((test_out.reshape(-1,1),check1.reshape(-1,1)),axis=1)
model.evaluate(train_inp,train_out)
y_pred=model.predict(test_inp)
print(test_out)