#Simple Python Project using CNN and Keras
#Coded by hs_makkar
#Uses the Already Available Dataset in Keras


#Import Libraries
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
#Load the Dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#x_train has 60k samples of size 28x28


#Preprocess
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
num_classes=np.unique(y_train)
tot_classes=len(num_classes)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255
y_train=to_categorical(y_train,tot_classes)
y_test=to_categorical(y_test,tot_classes)



batch_size = 128
num_classes = 10
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

model.save('hand_wr.h5')
print("Saved Succesfuly")
#Will Get Good Accuracy as Overfitting has been prevented







