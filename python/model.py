# importing the dependencies
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import keras
from keras import backend as k
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

# a function to resize the image into appropriate dimensions
def resize(img):
    img = cv2.resize(img,(20,20))
    return img
    

X_train = []
y_train = []

# to get the name of the folder
for name_folder in os.listdir("extracted_letter_images") :

    name = 'extracted_letter_images/' + name_folder
    for f in listdir(name):
        # name of the folder is the name of the output
        y_train.append(np.asarray(name_folder))
        
        # constructing full path to the image
        name = 'extracted_letter_images/' + name_folder + '/' + f
        
        # reading the image
        image = cv2.imread(name,0)/255
        
        # appending to form the image list
        image = np.asarray(image)
        image = resize(image)
        X_train.append([image])

# converting the lsit into an numoy array so that it can be fed into neural network
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_train = np.reshape(X_train, [-1,20,20,1])

X_train.shape

# one hot encoding the output labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)

# defining the architecture of the model
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(20,20,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# third convolutional layer with max pooling
model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(100, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(32, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# splitting the samples into training and testing sets so that we cross validate our results
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train ,test_size = .2)

# shape of the training output
y_train.shape

# fitting the model
model.fit(X_train, y_train,
          validation_data=[X_test,y_test],
          batch_size=2000, 
          epochs=10, 
          verbose=1)

#saving the model 
model.save('models/model.h5')

np.save('models/xtrain',X_train,allow_pickle=True)



