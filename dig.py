import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

# import matplotlib.pyplot
# import matplotlib
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

path1 = "/Users/poorvadixit/Documents/designlab/by_class"
path2 = "/Users/poorvadixit/Documents/designlab/data"

directories = os.listdir(path1)

# print directories

label = []
rows = 28
cols = 28

for directory in directories:
	if os.path.isdir(path1 + '/' + directory):
		print directory
		print "hey"
		curr_label = int(directory,16)
		print curr_label
		sub_directories = os.listdir(path1 + '/' + directory)
		for sub_directory in sub_directories:
			if os.path.isdir(path1 + '/' + directory + '/' + sub_directory):
				listing = os.listdir(path1 + '/' + directory + '/' + sub_directory)
				for file in listing:
					if file != '.DS_Store':
						im = Image.open(path1 + '/' + directory + '/' + sub_directory + '/' + file)
						img = im.resize((rows,cols),Image.ANTIALIAS)
						gray = img.convert('L')
						gray.save(path2 + '/' + directory+sub_directory+file, "JPEG")
						label.append(curr_label)

directory = os.listdir(path2)

im1 = array(Image.open(path2 + '/' + directory[0]))
m,n = im1.shape[0:2]
num_img = len(directory)
print num_img
# print label

# directory.remove('.DS_Store')

immatrix = array([array(Image.open(path2 + '/' + im2)).flatten()
	                for im2 in directory] , 'f')

data,label = shuffle(immatrix,label,random_state=2)
train_data = [data,label]

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=4)

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print X_train.shape,num_classes

def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = larger_model()
# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

