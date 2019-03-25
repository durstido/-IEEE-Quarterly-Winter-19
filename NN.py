from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionResNetV2
import random
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Model

#collect data
blue = ["/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/bluePill/{}".format(i) for i in os.listdir("/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/bluePill")]
green = ["/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/greenPill/{}".format(i) for i in os.listdir("/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/greenPill")]
red = ["/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/redPill/{}".format(i) for i in os.listdir("/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/redPill")]
white = ["/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/whitePill/{}".format(i) for i in os.listdir("/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/whitePill")]
train_imgs = blue + green + red + white
#shuffle the data to make it random
random.shuffle(train_imgs)

#size of rescaled images
nrows = 150
ncols = 150
channels = 3

#method to read png images and change their sizes, as well as record the labels
def read_and_process_image(list_of_images):

    X = [] #images
    y = [] #labels

    for image in list_of_images:
        img = cv2.imread(image, 0)
        try:
            img.shape #proceed only if the image actually exists
            X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncols)).astype('float32'))
            #get the labels
            if 'blue' in image:
                y.append(0)
            elif 'green' in image:
                y.append(1)
            elif 'red' in image:
                y.append(2)
            elif 'white' in image:
                y.append(3)
        except AttributeError:
            print(image + " shape not found")

    return X, y

#read, change size, and convert data images to arrays
X, y = read_and_process_image(train_imgs)
X = np.stack(X)
y = np.array(y)

#split the data images to training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2, random_state=2)

#convert the label arrays to one-hot vector
y_train_ohe = np_utils.to_categorical(y_train)
y_val_ohe = np_utils.to_categorical(y_val)

#get length of train/val data
ntrain = len(X_train)
nval = len(X_val)

batch_size = 32

#trasnfer learning part:
#load InceptionResnetV2 to use as basis for our network
conv_base = InceptionResNetV2(weights = 'imagenet', include_top=False, input_shape=(150, 150, 3))

#adjust the ending of the bast model to fit our data
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten()) 
model.add(layers.Dense(256, activation='sigmoid'))
model.add(layers.Dense(4, activation='softmax')) #4 because we have 4 classes to classify into


#when called, will do image augmentation to the data images
train_datagen = ImageDataGenerator (rescale = 1/255,
                                    rotation_range=40,
                                    width_shift_range=.2,
                                    height_shift_range = .2,
                                    shear_range=.2,
                                    zoom_range=.2,
                                    horizontal_flip=True,)

#when called, will do image augmentation to the data images
val_datagen = ImageDataGenerator(rescale=1/255)

#flow takes the data and labels and creates batches of augmented data
train_generator = train_datagen.flow(X_train, y_train_ohe, batch_size = batch_size)
val_generator = val_datagen.flow(X_val, y_val_ohe, batch_size=batch_size)

#freeze the base network and only use ours from here on
print('Number weights before freezing:', len(model.trainable_weights))
conv_base.trainable = False
print('Number of weights after freezing:', len(model.trainable_weights))

#compile the network with the appropriate loss and metrices
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 

#fit the model and begin the training
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain/batch_size,
                              epochs = 50,
                              validation_data=val_generator,
                              validation_steps=nval/batch_size)

#save the model and the weights
model.save_weights('model_weights.h5')
model.save('model.h5')

#uncomment to view the graphs for accuracy and loss
'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
