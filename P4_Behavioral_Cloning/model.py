import os
import csv

import cv2
import numpy as np
import sklearn
import math

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


# reads CSV-file and returns list of CSV-lines
def getDataFromCSV(pathCSV):
    samples = []

    folderPathList = pathCSV.split('/')
    folderPathList.pop()
    folderPath = ''
    for s in folderPathList:
        folderPath += s
        folderPath += '/'

    with open(pathCSV) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0] = folderPath + 'IMG/' + line[0].split('/')[-1]
            line[1] = folderPath + 'IMG/' + line[1].split('/')[-1]
            line[2] = folderPath + 'IMG/' + line[2].split('/')[-1]
            samples.append(line)

    # remove first row with captions
    samples.pop(0)

    return samples



# searches for CSV-files in the given directory, returns lines from all CSV-files
def loadAllTrainingData(strPath):
    samples = []
    
    subdirs = [x[0] for x in os.walk(strPath)]
    subdirs.pop(0)
    for sd in subdirs:
        if '/IMG' in sd:
            subdirs.remove(sd)

    for sd in subdirs:
        samples.extend(getDataFromCSV(sd+'/driving_log.csv'))
        
    return samples



# load data from all CSV-files
samples = []
# the training data was saved in google drive and can be downloaded to /opt directory using ./dowload_data.sh script
samples = loadAllTrainingData('/opt/trainingData')

print('Number of samples: ', len(samples))

# split training and testing data: 80% - training data, 20% - testing data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Image augmentation by image flipping
def augmentImg(img, angle):
    # augment image by flipping
    center_image = np.fliplr(img)
    center_angle = -angle

    return center_image, center_angle


# generator returns a defined number of samples each iteration. This approach saves memory
def generator(samples, batch_size=32, do_augmentation=False):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # load image and convert to RGB
                center_image = cv2.imread(batch_sample[0])
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                if do_augmentation:
                    # augment image by flipping to create additional data
                    flippedCenterImg, flippedCenterAngle = augmentImg(center_image, center_angle)
                    images.append(flippedCenterImg)
                    angles.append(flippedCenterAngle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size, do_augmentation=True)
validation_generator = generator(validation_samples, batch_size=batch_size, do_augmentation=False)


# model from NVIDIA paper
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
# image normalization
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3)))
# cropping of image to remove not important image parts (sky, trees...)
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
# convolution layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3,  activation='relu'))
model.add(Convolution2D(64, 3, 3,  activation='relu'))

model.add(Flatten())

# Dense layers
model.add(Dense(100))
# Dropout used to prevent overfitting. Dropout probability of 0.5 shows good results
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

# adam optimizer allows not to specify the learning rate
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=math.ceil(len(validation_samples) / batch_size), epochs=8, verbose=1)

# save the trained model
model.save('model.h5')

