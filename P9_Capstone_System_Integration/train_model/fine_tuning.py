import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
#from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

# training data
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)


# print(train_batches.n)


mobile = keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = 'images/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# # img = mpimg.imread('images/red_1.jpg')
# # plt.imshow(img)
# # plt.show()

# preprocessed_image = prepare_image('red_1.jpg')
# predictions = mobile.predict(preprocessed_image)

# results = imagenet_utils.decode_predictions(predictions)
# print(results)

#print('works!')

print(mobile.summary())

x = mobile.layers[-6].output
output = Dense(units=3, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output)

print(model.summary())

for layer in model.layers[:-23]:
    layer.trainable = False

# print amount of batches
print('train_batches: ', train_batches.n)
print('valid_batches: ', valid_batches.n)
print('test_batches: ', test_batches.n)

# Train the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,
            steps_per_epoch=train_batches.n//10,
            epochs=4,
            validation_data=valid_batches,
            validation_steps=valid_batches.n//10
)

model.save('model_1.h5')
