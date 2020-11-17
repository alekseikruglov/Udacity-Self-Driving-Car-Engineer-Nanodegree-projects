import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from keras.utils.generic_utils import CustomObjectScope
#from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# model = load_model('model_1.h5')

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('model_1.h5')

# testing data
# test_path = 'data/small_test'
test_path = 'data/test'

test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=150, shuffle=False)



# test_imgs, test_labels = next(test_batches)
# for i in range(len(test_imgs)):
#     plt.figure()
#     plt.imshow(test_imgs[i])
#     print('label: ', test_labels[i])


# plt.show()

samples = len(test_batches.filenames)
print('samples: ', samples)

test_labels = test_batches.classes

predictions = model.predict_generator(test_batches, steps=1, verbose=0)

print('len(predictions)', len(predictions))

for i in range(samples):
    print('prediction: ', predictions[i], '; label: ', test_labels[i])


# img = image.load_img('data/test/0/0.jpg', target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
 

# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)
# print classes
