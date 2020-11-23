from styx_msgs.msg import TrafficLight

import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.utils.generic_utils import CustomObjectScope


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        path = os.path.dirname(os.path.realpath(__file__))
        
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model(path+'/model_last.h5')
            #print(self.model.summary())

        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
            
        self.labels = {
            0: TrafficLight.RED,
            1: TrafficLight.YELLOW,
            2: TrafficLight.GREEN
        }


    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        #TODO implement light color prediction
              
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))
        img_array = image.img_to_array(img_rgb)
        img_batch = np.expand_dims(img_array, axis=0)
        
        img_preprocessed = keras.applications.mobilenet.preprocess_input(img_batch)

        with self.graph.as_default():
            predictions = self.model.predict(img_preprocessed)
            if predictions[0][np.argmax(predictions[0])] > 0.8: # make shure, the prediction probability is high enough
                return self.labels[np.argmax(predictions[0])]


        return TrafficLight.UNKNOWN
