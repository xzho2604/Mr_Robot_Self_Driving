from styx_msgs.msg import TrafficLight
import rospy

import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model

import numpy as np
import cv2

import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
#from models.object_detection.utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # ----------------------------------------------------------------------------
        # START traffic light model params
        self.signal_classes_str = ['Green', 'Red','Yellow']
        #self.signal_classes = ['Red','Green','Yellow']
        self.signal_classes = [TrafficLight.GREEN,TrafficLight.RED,TrafficLight.YELLOW]
        self.signal_status = None 
        self.tl_box = None
        self.cls_model = load_model('my_model.h5') # keras classification model
        self.graph = tf.get_default_graph()
        self.detect_thresh = 0.1 # detection confidence threshhold 
        self.light_class_num = 10 # classification number of traffice light in mobile net

        # object detection model for traffic light
        detect_model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # load frozen tensorflow detection model and initialize the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

        # END traffic light model params
        # ----------------------------------------------------------------------------


    # helper function for object detection
    def object_box(self, image, visual=False):
        """Detect trafffic light and draw the bounding boxes

        Args:np.array(heigh, width,3)
            image: camera image

        Returns: list(int)
            bounding box coordinates [x_left, y_up, x_right, y_down]
        """
        with self.detection_graph.as_default():
            # run the pretrained model for traffic light detection
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).tolist()
            scores = np.squeeze(scores)

            # find index of the traffic light class 10, if exits else return -1
            idx = classes.index(self.light_class_num) if self.light_class_num in classes else -1
            print("[info ]idx:", idx, "socre:", scores[idx])

            # if traffic light not found return empty box
            # else found traffic light filter out weak detection 
            if idx == -1 or scores[idx] <= self.detect_thresh: return []

            # find traffic light find bounding boxes since boxes output are percentage of the 
            # whole frame need to normalised it to nearest pixels
            heigh, width = image.shape[:2]
            box =np.array(list(map(lambda a,b: int(a*b),boxes[idx],[heigh,width]*2)))
            box_h = box[2] - box[0]
            box_w = box[3] - box[1]
            ratio = box_h/(box_w + 0.01)

            # sanity check after getting the box
            if(box_h < 10 or box_w < 10 or ratio < 1.5): return []

            # else good detection 
            self.tl_box = box
        return box


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #rospy.loginfo("[debug] shape of the cropped image {} type: {}".format(image.shape,type(image)))
        #image = np.asarray(image, dtype="int32")
        img_copy = np.copy(image)

        b = self.object_box(image) # detect traffic light
        if b == []: # no traffic light detected
            rospy.loginfo("[traffic] Unknown")
            return TrafficLight.UNKNOWN

        
        # if there is traffic light detected classify 
        #rospy.loginfo("[debug] box: {} {} {} {} ".format(b[0],b[2],b[1],b[3]))
        img_copy = cv2.resize(img_copy[b[0]:b[2], b[1]:b[3]], (32, 32))

        image = img_copy.astype(np.float32)
        img_resize = preprocess_input(image)
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')

        #rospy.loginfo("[debug] shape of the resized {} type: {}".format(img_resize.shape,type(img_resize)))
        
        with self.graph.as_default():
            predict = self.cls_model.predict(img_resize)
        predict = np.squeeze(predict, axis =0)
        # Get color classification
        tl_color = self.signal_classes_str[np.argmax(predict)]
        rospy.loginfo("[traffic] {}".format(tl_color))
        #print("[info] ",tl_color,', Classification confidence:', predict[np.argmax(predict)])

        # TrafficLight message
        self.signal_status = self.signal_classes[np.argmax(predict)]

        return self.signal_status

