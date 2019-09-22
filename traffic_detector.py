import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
#from models.object_detection.utils import visualization_utils as vis_util
from keras.applications.mobilenet import preprocess_input

class TLClassifier(object):
    def __init__(self):

        self.signal_classes = ['Green', 'Red','Yellow']
        #self.signal_classes = ['Red','Green','Yellow']
        self.signal_status = None
        self.tl_box = None
        #self.cls_model = load_model('tl_model_1.h5') # keras classification model
        self.cls_model = load_model('my_model.h5') # keras classification model
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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:np.array
            image (cv::Mat): cropped image containing the traffic light

        Returns:
            [green,red,yellow]
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img_resize = preprocess_input(image)
        img_resize = np.expand_dims(img_resize, axis=0).astype('float32')

        predict = self.cls_model.predict(img_resize)
        predict = np.squeeze(predict, axis =0)
        # Get color classification
        tl_color = self.signal_classes[np.argmax(predict)]
        print("[info] ",tl_color,', Classification confidence:', predict[np.argmax(predict)])

        # TrafficLight message
        self.signal_status = tl_color

        return self.signal_status


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

# ========================================================================================
if __name__ == '__main__':

        tl_cls =TLClassifier()
        TEST_IMAGE_PATHS= glob(os.path.join('light_test_images/', '*.jpg'))

        for i, image_path in enumerate(TEST_IMAGE_PATHS):
            print('[info] Detecting test images...')

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # change to RBG channel
            img_copy = np.copy(img) # for cropped iamge
            print("[info] Image shape",img.shape)

            start = time.time()
            b = tl_cls.object_box(img, visual=False)
            end = time.time()
            print('[info] Detection time: {0:.2f}'.format(end-start))

            # If there is no detection or low-confidence detection
            if b ==[]:
               print ('Traffic light unknown')
               plt.figure(figsize=(9,6))
               plt.imshow(img)
               plt.show()
            else:
               cv2.rectangle(img,(b[1],b[0]),(b[3],b[2]),(0,255,0),2)
               plt.figure(figsize=(9,6))
               plt.imshow(img)
               plt.show()

               # show the cropped image of detected traffic light
               img_copy = cv2.resize(img_copy[b[0]:b[2], b[1]:b[3]], (32, 32))
               tl_cls.get_classification(img_copy)
               plt.figure(figsize=(5,5))
               plt.imshow(img_copy)
               plt.show()





