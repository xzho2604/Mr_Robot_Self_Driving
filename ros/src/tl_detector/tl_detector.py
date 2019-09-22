#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []


        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # base_waypoints will be called only once since the base way point
        # would not change ,so it will be stroed in the class
        self.waypoints = waypoints
        if not self.waypoints_2d:
            # just to get the coordinates of the waypoints (x,y)
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] \
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d) # constructa KDTree using the 2d waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state: # if state change we start the counter
            self.state_count = 0
            self.state = state
        # since the classifier could be unstable and keep changing all the time
        # we will only take action of the classifier stays unchanged for a certain
        # threshold of classification loops 
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state # record the last state
            light_wp = light_wp if state == TrafficLight.RED else -1 # we only interested in the red light
            self.last_wp = light_wp # record the previous traffic light state
            self.upcoming_red_light_pub.publish(Int32(light_wp)) # publish the confident traffic light state
        else:
            # if we are not confident just publish the previous traffic light state
            self.upcoming_red_light_pub.publish(Int32(self.last_wp)) 
        self.state_count += 1

    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx 



    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #return light.state # get the light state provided by the simulator

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        # change from ros image message to cv rgb image
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8") 
        status = self.light_classifier.get_classification(cv_image) 
        #rospy.loginfo("[traffic] ",tl_color," traffic light detected")

        #Get classification
        return status 

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x,
                                                   self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints) # number of visible points ahead of the car
            # loop through all possible stop line and find the one closest visible stopline 
            for i , light in enumerate(self.lights): 
                line = stop_line_positions[i] # get the stop line waypoint index
                # get the closest waypoint index of this traffic light coordinates
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                d = temp_wp_idx - car_wp_idx


                if d >= 0 and d < diff: # check to see if stop line is ahead and visible infront of the car
                    #rospy.loginfo("[debug] light: {}, car_wp_indx: {}, wp_indx: {}, d: {}".format(
                    #    i, car_wp_idx, temp_wp_idx, d))
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
                    break

        if closest_light and diff <100:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state # return the stop line index is there is visible and the state of the light

        return -1, TrafficLight.UNKNOWN # return -1 if there is no visible traffice light

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
