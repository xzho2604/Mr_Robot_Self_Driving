#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32 
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # basic_waypoints are static and will not change so will 
        # be loaded only subscribed only once
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # get the traffic light info to adjust final published way points
        # in case need to slow down and stop at the traffic light stop line
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        # TODO: Add other member variables you need below
        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # get the closest waypoints
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        # get the currenty position of the car
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # find the index of the points that is cloese to the current car position 
        # we ask to return 1 closet point KDTree returns the position as well as the index of
        # the cloese waypoint, index would be the same order as we construct the KDTree
        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        # check the closet way point is ahead or behind the car
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # equation for hyperplane through closest_coords
        # test to see if the prev_vect and cl_vect are the same direction as 
        # from current to cl_vect
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect,pos_vect-cl_vect)

        if val < 0: # the direction is opposite the closet point is behind the car
            # we need to use the next points insteand , modula len to wrap around
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
        """
        lane = Lane()
        lane.header = self.base_lane.header # same format we do not need hearder anyway
        # no need to worry about the greater than the base waypoint len since python slice will
        # just slice to the end if the lengths is greater 
        lane.waypoints = self.base_lane.waypoints[closest_idx: closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)
        """

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        
        # if there is no traffic light or the traffic light is further away
        # than th e furthest planing rout we just publish the base_waypoint ahead of the car
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints 
        # else there is a red traffic light in the planning route and need to deaccelarte
        # to stop at the traffic light stop line
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints,closest_idx)

        return lane

    def decelerate_waypoints(self,waypoints, closest_idx):
        temp = []
        for i , wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            # since the current waypoint of the car is based on the centre of the car
            # we want to back 2 waypoints to let the front of the car to stop at the
            # stop line instead of the centre of the car
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2,0)
            # calculate distance of the current waypoint to the stoping point
            dist = self.distance(waypoints,i,stop_idx) 
            
            # based on the distance to the stoping point we fit in a sqrt curve
            # for smooth deacceleration  
            # could use just a linear factor as well
            vel = math.sqrt(2*MAX_DECEL * dist)
            if vel < 1:
                vel = 0.0
            
            # when the distance is large the sqrt computed velocity could be 
            # large as well, so we need to cap it with the original velocity
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp



    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg # store the car's pose about 50hz


    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # base_waypoints will be called only once since the base way point
        # would not change ,so it will be stroed in the class
        self.base_lane = waypoints
        if not self.waypoints_2d:
            # just to get the coordinates of the waypoints (x,y)
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] \
                                 for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d) # constructa KDTree using the 2d waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
