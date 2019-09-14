import rospy
from yaw_controller import *
from pid import *
from lowpass import *

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        self.yaw_controller = YawController(wheel_base,steer_ratio,0.1, max_lat_accel,max_steer_angle)

        kp = 0.3 # the parametre for the pid controller
        ki = 0.1
        kd = 0.0
        mn = 0.0 # min throttle value
        mx = 0.2 #max throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # use the low pass filter to filter the noise of the velocity message
        tau = 0.5 #1/(2pi*tau)a = cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau,ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decle_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # we need to check if dbw is on since , the self driving mode 
        # could be toggled  like when waiting for traffic light it will be off
        # if does not checked ,then pid intergral part would accumulate error
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0,0.0,0.0

        # find the velocity difference as well as the steering angle
        current_vel = self.vel_lpf.filt(current_vel)
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel,current_vel)
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        # to find the time difference sample time
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0
            brake = 400 # # N*m - to hold the car in place if we are stopped at a light. Accelaration - 1m/s^2

        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error,self.decle_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N8m

        return throttle, brake, steer
