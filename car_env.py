#!/usr/bin/env python3
import glob
import os
import sys
import time
import sys
import numpy as np
import cv2
import math
import gym
import copy
import random
from keras.applications.xception import Xception, preprocess_input
from keras import Sequential
from keras.layers import GlobalAveragePooling2D

from global_planner import GlobalRoutePlanner
from frenet_frame import FrenetFrame
from geo2enu import geodetic_to_enu

"Starting script for any carla programming"

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

STATE_SHAPE = (1,)
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 60
MAX_LAT = 1.2
SHOW_PREVIEW  = False    ## for debugging purpose

MIN_REWARD = -0.2

LEFT_BIAS = 1.75
RIGHT_BIAS = 5.25

def copy_tranfrom(transform):
    loc = carla.Location()
    loc.x = transform.location.x
    loc.y = transform.location.y
    loc.z = transform.location.z


    rot = carla.Rotation()
    rot.roll = transform.rotation.roll
    rot.pitch = transform.rotation.pitch
    rot.yaw = transform.rotation.yaw

    return carla.Transform(loc, rot)

def scale_data(a, min, max, dtype=np.float32):
    """ Scales an array of values from specified min, max range to -1 to 1
        Optionally specify the data type of the output (default is uint8)
    """
    return ((((a - min) / float(max - min)) - 0.5)*2).astype(dtype)


class CarEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0   ## full turn for every single time
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    WAYPT_RESOLUTION = 2.0
    WAYPT_VISUALIZE = True

    def __init__(self, config = None):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world('Town03')
        self.map = self.world.get_map()   ## added for map creating
        self.blueprint_library = self.world.get_blueprint_library()    
        self.model_3 = self.blueprint_library.filter("model3")[0]  ## grab tesla model3 from 
        self.grp = GlobalRoutePlanner(self.map, self.WAYPT_RESOLUTION)
        self.frenet = FrenetFrame()
        self.spawn_points = self.map.get_spawn_points()

        self.action_space = gym.spaces.Discrete(5)

        # car location
        self.vehicle_location = None

        # image 
        self.lidar_image = None
        self.image_shape = (300, 300)

        self.collision_hist = []    
        self.actor_list = []

        # training
        self.training = True
        
        
    def generate_global_path(self):

        # generate global path
        route_idx = random.randint(0, 2)

        if route_idx == 0:
            # 1, LT
            start = self.map.get_waypoint(carla.Location(5, -120)).transform.location
            end = self.map.get_waypoint(carla.Location(-30, -143)).transform.location
        elif route_idx == 1:
            # 1, S
            start = self.map.get_waypoint(carla.Location(5, -80)).transform.location
            end = self.map.get_waypoint(carla.Location(5, -130)).transform.location
        elif route_idx == 2:
            # 2, RT
            start = self.map.get_waypoint(carla.Location(30, -6)).transform.location
            end = self.spawn_points[200].location

        self.waypoints = self.grp.trace_route(start, end)[:20]
        if len(self.waypoints) < 2:
            raise ValueError("number of generated waypoints are less than 2")


        if self.WAYPT_VISUALIZE:
            for i, waypoint in enumerate(self.waypoints):
                self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0) if i != 14 else carla.Color(r=255, g=0, b=0), life_time=10,
                                        persistent_lines=True)
        # set distance calculator
        self.frenet.set_waypoints(self.waypoints)

    def spawn_obstacle(self):

        obs_idx = random.randint(6, 12)
        obstacle_waypoint = self.waypoints[obs_idx]

        self.obstacle_point = copy_tranfrom(obstacle_waypoint.transform)
        self.obstacle_point.location.z += 2

        self.obstacle = self.world.spawn_actor(self.model_3, self.obstacle_point)
        self.actor_list.append(self.obstacle)
        

    def reset(self):
        # remove previous actors
        for actor in self.actor_list:
            actor.destroy()

        self.collision_hist = []    
        self.actor_list = []

        # global path
        self.generate_global_path()

        # spawn obstacle
        self.spawn_obstacle()

        # spawn the vehcle
        self.spawn_point = self.waypoints[0].transform
        self.spawn_point.location.z += 2
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints
        self.actor_list.append(self.vehicle)

        # add lidar
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(90000))
        lidar_bp.set_attribute('rotation_frequency',str(40))
        lidar_bp.set_attribute('range',str(20))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        self.lidar_sen = self.world.spawn_actor(lidar_bp,lidar_transform,attach_to=self.vehicle)
        self.actor_list.append(self.lidar_sen)
        self.lidar_sen.listen(lambda point_cloud: self.process_lidar(point_cloud))
        


        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # initially passing some commands seems to help with time. Not sure why.
        time.sleep(1)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        # gnss
        if not self.training:
            gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
            gnss_location = carla.Location(1.5, 0, 0)
            gnss_rotation = carla.Rotation(0,0,0)
            gnss_transform = carla.Transform(gnss_location,gnss_rotation)

            gnss_bp.set_attribute("sensor_tick",str(0.1))
            self.gnss = self.world.spawn_actor(gnss_bp,gnss_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
            self.actor_list.append(self.gnss)
            self.gnss.listen(lambda gnss: self.process_gnss(gnss))
        else:
            self.vehicle_location = self.vehicle.get_transform().location

        # add collision sensor
        colsensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, colsensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        # reset image input and check sensor is working
        self.lidar_image = None
        while self.lidar_image is None or self.vehicle_location is None:  
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))


        # get frenet coordinate
        long, lat = self.frenet.get_distance(self.vehicle_location)

        # observation
        observation = [self.lidar_image, np.array([lat])]

        return observation

    def process_lidar(self, point_cloud):
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        side_range=(-15, 15)
        fwd_range=(-15,15)
        res=0.1
        min_height = -2.73
        max_height = 1.27

        x_lidar = data[:, 0]
        y_lidar = data[:, 1]
        z_lidar = data[:, 2]
        # r_lidar = points[:, 3]  # Reflectance
        

        # INDICES FILTER - of values within the desired rectangle
        # Note left side is positive y axis in LIDAR coordinates
        ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
        ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
        indices = np.argwhere(np.logical_and(ff,ss)).flatten()

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
        y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                        # will be inverted later

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img -= int(np.floor(side_range[0]/res))
        y_img -= int(np.floor(fwd_range[0]/res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = np.clip(a = z_lidar[indices],
                            a_min=min_height,
                            a_max=max_height)

        # RESCALE THE HEIGHT VALUES - to be between the range -1 to 1
        pixel_values  = scale_data(pixel_values, min=min_height, max=max_height)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        x_max = int((side_range[1] - side_range[0])/res)
        y_max = int((fwd_range[1] - fwd_range[0])/res)
        im = np.zeros([y_max, x_max], dtype=np.float32)
        im[-y_img, x_img] = pixel_values # -y because images start from top left

        self.lidar_image = im.reshape(*self.image_shape, 1)

    def step(self, action):
        '''
        For now let's just pass steer left, straight, right
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=-0.7*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.35*self.STEER_AMT))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer= 0.0 ))   
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.35*self.STEER_AMT))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0.7*self.STEER_AMT))

        # self.world.debug.draw_string(front_wheel_loc, 'O', draw_shadow=False,
        #                                 color=carla.Color(r=255, g=0, b=0), life_time=3,
        #                                 persistent_lines=True)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if self.training:
           self.vehicle_location = self.vehicle.get_transform().location 

        long, lat = self.frenet.get_distance(self.vehicle_location)
        # long, lat = self.frenet.get_distance(self.vehicle_location)
        # print("car's longitudinal distance from start: ", long)
        # print("distance between car and path", lat)
        # print("loc: ", self.vehicle_location)

        center_waypt = self.get_closest_waypt(self.vehicle_location)
        right_waypt = center_waypt.get_right_lane()
        left_waypt = center_waypt.get_left_lane()

        # self.world.debug.draw_string(center_waypt.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=255, g=255, b=255), life_time=5,
        #                                     persistent_lines=True)

        # if not right_waypt is None: 
        #     self.world.debug.draw_string(right_waypt.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=255, g=0, b=0), life_time=5,
        #                                     persistent_lines=True)
        # else:
        #     print("no right waypt")

        # if not left_waypt is None:
        #     self.world.debug.draw_string(left_waypt.transform.location, 'O', draw_shadow=False,
        #                                     color=carla.Color(r=0, g=0, b=255), life_time=5,
        #                                     persistent_lines=True)
        # else:
        #     print("no left waypt")        

        if len(self.collision_hist) != 0:
                done = True
                reward = MIN_REWARD
        else:

            if lat > 0 :
                reward = (RIGHT_BIAS - abs(lat)) / RIGHT_BIAS

            else:
                reward = (LEFT_BIAS - abs(lat)) / LEFT_BIAS
                

            if reward < MIN_REWARD:
                done = True
            else:
                # if reward > 0:
                #     angle_diff = self.get_angle_error() / np.pi
                #     reward = reward * (1 - angle_diff)

                done = False

        time_now = time.time()

        if self.episode_start + SECONDS_PER_EPISODE < time_now:  ## when to stop
            done = True
        elif self.get_closest_waypt_idx(self.vehicle_location) == len(self.waypoints) - 1:
            done = True

        # observation
        observation = [self.lidar_image, np.array([lat])]
        return observation, reward, done, dict()

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_gnss(self, gnss):
        x, y, z = geodetic_to_enu(gnss.latitude, gnss.longitude, gnss.altitude)
        self.vehicle_location = carla.Location(x, -y, z)

    # def get_angle_error(self):
    #     """ calculate absolute value of angle between path direction and vehcile heading direction. \n
    #     """
    #     vehicle_transform = self.vehicle.get_transform()
    #     vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
    #     path_yaw = self.frenet.get_path_direction(vehicle_transform.location)

    #     diff = (vehicle_yaw - path_yaw + np.pi) % (2 * np.pi) - np.pi

    #     return np.abs(diff)

    def get_closest_waypt(self, location):
        """
        location : carla.Location
        """

        return min(self.waypoints, key = lambda waypt : waypt.transform.location.distance(location))

    def get_closest_waypt_idx(self, location):
        """
        location : carla.Location
        """

        return min(range(len(self.waypoints)), key = lambda i : self.waypoints[i].transform.location.distance(location))
    
    def get_obs_shape(self):
        """ return shape of the observation space """
        return self.observation_space.shape

    def get_num_actions(self):
        """ return number of actions """
        return self.action_space.n


if __name__ == "__main__":

    import yaml

    import ray
    from ray.tune.registry import register_env
    from ray.tune.logger import pretty_print 
    from ray.rllib.agents import dqn

    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = CarEnv(config)

    # ray.rllib.utils.check_env(env)

    # register_env("multienv", lambda config: CarEnv(config))
    
    dqn_config = dqn.DEFAULT_CONFIG.copy()
    dqn_config.update(config)
    
    trainer = dqn.DQNTrainer(config=dqn_config, env=CarEnv)
    for i in range(100):
        result = trainer.train()
        print(pretty_print(result))



