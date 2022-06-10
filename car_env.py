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
MIN_REWARD = -2

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

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low = - MAX_LAT * 2, high = MAX_LAT * 2, shape=(1,), dtype=np.float32)

        # car location
        self.vehicle_location = None

        # image 
        self.image_feature = None
        self.image_shape = (self.im_height, self.im_width, 3)
        self.image_process_model = Sequential([Xception(weights="imagenet", include_top=False, input_shape=self.image_shape),
                                               GlobalAveragePooling2D()])
        self.image_process_model.trainable = False

        self.collision_hist = []    
        self.actor_list = []

        # training
        self.training = True
        
        
    def generate_global_path(self):

        # generate global path
        route_idx = random.randint(0, 3)
        

        if route_idx == 0:
            # 1, LT
            start = self.map.get_waypoint(carla.Location(5, y=-120)).transform.location
            end = self.map.get_waypoint(carla.Location(-30, -143)).transform.location
        elif route_idx == 1:
            # 1, S
            start = self.map.get_waypoint(carla.Location(5, -80)).transform.location
            end = self.map.get_waypoint(carla.Location(5, -130)).transform.location
        elif route_idx == 2:
            # 2, S
            start = self.map.get_waypoint(carla.Location(10, -80)).transform.location
            end = self.map.get_waypoint(carla.Location(10, -130)).transform.location
        elif route_idx == 3:
            # 2, RT
            start = self.map.get_waypoint(carla.Location(-20, -130)).transform.location
            end = self.map.get_waypoint(carla.Location(-5, -100)).transform.location

        self.waypoints = self.grp.trace_route(start, end)[:20]
        if len(self.waypoints) < 2:
            raise ValueError("number of generated waypoints are less than 2")


        if self.WAYPT_VISUALIZE:
            for i, waypoint in enumerate(self.waypoints):
                self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0) if i != 14 else carla.Color(r=255, g=0, b=0), life_time=1000,
                                        persistent_lines=True)
        # set distance calculator
        self.frenet.set_waypoints(self.waypoints)

    def spawn_obstacle(self):

        obs_idx = random.randint(7, 14)
        obs_bias = random.randint(1, 10)
        obstacle_waypoint = self.waypoints[obs_idx]

        left = obstacle_waypoint.get_left_lane()
        right = obstacle_waypoint.get_right_lane()

        BOTH = carla.LaneChange.Both
        RIGHT = carla.LaneChange.Right
        LEFT = carla.LaneChange.Left

        if obs_bias > 6 and obs_bias < 9:
            if not left is None and left.lane_type == carla.LaneType.Driving:
                obstacle_waypoint = left
            else:
                obstacle_waypoint = right
        else:
            if not right is None and right.lane_type == carla.LaneType.Driving:
                obstacle_waypoint = right
            else:
                obstacle_waypoint = left

        self.obstacle_point = copy_tranfrom(obstacle_waypoint.transform)
        self.obstacle_point.location.z += 2

        self.obstacle = self.world.spawn_actor(self.model_3, self.obstacle_point)  ## changed for adding waypoints
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

        # add rgb camera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")  ## fov, field of view

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
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
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        # reset image input and check sensor is working
        self.image_feature = None
        while self.image_feature is None or self.vehicle_location is None:  
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))


        # get frenet coordinate
        long, lat = self.frenet.get_distance(self.vehicle_location)

        # observation
        observation = np.append(self.image_feature, np.array([lat]))

        return observation

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
            # width of current road 
            center_width = center_waypt.lane_width
            reward = center_width / 2 - abs(lat)

            # print("center: ", center_width)

            if lat > 0 and not right_waypt is None and right_waypt.lane_type == carla.LaneType.Driving:
                right_width = right_waypt.lane_width
                reward = (center_width + right_width) / 2 - abs(lat)

            elif lat <= 0 and not left_waypt is None and left_waypt.lane_type == carla.LaneType.Driving:
                left_width = left_waypt.lane_width
                reward = (center_width + left_width) / 2 - abs(lat)
                

            if reward < 0:
                done = True
            else:
                # if reward > 0:
                #     angle_diff = self.get_angle_error() / np.pi
                #     reward = reward * (1 - angle_diff)

                done = False

        

        if self.episode_start + SECONDS_PER_EPISODE < time.time():  ## when to stop
            done = True

        if self.get_closest_waypt_idx(self.vehicle_location) == len(self.waypoints) - 1:
            done = True
            reward = (self.episode_start + SECONDS_PER_EPISODE - time.time()) * 5 * center_width / 2

        # observation
        observation = np.append(self.image_feature, np.array([lat]))
        return observation, reward, done, dict()

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)

        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]

        self.front_camera = i3  ## remember to scale this down between 0 and 1 for CNN input purpose
        preprocessed = preprocess_input(self.front_camera.astype(np.float32)).reshape(-1, *self.image_shape)
        self.image_feature = self.image_process_model(preprocessed)

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



