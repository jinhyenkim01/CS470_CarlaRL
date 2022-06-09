import glob
from hashlib import new
import os
import sys
import random
import time
import numpy as np
import math
import datetime
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
from threading import Thread, Lock

from tqdm import tqdm

from car_env import CarEnv
import gym

from scalar_writer import ScalarWriter

IM_WIDTH = 640
IM_HEIGHT = 480

UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -2
SAVE_MIN_REWARD = 10

class DQNAgent:
    def __init__(self, observation_shape, num_actions):
        self.num_actions = num_actions
        self.observation_shape = observation_shape

        # hyperparameters
        self.discount = 0.998
        self.learning_rate = 0.001

        # batch size
        self.minibatch_size = 64
        self.trainbatch_size = 64
        self.predictbatch_size = 1

        # replay buffer size
        self.replay_buffer_size = 2000
        self.min_replay_buffer_size = 1000

        # lock
        self.replay_buffer_lock = Lock()
        self.target_model_lock = Lock()

        # model for action-value function
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()

        # replay memory
        self.replay_memory = deque(maxlen=self.replay_buffer_size)

        
        self.target_update_counter = 0
        self.last_logged_episode = 0
        self.training_initialized = False

    def my_model(self):

        return tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_shape=self.observation_shape, activation = "relu"),
            tf.keras.layers.Dense(24, activation = "relu"),
            tf.keras.layers.Dense(self.num_actions, activation="linear")]
        )

    def xception_model(self):

        base_model = Xception(weights="imagenet", include_top=False, input_shape=self.observation_shape)

        return tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_actions, activation="linear")
        ])

    def create_model(self):
        model = self.my_model()
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        return model

    def update_target_model(self):
        self.target_model_lock.acquire()
        self.target_model.set_weights(self.model.get_weights())
        self.target_model_lock.release()

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_buffer_lock.acquire()
        self.replay_memory.append(transition)
        self.replay_buffer_lock.release()

    def train_model(self):
        if len(self.replay_memory) < self.min_replay_buffer_size:
            return

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        self.replay_buffer_lock.acquire()
        current_states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        new_current_states = np.array([transition[3] for transition in minibatch])
        done = np.array([transition[4] for transition in minibatch])
        self.replay_buffer_lock.release()

        current_qs_list = self.model(current_states).numpy()

        # get Q value from target net
        self.target_model_lock.acquire()
        future_qs_list_tnet = self.target_model(new_current_states).numpy()
        self.target_model_lock.release()

        # future_qs_list_qnet = self.model.predict(new_current_states,  self.minibatch_size)
        # max_a_idx = np.argmax(future_qs_list_qnet, axis = 1)

        # bellman update
        max_future_q = np.max(future_qs_list_tnet, axis = 1)
        new_q = rewards + (1 - done) * self.discount * max_future_q
        current_qs_list[np.arange(len(actions)), actions] = new_q

        X = current_states
        y = current_qs_list

        history = self.model.fit(X, y, 
                       batch_size= self.minibatch_size, 
                       verbose=0, 
                       shuffle=False)

        return history.history['accuracy'][0]

    def get_qs(self, state):
        """ predict Q value for a state """
        state = np.array(state)
        return self.model.predict(state.reshape(-1, *state.shape))[0]
        

class DQNTrainer:
    def __init__(self, env):
        """
        initialize dqn trainer. 
        env : Gym.Env. its observation space should be gym.Box and action space gym.Discrete
        """
        self.env = env

        self.observation_shape = env.observation_space.shape
        self.action_size = env.action_space.n
        self.agent = DQNAgent(self.observation_shape, self.action_size)

        # hyperparmeter
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.num_episodes = 2000
        self.update_target_every = 1

        # render option
        self.render = False

        self.terminate = False
        self.training_initialized = False

        self.FPS = 60

        # tf summary writer
        self.scalar_writer = ScalarWriter()
        self.aggregate_stats_every = 10

        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

    def train_mode_th(self):
        X = np.random.uniform(size=(1, ) + self.observation_shape).astype(np.float32)
        y = np.random.uniform(size=(1, self.action_size)).astype(np.float32)
        self.agent.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            accuracy = self.agent.train_model()
            if not accuracy is None:
                self.scalar_writer.record("accuracy", accuracy)
            time.sleep(0.01)

    def start_train(self):

        trainer_thread = Thread(target=self.train_mode_th, daemon=True)
        trainer_thread.start()

        while not self.training_initialized:
            time.sleep(0.01)

        ep_rewards = []
        
        for episode in tqdm(range(1, self.num_episodes + 1), ascii=True, unit='episodes'):

            # Restarting episode - reset episode reward and step number
            episode_reward = 0

            # Reset environment and get initial state
            current_state = self.env.reset()

            # Reset flag and start iterating until episode ends
            done = False

            # Play for given number of seconds only
            while True:

                if self.render:
                    self.env.render()
                
                action = self.get_action(current_state)
                new_state, reward, done, _ = self.env.step(action)

                # TODO : remove this part. it is only for cartpole
                # reward = 0.1 if not done else -1

                episode_reward += reward

                # Every step we update replay memory
                self.agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state

                if done:
                    break

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % self.aggregate_stats_every or episode == 1:        
                average_reward = sum(ep_rewards[-self.aggregate_stats_every:])/len(ep_rewards[-self.aggregate_stats_every:])
                min_reward = min(ep_rewards[-self.aggregate_stats_every:])
                max_reward = max(ep_rewards[-self.aggregate_stats_every:])

                # log data
                self.scalar_writer.record('avg_reward', average_reward, episode)
                self.scalar_writer.record('min_reward', min_reward, episode)
                self.scalar_writer.record('max_reward', max_reward, episode)
                self.scalar_writer.record('epsilon', self.epsilon, episode)

                # save model
                if min_reward >= SAVE_MIN_REWARD:
                    self.agent.model.save_weights(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            if episode % self.update_target_every == 0:
                self.agent.update_target_model()
            self.decay_epilon()

        self.terminate = True
        trainer_thread.join()
        self.agent.model.save_weights(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    def decay_epilon(self):
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_action(self, current_state):
        if np.random.random() > self.epsilon:
            # Get action from Q table
            qs = self.agent.get_qs(current_state)
            action = np.argmax(qs)
        else:
            # Get random action
            action = np.random.randint(0, self.action_size)
            time.sleep(1 / self.FPS)
            # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
        return action

if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [MIN_REWARD]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Create agent and environment
    env = CarEnv()
    # agent = DQNAgent(env.get_obs_shape(), env.get_num_actions())
    # env = gym.make('CartPole-v1')

    trainer = DQNTrainer(env)
    trainer.start_train()