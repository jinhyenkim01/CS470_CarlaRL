import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
# import tensorflow.keras.backend as backend
from car_env import CarEnv
from dqn import DQNAgent




MODEL_PATH = '/home/jongk/dqn_carla/models/Xception__2080.62max_1345.42avg__320.74min__1654588697.model'

if __name__ == '__main__':

    # # Memory fraction
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Create environment
    env = CarEnv()

    # Load the model
    agent = DQNAgent(env.get_obs_shape(), env.get_num_actions())
    model = agent.create_model()
    model.load_weights(MODEL_PATH)

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.ones((1, ) + env.get_obs_shape()))

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            # cv2.imshow(f'Agent - preview', current_state)
            # cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)