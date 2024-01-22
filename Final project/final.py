"""
A Minimal Deep Q-Learning Implementation (minDQN)

Running this code will render the agent solving the CartPole environment using OpenAI gym. Our Minimal Deep Q-Network is approximately 150 lines of code. In addition, this implementation uses Tensorflow and Keras and should generally run in less than 15 minutes.

Usage: python3 minDQN.py
"""

import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, LSTM, Reshape, Dropout, Conv2D, Flatten
from keras.optimizers import Adam

from IPython.utils import io
import imageio

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = gym.make("ALE/KungFuMaster-v5", render_mode='rgb_array')
env.metadata['render_fps'] = 2
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 300
test_episodes = 100

# def agent(state_shape, action_shape):
#     """ The agent maps X-states to Y-actions
#     e.g. The neural network output is [.1, .7, .1, .3]
#     The highest value 0.7 is the Q-Value.
#     The index of the highest action (0.7) is action #1.
#     """
#     learning_rate = 0.001
#     init = tf.keras.initializers.HeUniform()
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
#     model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
#     model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
#     model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
#     return model

def agent(state_shape, action_shape):
    learn_rate = 0.001
    model = Sequential()
    model.add(Input(state_shape))
    model.add(Conv2D(filters = 32,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(filters = 64,kernel_size = (4,4),strides = 2,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = 1,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Flatten())
    model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Dense(action_shape, activation = 'linear'))
    optimizer = Adam(learn_rate)
    model.compile(optimizer, loss=tf.keras.losses.Huber())
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] if len(transition[0]) == 210 else transition[0][0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        if len(observation) == 2:
            X.append(observation[0])
        else:
            X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def test(env, model, num, record=True):
    episode_rewards = None if record else []

    for episode in range(1 if record else 100):
        total_test_rewards = 0
        observation = env.reset()
        done = False
        frames = [] if record else None

        while not done:

            if record:
                frame = env.render()
                frames.append(frame)
            else:
                env.render()

            if isinstance(observation, tuple):
                observation = observation[0]

            encoded = observation
            encoded_reshaped = encoded.reshape([1, *encoded.shape])
            predicted = model.predict(encoded_reshaped).flatten()
            action = np.argmax(predicted)
            new_observation, reward, done, termination, info = env.step(action)
            done = done or termination
            observation = new_observation
            total_test_rewards += reward

        if episode_rewards is not None:
            episode_rewards.append(total_test_rewards)

        if done:
            print('Total test rewards: {} in episode {}'.format(total_test_rewards, episode))
            total_test_rewards = 0

        # Save frames as a video if recording is enabled
        if record:
            video_path = f'test_video_{num}.mp4'
            imageio.mimsave(video_path, frames)
            print(f"Video saved at {video_path}")

    if episode_rewards is not None:
        print('Mean of rewards: {}'.format(np.mean(episode_rewards)))
        print('Min of rewards: {}'.format(np.min(episode_rewards)))
        print('Max of rewards: {}'.format(np.max(episode_rewards)))
        print('Standard deviation of rewards: {}'.format(np.std(episode_rewards)))

    
def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent(env.observation_space.shape, env.action_space.n)
    # Target Model (updated every 100 steps)
    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        print("Episode: ", episode)
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            if True:
                env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                
                if isinstance(observation, tuple):
                    observation = observation[0]

                encoded = observation
                encoded_reshaped = encoded.reshape([1, *encoded.shape])
                # print(episode, observation.shape)
                # print(observation)
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, termination, info = env.step(action)
            done = done or termination
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                # print('Training loop: ', steps_to_update_target_model)
                train(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        if episode % 15 == 0:
            test(env, model, episode)
            model.save_weights('model_weights.h5')
    
    test(env, model, 'final')
    test(env, model, 'final', record=False)

    env.close()

if __name__ == '__main__':
    main()