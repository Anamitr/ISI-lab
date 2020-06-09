from collections import deque
import gym
import numpy as np
import random
import time as tm

from keras import Model, Input
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import keras.utils as K_utils


def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    based on https://github.com/yandexdataschool/Practical_RL/blob/spring20/week06_policy_based/reinforce_tensorflow.ipynb
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative rewards R(s,a) (a.k.a. G(s,a) in Sutton '16)
    R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute R_t = r_t + gamma*R_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """

    cumulative_rewards = []
    for i in range(len(rewards)):
        cum_reward = 0
        k = i
        for j in range(len(rewards)):
            if k + j < len(rewards):
                cum_reward += rewards[k + j] * (gamma ** j)
        cumulative_rewards.append(cum_reward)
    # print(cumulative_rewards)

    return cumulative_rewards


assert len(get_cumulative_rewards(range(100))) == 100
assert np.allclose(get_cumulative_rewards([0, 0, 1, 0, 0, 1, 0], gamma=0.9),
                   [1.40049, 1.5561, 1.729, 0.81, 0.9, 1.0, 0.0])
assert np.allclose(get_cumulative_rewards([0, 0, 1, -2, 3, -4, 0], gamma=0.5),
                   [0.0625, 0.125, 0.25, -1.5, 1.0, -4.0, 0.0])
assert np.allclose(get_cumulative_rewards([0, 0, 1, 2, 3, 4, 0], gamma=0), [0, 0, 1, 2, 3, 4, 0])


class REINFORCEAgent:
    def __init__(self, state_size, action_size, policy_model, predict_model):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # discount rate
        self.learning_rate = 0.001

        self.policy = policy_model
        self.predict = predict_model
        self.action_space = [i for i in range(action_size)]

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def get_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def remember(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def learn(self):
        state_memory_np_array = np.array(self.state_memory)
        action_memory_np_array = np.array(self.action_memory)
        reward_memory_np_array = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory_np_array), self.action_size])
        for i in range(len(action_memory_np_array)):
            actions[i][action_memory_np_array[i]] = 1

        cumulative_rewards = np.array(get_cumulative_rewards(reward_memory_np_array, self.gamma))

        cost = self.policy.train_on_batch([state_memory_np_array, cumulative_rewards], actions)

        self.clear_memory()

        return cost


env = gym.make("CartPole-v0").env
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001


def build_model():
    states_input = Input(shape=(state_size,))
    cumulative_reward_input = Input(shape=[1])
    dense1 = (Dense(32, activation="relu"))(states_input)
    dense2 = (Dense(64, activation="relu"))(dense1)
    dense3 = (Dense(32, activation="relu"))(dense2)
    probs = Dense(action_size, activation='softmax')(dense3)

    def custom_loss(y_true, y_pred):
        # y_pred_mean = K.mean(y_pred)
        # y_pred_std = K.std(y_pred)
        # y_pred_normalized = (y_pred - y_pred_mean) / y_pred_std
        # y_pred_normalized = K.clip(y_pred, 0, 1)
        # return -(learning_rate * cumulative_reward_input * y_true * y_pred_normalized)
        # return -(cumulative_reward_input * y_true * K.log(y_pred_normalized))
        # y_true * K.log(y_pred_normalized) - mnożymy akcję którą wybraliśmy przez oszacowanie tej akcji
        # Gradient Policy(s,a)
        return -(cumulative_reward_input * y_true * K.log(K.clip(y_pred, 0, 1)))

    policy_model = Model(input=[states_input, cumulative_reward_input], output=[probs])
    policy_model.compile(optimizer=Adam(lr=learning_rate), loss=custom_loss)
    predict_model = Model(input=[states_input], output=[probs])
    return policy_model, predict_model


policy_model, predict_model = build_model()

agent = REINFORCEAgent(state_size, action_size, policy_model, predict_model)


def generate_session(t_max=1000):
    """play env with REINFORCE agent and train at the session end"""

    reward = 0

    s = env.reset()

    for t in range(t_max):

        # chose action
        a = agent.get_action(s)

        new_s, r, done, info = env.step(a)

        # record session history to train later
        agent.remember(s, a, r)

        reward += r

        s = new_s
        if done:
            break

    agent.learn()

    return reward


for i in range(100):

    rewards = [generate_session() for _ in range(100)]  # generate new sessions

    print("mean reward:%.3f" % (np.mean(rewards)))

    if np.mean(rewards) > 300:
        print("You Win!")
        break
