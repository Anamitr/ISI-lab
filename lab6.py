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


class Agent(object):
    def __init__(self, action_size, policy, predict):
        self.G = 0
        self.gamma = 0.99
        self.action_size = action_size
        self.policy = policy
        self.predict = predict
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

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.action_size])
        actions[np.arange(len(action_memory)), action_memory] = 1

        self.G = np.array(get_cumulative_rewards(reward_memory, self.gamma))

        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return cost


env = gym.make("CartPole-v0").env
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001

input = Input(shape=(state_size,))
advantages = Input(shape=[1])
dense1 = Dense(64, activation='relu')(input)
dense2 = Dense(64, activation='relu')(dense1)
probs = Dense(action_size, activation='softmax')(dense2)

def custom_loss(y_true, y_pred):
    out = K.clip(y_pred, 1e-8, 1 - 1e-8)
    log_lik = y_true * K.log(out)

    return K.sum(-log_lik * advantages)

policy = Model(input=[input, advantages], output=[probs])

policy.compile(optimizer=Adam(lr=0.0005), loss=custom_loss)

predict = Model(input=[input], output=[probs])

agent = Agent(action_size, policy, predict)


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
        if done: break

    agent.learn()

    return reward


for i in range(100):

    rewards = [generate_session() for _ in range(100)]  # generate new sessions

    print("mean reward:%.3f" % (np.mean(rewards)))

    if np.mean(rewards) > 300:
        print("You Win!")
        break
