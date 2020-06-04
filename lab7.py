
from collections import deque
import gym
import numpy as np
import random
import time as tm

from keras import backend as K
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam


class ActorCriticAgent:
    def __init__(self, action_size, actor, critic, policy):
        self.gamma = 0.99
        self.action_size = action_size

        self.actor = actor
        self.critic = critic
        self.policy = policy
        self.action_space = [i for i in range(action_size)]

    def get_action(self, state):
        state = state[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        critic_value_next_state = self.critic.predict(next_state)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma * critic_value_next_state * (1 - int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.action_size])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, delta], actions, verbose=0)

        self.critic.fit(state, target, verbose=0)


env = gym.make("CartPole-v0").env
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
alpha_learning_rate = 0.0001
beta_learning_rate = 0.0005

input = Input(shape=(state_size,))
delta = Input(shape=[1])
dense1 = Dense(64, activation='relu')(input)
probs = Dense(action_size, activation='softmax')(dense1)
values = Dense(1, activation='linear')(dense1)


def custom_loss(y_true, y_pred):
    out = K.clip(y_pred, 1e-8, 1 - 1e-8)
    log_lik = y_true * K.log(out)

    return K.sum(-log_lik * delta)


actor_model = Model(input=[input, delta], output=[probs])

actor_model.compile(optimizer=Adam(lr=alpha_learning_rate), loss=custom_loss)

critic_model = Model(input=[input], output=[values])

critic_model.compile(optimizer=Adam(lr=beta_learning_rate), loss='mean_squared_error')

policy = Model(input=[input], output=[probs])


agent = ActorCriticAgent(action_size, actor_model, critic_model, policy)

for i in range(100):
    score_history = []

    for i in range(100):
        done = False
        score = 0
        state = env.reset()
        for t in range(1000):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if (done):
                break
        score_history.append(score)

    print("mean reward:%.3f" % (np.mean(score_history)))

    if np.mean(score_history) > 300:
        print("You Win!")
        break
