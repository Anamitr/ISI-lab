from collections import deque, defaultdict
import gym
import numpy as np
import random
import time as tm
from tensorflow import keras
from tensorflow.keras import layers

from keras import Model, Input
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

from env.FrozenLakeMDP import frozenLake
from env.FrozenLakeMDPExtended import frozenLakeExtended


class DQNAgent:
    def __init__(self, action_size, learning_rate, model: Model, get_legal_actions):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = learning_rate
        self.model = model

        self.get_legal_actions = get_legal_actions
        # self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))

    # def get_qvalue(self, state, action):
    #     """ Returns Q(state,action) """
    #     return self._qvalues[state][action]
    #
    # def set_qvalue(self, state, action, value):
    #     """ Sets the Qvalue for [state,action] to the given value """
    #     self._qvalues[state][action] = value

    def remember(self, state, action, reward, next_state, done):
        # Function adds information to the memory about last action and its results
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # self.lower_epsilon()
        # agent parameters:
        epsilon = self.epsilon

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #

        best_action = self.get_best_action(state)
        chosen_action = best_action

        if random.uniform(0, 1) < epsilon:
            random_actions = possible_actions.copy()
            random_actions.remove(best_action)
            chosen_action = random.choice(random_actions if random_actions else [best_action])

        return chosen_action

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #

        # best_action_value = self.get_qvalue(state, possible_actions[0])
        # best_actions = [possible_actions[0]]
        # for action in possible_actions[1:]:
        #     value = self.get_qvalue(state, action)
        #     if value > best_action_value:
        #         best_actions = [action]
        #         best_action_value = value
        #     elif value == best_action_value:
        #         best_actions.append(action)
        #
        # best_action = random.choice(best_actions)
        #
        # return best_action

        self.model.predict(state)

        return np.argmax(self.model.predict(state))

    def lower_epsilon(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon >= self.epsilon_min:
            self.epsilon = new_epsilon

    def replay(self, batch_size):
        """
        Function learn network using randomly selected actions from the memory.
        First calculates Q value for the next state and choose action with the biggest value.
        Target value is calculated according to:
                Q(s,a) := (r + gamma * max_a(Q(s', a)))
        except the situation when the next action is the last action, in such case Q(s, a) := r.
        In order to change only those weights responsible for chosing given action, the rest values should be those
        returned by the network for state state.
        The network should be trained on batch_size samples.
        Also every time the function replay is called self.epsilon value should be updated according to equation:
        self.epsilon *= self.epsilon_decay
        """
        #
        # INSERT CODE HERE to train network
        #

        # batch_size = 32
        # if len(self.memory) < batch_size:
        #     return

        samples = random.sample(self.memory, batch_size)
        states_list = []
        targets_list = []
        for sample in samples:
            state, action, reward, next_state, done = sample
            states_list.append(state.flatten())
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            targets_list.append(target.flatten())

        states_array = np.array(states_list)  # .flatten() .reshape(batch_size, -1)
        targets_array = np.array(targets_list)  # .flatten() # .reshape(batch_size, -1)

        # self.model.fit(states_array, targets_array, epochs=1, verbose=0)
        # targets_array = to_categorical(targets_array)
        self.model.train_on_batch(states_array, targets_array)

        self.lower_epsilon()


def create_model(state_size, action_size, learning_rate):
    model = Sequential()
    # model.add(Input(shape=(state_size,), name="model_input"))
    model.add(Dense(16, input_dim=state_size, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(action_size))  # wyjście
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=learning_rate))

    # model = keras.Sequential([
    #     keras.Input(shape=(state_size)),
    #     layers.Dense(16, activation='relu'),
    #     layers.Dense(32, activation='relu'),
    #     layers.Dense(16, activation='relu'),
    #     layers.Dense(action_size),
    # ])
    # opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss='mse', optimizer=opt)
    return model


env = frozenLake("4x4")

state_size = env.get_number_of_states()
action_size = len(env.get_possible_actions(None))
learning_rate = 0.001

model = create_model(state_size, action_size, learning_rate)

agent = DQNAgent(action_size, learning_rate, model, get_legal_actions=env.get_possible_actions)

# agent.epsilon = 0.75

done = False
batch_size = 64
EPISODES = 1000
counter = 0


def reshape_state(state):
    return np.array([state]).reshape(1, 1)


for e in range(EPISODES):
    start = tm.time()
    summary = []
    for _ in range(100):
        total_reward = 0
        env_state = env.reset()

        #
        # INSERT CODE HERE to prepare appropriate format of the state for network
        #
        # state = env_state.reshape(1, 2)
        # state = reshape_state(env_state)
        # state = to_categorical(env_state, num_classes=state_size)
        state = np.zeros((1, state_size))
        state[0][env_state] = 1

        for time in range(500):
            action = agent.get_action(state)
            next_state_env, reward, done, _ = env.step(action)
            total_reward += reward

            #
            # INSERT CODE HERE to prepare appropriate format of the next state for network
            #
            # next_state = reshape_state(next_state)
            # next_state = to_categorical(next_state, num_classes=state_size)
            next_state = np.zeros((1, state_size))
            next_state[0][next_state_env] = 1

            # add to experience memory
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        #
        # INSERT CODE HERE to train network if in the memory is more samples then size of the batch
        #
        if len(agent.memory) > batch_size:
            agent.replay(64)

        summary.append(total_reward)

    end = tm.time()
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}\ttime = {}".format(e, np.mean(summary), agent.epsilon,
                                                                                end - start))
    if np.mean(total_reward) > 0.9:
        print("You Win!")
        break
