import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from Pacman.PacmanState import PacmanState
from Pacman.feature_extractor import FeatureExtractor
from Pacman.finding_path_util import find_shortest_path
from env.FrozenLakeMDP import frozenLake
from env.CliffWorldMDP import CliffWorld

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class LinearApproximationAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, movable_positions_graph):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self.NUM_OF_FEATURES = 2
        self.weights = [random.random()] * self.NUM_OF_FEATURES
        self.features = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.movable_positions_graph = movable_positions_graph
        self.feature_extractor = FeatureExtractor(movable_positions_graph)

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        # return self._qvalues[state][action]
        result = 0
        state_action_features = self.get_features_for_state_action(state, action)
        for i in range(0, self.NUM_OF_FEATURES):
            result += self.weights[i] * state_action_features[i]
        return result

    # ---------------------START OF YOUR CODE---------------------#

    def get_state_with_moved_pacman(self, state, action):
        new_state = copy.deepcopy(state)
        if action == LEFT:
            new_state.pacman_position[1] -= 1
        if action == RIGHT:
            new_state.pacman_position[1] += 1
        if action == UP:
            new_state.pacman_position[0] -= 1
        if action == DOWN:
            new_state.pacman_position[0] += 1
        return new_state

    def get_features_for_state_action(self, state: PacmanState, action):
        """
        Works best with only these to basic features
        """

        if (state, action) not in self.features:
            state_action_features = []

            state_with_moved_pacman = self.get_state_with_moved_pacman(state, action)

            state_action_features.append(self.feature_extractor.get_shortest_ghost_distance(state_with_moved_pacman))

            state_action_features.append(self.feature_extractor.get_shortest_food_distance(state_with_moved_pacman))

            # state_action_features.append(self.feature_extractor.get_is_ghost_close(state_with_moved_pacman))

            # state_action_features.append(self.feature_extractor.get_is_food_close(state_with_moved_pacman))

            self.features[(state, action)] = state_action_features
        return self.features[(state, action)]

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #

        max_value = self.get_qvalue(state, possible_actions[0])
        for action in possible_actions[1:]:
            qvalue = self.get_qvalue(state, action)
            if qvalue > max_value:
                max_value = qvalue

        return max_value

    def update(self, state, action, reward, next_state):

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #

        error = reward + gamma * self.get_value(next_state) - self.get_qvalue(state, action)
        # print("Error =", error)
        for i in range(0, self.NUM_OF_FEATURES):
            self.weights[i] = self.weights[i] + learning_rate * error * \
                              self.get_features_for_state_action(state, action)[i]
        # print("weights =", self.weights)

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

        best_action_value = self.get_qvalue(state, possible_actions[0])
        best_actions = [possible_actions[0]]
        for action in possible_actions[1:]:
            value = self.get_qvalue(state, action)
            if value > best_action_value:
                best_actions = [action]
                best_action_value = value
            elif value == best_action_value:
                best_actions.append(action)

        best_action = random.choice(best_actions)

        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

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

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0
