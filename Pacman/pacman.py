import copy
import random
import numpy as np
import time as tm

import pygame

from Pacman import boards
from Pacman.PacmanState import PacmanState
from Pacman.anagram_cal import permutations_with_partial_repetitions
from Pacman.finding_path_util import find_shortest_path
from Pacman.linear_approximation import LinearApproximationAgent
from Pacman.pacman_util import load_obj, save_obj
from Pacman.pacman_value_iteration import value_iteration
from Pacman.qlearning import QLearningAgent
from deep_q_learning import DQNAgent

from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
actions = [LEFT, DOWN, RIGHT, UP]


class Pacman:

    def __init__(self, board):
        """
            Pacman:
        """

        self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))

        self.display_mode_on = True

        self.board = board
        self.cell_size = 60
        pygame.init()
        self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))
        self.player_pos = dict()
        self.ghosts = []
        self.foods = []
        self.walls = []
        self.score = 0
        self.movable_positions = self.get_movable_positions()
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x] == 'p':
                    self.player_pos['x'] = x
                    self.player_pos['y'] = y
                    self.init_player_pos = self.player_pos.copy()
                elif self.board[y][x] == 'g':
                    ghost = dict()
                    ghost['x'] = x
                    ghost['y'] = y
                    ghost['direction'] = random.choice([LEFT, DOWN])
                    self.ghosts.append(ghost)
                elif self.board[y][x] == '*':
                    food = dict()
                    food['x'] = x
                    food['y'] = y
                    self.foods.append(food)
                elif self.board[y][x] == 'w':
                    self.walls.append((y, x))

        self.init_foods = copy.deepcopy(self.foods)
        self.init_ghosts = copy.deepcopy(self.ghosts)
        self.__draw_board()

    def reset(self) -> PacmanState:
        """ resets state of the environment """
        self.foods = copy.deepcopy(self.init_foods)
        self.ghosts = copy.deepcopy(self.init_ghosts)
        self.player_pos = self.init_player_pos.copy()
        self.score = 0
        return self.__get_state()

    def get_all_states(self):
        """ return a list of all possible states """
        num_of_ghosts = 0
        num_of_food = 0
        movable_positions = []
        list_of_fields = ['p']
        list_of_objects = ['p']

        for y in range(0, len(board)):
            row = board[y]
            for x in range(0, len(row)):
                if board[y][x] == 'w':
                    continue
                else:
                    movable_positions.append((y, x))
                    if board[y][x] == 'g':
                        num_of_ghosts += 1
                        list_of_fields.append('g')
                        list_of_objects.append('g')
                    elif board[y][x] == '*':
                        num_of_food += 1
                        list_of_fields.append('*')
                        list_of_objects.append('*')
                    elif board[y][x] != 'p':
                        list_of_fields.append(' ')

        print("Found", num_of_ghosts, "ghosts and", num_of_food, "food")
        print("Movable positions:", movable_positions)
        print("Objects:", list_of_fields)
        positions_length = len(movable_positions)

        permutations_with_objects_in_different_places = permutations_with_partial_repetitions(list_of_fields)
        empty_list = [' '] * 15
        permutations_with_objects_in_one_place = [empty_list[:i] + ['pg*'] + empty_list[(i + 1):] for i in
                                                  range(0, positions_length)]

        other_permutations = []
        for object in list_of_objects:
            other_objects = ''.join([obj for obj in list_of_objects if obj != object])
            for i in range(0, positions_length):
                for j in range(0, positions_length):
                    if j != i:
                        new_perm = empty_list.copy()
                        new_perm[i] = object
                        new_perm[j] = other_objects
                        other_permutations.append(new_perm)

        all_permutations = permutations_with_objects_in_different_places + permutations_with_objects_in_one_place + other_permutations
        print("All permutations length:", len(all_permutations))
        # 2730 + 15 + 14 * 15 * 3
        # 2730 + 15 + 630 = 3375

        pacman_states = []
        for perm in all_permutations:
            pacman_states.append(PacmanState(movable_positions, perm))

        return pacman_states

    def get_movable_positions(self):
        movable_positions = []
        for y in range(0, len(board)):
            row = board[y]
            for x in range(0, len(row)):
                if board[y][x] == 'w':
                    continue
                else:
                    movable_positions.append((y, x))
        return movable_positions

    def is_terminal(self, state: PacmanState):
        """
        return true if state is terminal or false otherwise
        state is terminal when ghost is on the same position as pacman or all capsules are eaten
        """
        result = False
        is_last_food = False
        if len(state.food_positions) == 1:
            is_last_food = True

        for ghost_position in state.ghosts_positions:
            if state.pacman_position == ghost_position:
                result = True

        if is_last_food and state.pacman_position == state.food_positions[0]:
            result = True

        for wall in self.walls:
            if tuple(state.pacman_position) == wall:
                # print("Run into wall")
                result = True

        if len(self.foods) == 0:
            result = True

        return result

    # def get_possible_actions(self, state):
    #     """ return a tuple of possible actions in a given state """
    #     possible_actions = []
    #
    #     width = len(self.board[0])
    #     height = len(self.board)
    #
    #     if state.pacman_position[1] > 0:
    #         if self.board[state.pacman_position[0]][state.pacman_position[1] - 1] != 'w':
    #             possible_actions.append(LEFT)
    #     if state.pacman_position[1] + 1 < width:
    #         if self.board[state.pacman_position[0]][state.pacman_position[1] + 1] != 'w':
    #             possible_actions.append(RIGHT)
    #     if state.pacman_position[0] > 0:
    #         if self.board[state.pacman_position[0] - 1][state.pacman_position[1]] != 'w':
    #             possible_actions.append(UP)
    #     if state.pacman_position[0] + 1 < height:
    #         if self.board[state.pacman_position[0] + 1][state.pacman_position[1]] != 'w':
    #             possible_actions.append(DOWN)
    #     return possible_actions

    def get_possible_actions_from_state_as_array(self, state_array):
        possible_actions = []
        state_array_copy = copy.deepcopy(state_array)
        state_array_copy = state_array_copy.flatten()

        width = len(self.board[0])
        height = len(self.board)

        if state_array_copy[1] > 0:
            if self.board[state_array_copy[0]][state_array_copy[1] - 1] != 'w':
                possible_actions.append(LEFT)
        if state_array_copy[1] + 1 < width:
            if self.board[state_array_copy[0]][state_array_copy[1] + 1] != 'w':
                possible_actions.append(RIGHT)
        if state_array_copy[0] > 0:
            if self.board[state_array_copy[0] - 1][state_array_copy[1]] != 'w':
                possible_actions.append(UP)
        if state_array_copy[0] + 1 < height:
            if self.board[state_array_copy[0] + 1][state_array_copy[1]] != 'w':
                possible_actions.append(DOWN)
        return possible_actions

    def get_possible_actions(self, state):
        return [LEFT, DOWN, RIGHT, UP]

    def get_next_states(self, state: PacmanState, action):
        """
        return a set of possible next states and probabilities of moving into them
        assume that ghost can move in each possible direction with the same probability, ghost cannot stay in place
        """

        next_states = []
        next_state_base = copy.deepcopy(state)

        width = len(self.board[0])
        height = len(self.board)

        if action == LEFT and state.pacman_position[1] > 0:
            # if self.board[state.pacman_position[0]][state.pacman_position[1] - 1] != 'w':
            next_state_base.pacman_position[1] -= 1
        if action == RIGHT and state.pacman_position[1] + 1 < width:
            # if self.board[state.pacman_position[0]][state.pacman_position[1] + 1] != 'w':
            next_state_base.pacman_position[1] += 1
        if action == UP and state.pacman_position[0] > 0:
            # if self.board[state.pacman_position[0] - 1][state.pacman_position[1]] != 'w':
            next_state_base.pacman_position[0] -= 1
        if action == DOWN and state.pacman_position[0] + 1 < height:
            # if self.board[state.pacman_position[0] + 1][state.pacman_position[1]] != 'w':
            next_state_base.pacman_position[0] += 1

        # Assume there is only one ghost
        ghost_position = next_state_base.ghosts_positions[0]

        # Check if pacman didn't run into ghost
        if next_state_base.pacman_position == ghost_position:
            next_states.append(next_state_base)
        else:
            if ghost_position[1] > 0:
                if self.board[ghost_position[0]][ghost_position[1] - 1] != 'w':
                    new_next_state = copy.deepcopy(next_state_base)
                    new_next_state.ghosts_positions[0][1] -= 1
                    next_states.append(new_next_state)
            if ghost_position[1] + 1 < width:
                if self.board[ghost_position[0]][ghost_position[1] + 1] != 'w':
                    new_next_state = copy.deepcopy(next_state_base)
                    new_next_state.ghosts_positions[0][1] += 1
                    next_states.append(new_next_state)
            if ghost_position[0] > 0:
                if self.board[ghost_position[0] - 1][ghost_position[1]] != 'w':
                    new_next_state = copy.deepcopy(next_state_base)
                    new_next_state.ghosts_positions[0][0] -= 1
                    next_states.append(new_next_state)
            if ghost_position[0] + 1 < height:
                if self.board[ghost_position[0] + 1][ghost_position[1]] != 'w':
                    new_next_state = copy.deepcopy(next_state_base)
                    new_next_state.ghosts_positions[0][0] += 1
                    next_states.append(new_next_state)

        prob = 1 / len(next_states)
        next_states_dict = {}
        for next_state in next_states:
            next_states_dict[next_state] = prob

        return next_states_dict

    def get_reward(self, state, action, next_state):
        """
        return the reward after taking action in state and landing on next_state
            -1 for each step
            10 for eating capsule
            -500 for eating ghost
            500 for eating all capsules
        """
        result = 0

        for ghost_position in state.ghosts_positions:
            if state.pacman_position == ghost_position:
                return -500

        is_last_food = False
        if len(state.food_positions) == 1:
            is_last_food = True

        if is_last_food:
            if state.pacman_position == state.food_positions[0]:
                return 500
        else:
            for food_position in state.food_positions:
                if state.pacman_position == food_position:
                    return 50

        return -1

    def step(self, action):
        '''
        Function apply action. Do not change this code
        :returns:
        state - current state of the game
        reward - reward received by taking action (-1 for each step, 10 for eating capsule, -500 for eating ghost, 500 for eating all capsules)
        done - True if it is end of the game, False otherwise
        score - temporarily score of the game, later it will be displayed on the screen
        '''

        width = len(self.board[0])
        height = len(self.board)

        reward = 0

        # move player according to action

        if action == LEFT and self.player_pos['x'] > 0:
            if self.board[self.player_pos['y']][self.player_pos['x'] - 1] != 'w':
                self.player_pos['x'] -= 1
        if action == RIGHT and self.player_pos['x'] + 1 < width:
            if self.board[self.player_pos['y']][self.player_pos['x'] + 1] != 'w':
                self.player_pos['x'] += 1
        if action == UP and self.player_pos['y'] > 0:
            if self.board[self.player_pos['y'] - 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] -= 1
        if action == DOWN and self.player_pos['y'] + 1 < height:
            if self.board[self.player_pos['y'] + 1][self.player_pos['x']] != 'w':
                self.player_pos['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                # print("Run into ghost")
                self.score -= 500
                reward = -500
                self.__draw_board()
                return self.__get_state(), reward, True, self.score

        # check if player eats food

        for food in self.foods:
            if food['x'] == self.player_pos['x'] and food['y'] == self.player_pos['y']:
                self.score += 10
                reward = 10
                self.foods.remove(food)
                break
        else:
            self.score -= 1
            reward = -1

        # move ghosts
        for ghost in self.ghosts:
            moved = False
            ghost_moves = [LEFT, RIGHT, UP, DOWN]
            if ghost['x'] > 0 and self.board[ghost['y']][ghost['x'] - 1] != 'w':
                if ghost['direction'] == LEFT:
                    if RIGHT in ghost_moves:
                        ghost_moves.remove(RIGHT)
            else:
                if LEFT in ghost_moves:
                    ghost_moves.remove(LEFT)

            if ghost['x'] + 1 < width and self.board[ghost['y']][ghost['x'] + 1] != 'w':
                if ghost['direction'] == RIGHT:
                    if LEFT in ghost_moves:
                        ghost_moves.remove(LEFT)
            else:
                if RIGHT in ghost_moves:
                    ghost_moves.remove(RIGHT)

            if ghost['y'] > 0 and self.board[ghost['y'] - 1][ghost['x']] != 'w':
                if ghost['direction'] == UP:
                    if DOWN in ghost_moves:
                        ghost_moves.remove(DOWN)
            else:
                if UP in ghost_moves:
                    ghost_moves.remove(UP)

            if ghost['y'] + 1 < height and self.board[ghost['y'] + 1][ghost['x']] != 'w':
                if ghost['direction'] == DOWN:
                    if UP in ghost_moves:
                        ghost_moves.remove(UP)
            else:
                if DOWN in ghost_moves:
                    ghost_moves.remove(DOWN)

            ghost['direction'] = random.choice(ghost_moves)

            if ghost['direction'] == LEFT and ghost['x'] > 0:
                if self.board[ghost['y']][ghost['x'] - 1] != 'w':
                    ghost['x'] -= 1
            if ghost['direction'] == RIGHT and ghost['x'] + 1 < width:
                if self.board[ghost['y']][ghost['x'] + 1] != 'w':
                    ghost['x'] += 1
            if ghost['direction'] == UP and ghost['y'] > 0:
                if self.board[ghost['y'] - 1][ghost['x']] != 'w':
                    ghost['y'] -= 1
            if ghost['direction'] == DOWN and ghost['y'] + 1 < height:
                if self.board[ghost['y'] + 1][ghost['x']] != 'w':
                    ghost['y'] += 1

        for ghost in self.ghosts:
            if ghost['x'] == self.player_pos['x'] and ghost['y'] == self.player_pos['y']:
                self.score -= 500
                reward = -500
                self.__draw_board()
                return self.__get_state(), reward, True, self.score

        self.__draw_board()

        if len(self.foods) == 0:
            print("V", end='')
            reward = 500
            self.score += 500

        state = self.__get_state()
        # Nie zwraca się 500 za wszystkie zjedzone
        return state, reward, self.is_terminal(state), self.score

    def __draw_board(self):
        '''
        Function displays current state of the board. Do not change this code
        '''
        if self.display_mode_on:
            self.screen.fill((0, 0, 0))

            y = 0

            for line in board:
                x = 0
                for obj in line:
                    if obj == 'w':
                        color = (0, 255, 255)
                        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, 60, 60))
                    x += 60
                y += 60

            color = (255, 255, 0)
            # pygame.draw.rect(self.screen, color, pygame.Rect(self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15, 30, 30))
            self.screen.blit(self.player_image,
                             (self.player_pos['x'] * self.cell_size + 15, self.player_pos['y'] * self.cell_size + 15))

            color = (255, 0, 0)
            for ghost in self.ghosts:
                # pygame.draw.rect(self.screen, color, pygame.Rect(ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15, 30, 30))
                self.screen.blit(self.ghost_image,
                                 (ghost['x'] * self.cell_size + 15, ghost['y'] * self.cell_size + 15))

            color = (255, 255, 255)

            for food in self.foods:
                pygame.draw.ellipse(self.screen, color,
                                    pygame.Rect(food['x'] * self.cell_size + 25, food['y'] * self.cell_size + 25, 10,
                                                10))

            pygame.display.flip()

    def __get_state(self):
        '''
        Function returns current state of the game
        :return: state
        '''
        walls_positions = self.walls
        pacman_position = [self.player_pos['y'], self.player_pos['x']]
        ghosts_positions = [[ghost_pos['y'], ghost_pos['x']] for ghost_pos in self.ghosts]
        food_positions = [[food_pos['y'], food_pos['x']] for food_pos in self.foods]
        # num_of_fields = len(board) * len(board[0])

        movable_positions = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                movable_positions.append((i, j))

        state = PacmanState()
        state.set_values(pacman_position, ghosts_positions, food_positions, walls_positions, movable_positions)

        return state

    def get_state(self):
        return self.__get_state()

    def turn_off_display(self):
        self.display_mode_on = False

    def turn_on_display(self):
        self.display_mode_on = True

    def get_field_moves_graph(self):
        width = len(self.board[0])
        height = len(self.board)

        graph = {}
        state = PacmanState()
        for movable_pos in self.movable_positions:
            state.pacman_position = movable_pos
            # graph[movable_pos] = self.get_possible_actions(state)
            possible_next_pos = []
            if state.pacman_position[1] > 0:
                if self.board[state.pacman_position[0]][state.pacman_position[1] - 1] != 'w':
                    possible_next_pos.append((state.pacman_position[0], state.pacman_position[1] - 1))
            if state.pacman_position[1] + 1 < width:
                if self.board[state.pacman_position[0]][state.pacman_position[1] + 1] != 'w':
                    possible_next_pos.append((state.pacman_position[0], state.pacman_position[1] + 1))
            if state.pacman_position[0] > 0:
                if self.board[state.pacman_position[0] - 1][state.pacman_position[1]] != 'w':
                    possible_next_pos.append((state.pacman_position[0] - 1, state.pacman_position[1]))
            if state.pacman_position[0] + 1 < height:
                if self.board[state.pacman_position[0] + 1][state.pacman_position[1]] != 'w':
                    possible_next_pos.append((state.pacman_position[0] + 1, state.pacman_position[1]))
            graph[movable_pos] = possible_next_pos
        return graph

    # def get_state_size(self):
    #     pacman_state_size = 2
    #     for ghost in self.ghosts:
    #         pacman_state_size += 2
    #     for food in self.foods:
    #         pacman_state_size += 2
    #     return pacman_state_size

    def get_state_size(self):
        return len(board) * len(board[0]) * 4


board = boards.board

clock = pygame.time.Clock()

pacman = Pacman(board)
pacman.reset()


def play_and_display(env, agent):
    env.turn_on_display()
    total_reward = 0
    env_state = env.reset()

    state = np.array([np.array(env_state.get_as_triple_one_hot()).flatten()])

    for time in range(1000):
        action = agent.get_action(state)
        next_state_env, reward, done, _ = env.step(action)
        clock.tick(5)
        total_reward += reward

        next_state = np.array([np.array(next_state_env.get_as_triple_one_hot()).flatten()])

        state = next_state
        if done:
            break

    return total_reward


'''
Deep Q-Learning
'''

state_size = pacman.get_state_size()
action_size = 4
learning_rate = 0.001

model = Sequential()
model.add(Dense(128, input_dim=state_size, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(action_size))  # wyjście
model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=learning_rate))

env = pacman

print(board)

DISPLAY_MODE = False

if DISPLAY_MODE:
    agent = DQNAgent(action_size, learning_rate, model, env=pacman,
                     get_legal_actions=pacman.get_possible_actions, epsilon_start=0.02)
    pacman.turn_on_display()
    results_path = 'results/'
    weights_file = results_path + 'board12_weights'
    model.load_weights(weights_file)
    for i in range(10):
        play_and_display(pacman, agent)
else:
    pacman.turn_off_display()
    agent = DQNAgent(action_size, learning_rate, model, env=pacman,
                     get_legal_actions=pacman.get_possible_actions)

    done = False
    batch_size = 64
    EPISODES = 50
    counter = 0
    game_counter = 0
    MAX_MOVES_PER_GAME = 1000
    mean_rewards = []
    for e in range(EPISODES):
        start = tm.time()
        summary = []
        for _ in range(100):
            total_reward = 0
            env_state = env.reset()

            #
            # INSERT CODE HERE to prepare appropriate format of the state for network
            #
            state = np.array([np.array(env_state.get_as_triple_one_hot()).flatten()])

            for time in range(MAX_MOVES_PER_GAME):
                action = agent.get_action(state)
                next_state_env, reward, done, _ = env.step(action)
                total_reward += reward

                #
                # INSERT CODE HERE to prepare appropriate format of the next state for network
                #
                next_state = np.array([np.array(next_state_env.get_as_triple_one_hot()).flatten()])

                # add to experience memory
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    game_counter += 1
                    break

            #
            # INSERT CODE HERE to train network if in the memory is more samples then size of the batch
            #
            if len(agent.memory) > batch_size:
                agent.replay(64)

            summary.append(total_reward)

        end = tm.time()
        mean_reward = np.mean(summary)
        mean_rewards.append(mean_reward)
        print("\nAchieved rewards:", summary)
        print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}\ttime = {:.3f}\tgame num: {}\tbest mean reward = {:.3f}".format(e, mean_reward,
                                                                                                      agent.epsilon,
                                                                                                      end - start,
                                                                                                      game_counter,
                                                                                                      max(mean_rewards)))
        if np.mean(summary) > 300:
            print("You Win!")
            break

'''
- One hot
- Ściany jak dziury we frozenLake - wpadasz, koniec gry poniewaz wczesniej moglismy kazac
  agentowi wybrac z mozliwych akcji, teraz siec neuronowa musi się sama nauczyć nie wchodzić w ściany
- testy najprostszego do trudniejszych środowisk
'''

# Wiecej warstw (konwolucyjne)
# otoczyc scianami?

# przedebugowac jak teraz dziala z duchem
# jak we frozen lake zachowuje sie wobec wejscia w sciane?
# jak idzie w 8x8

# Czy karac za wejscie w sciane?
# Jak za wejscie w sciane nagroda 0 to uczy sie wchodzenia w sciane zeby go duszek nie zjadl
# MZ - wejscie w sciane to brak zmiany stanu i duch go dalej przepycha
# Moze mala nagroda za chodzenie zeby zachecic go do eksploracji?
