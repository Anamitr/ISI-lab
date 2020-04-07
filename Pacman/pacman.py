import pygame
import random

from Pacman.PacmanState import PacmanState
from Pacman.anagram_cal import permutations_with_partial_repetitions

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class Pacman:

    def __init__(self, board):
        """
            Pacman:
        """

        self.player_image = pygame.transform.scale(pygame.image.load("assets/pacman.png"), (30, 30))
        self.ghost_image = pygame.transform.scale(pygame.image.load("assets/red_ghost.png"), (30, 30))

        self.board = board
        self.cell_size = 60
        pygame.init()
        self.screen = pygame.display.set_mode((len(board[0]) * self.cell_size, (len(board) * self.cell_size)))
        self.player_pos = dict()
        self.ghosts = []
        self.foods = []
        self.score = 0
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

        self.init_foods = self.foods.copy()
        self.init_ghosts = self.ghosts.copy()
        self.__draw_board()

    def reset(self):
        """ resets state of the environment """
        self.init_foods = self.foods.copy()
        self.init_ghosts = self.ghosts.copy()
        self.init_player_pos = self.player_pos.copy()
        self.score = 0

    def get_all_states(self):
        """ return a list of all possible states """
        num_of_ghosts = 0
        num_of_food = 0
        # num_of_empty
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

        # Wygeneruj permutacje tak żeby nie trwało to miliona lat

        # all_permutations = list(itertools.permutations(list_of_objects))
        # all_permutations = ["".join(perm) for perm in itertools.permutations(list_of_objects)]
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

        pacman_states = []
        for perm in all_permutations:
            pacman_states.append(PacmanState(movable_positions, perm))

        return pacman_states

    def is_terminal(self, state: PacmanState):
        """
        return true if state is terminal or false otherwise
        state is terminal when ghost is on the same position as pacman or all capsules are eaten
        """
        result = False
        is_last_food = False
        if len(state.food_positions) == 1:
            is_last_food = True
        for field in state.fields_setup:
            if 'p' in field and 'g' in field:
                result = True
            elif is_last_food and 'p' in field and '*' in field:
                result = True
        return result

    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        actions = [LEFT, DOWN, RIGHT, UP]
        possible_actions = []

        width = len(self.board[0])
        height = len(self.board)

        if state.pacman_position[1] > 0:
            if self.board[ state.pacman_position[0]][ state.pacman_position[1] - 1] != 'w':
                possible_actions.append(LEFT)
        if state.pacman_position[1] + 1 < width:
            if self.board[ state.pacman_position[0]][ state.pacman_position[1] + 1] != 'w':
                possible_actions.append(RIGHT)
        if state.pacman_position[0] > 0:
            if self.board[ state.pacman_position[0] - 1][ state.pacman_position[1]] != 'w':
                possible_actions.append(UP)
        if state.pacman_position[0] + 1 < height:
            if self.board[ state.pacman_position[0] + 1][ state.pacman_position[1]] != 'w':
                possible_actions.append(DOWN)
        return possible_actions

    def get_next_states(self, state, action):
        """
        return a set of possible next states and probabilities of moving into them
        assume that ghost can move in each possible direction with the same probability, ghost cannot stay in place
        """
        pass

    def get_reward(self, state, action, next_state):
        """
        return the reward after taking action in state and landing on next_state
            -1 for each step
            10 for eating capsule
            -500 for eating ghost
            500 for eating all capsules
        """
        pass

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
            reward = 500
            self.score += 500

        return self.__get_state(), reward, len(self.foods) == 0, self.score

    def __draw_board(self):
        '''
        Function displays current state of the board. Do not change this code
        '''

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
                                pygame.Rect(food['x'] * self.cell_size + 25, food['y'] * self.cell_size + 25, 10, 10))

        pygame.display.flip()

    def __get_state(self):
        '''
        Function returns current state of the game
        :return: state
        '''
        pass


board = ["   g",
         " ww ",
         " w* ",
         " ww ",
         "p   "]

clock = pygame.time.Clock()

pacman = Pacman(board)

'''
Apply Value Iteration algorithm for Pacman
'''

# Tests
all_states = pacman.get_all_states()
pacman.get_possible_actions(all_states[300])
pass
#

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    '''
    move pacman according to the policy from Value Iteration
    '''

    # to be done

    # state, reward, done, score = pacman.step(action)
    # print(score)
    clock.tick(5)
