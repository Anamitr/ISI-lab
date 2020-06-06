import numpy as np


class PacmanState:

    def __init__(self, movable_positions=None, fields_setup=None):
        if fields_setup is None:
            fields_setup = []
        if movable_positions is None:
            movable_positions = []
        self.movable_positions = movable_positions
        self.fields_setup = fields_setup  # this field is not updated, watch out where you are using it
        self.walls_positions = []
        self.pacman_position = tuple
        self.ghosts_positions = []
        self.food_positions = []

        for i in range(0, len(fields_setup)):
            if 'p' in fields_setup[i]:
                self.pacman_position = list(movable_positions[i])
            if 'g' in fields_setup[i]:
                self.ghosts_positions.append(list(movable_positions[i]))
            if '*' in fields_setup[i]:
                self.food_positions.append(list(movable_positions[i]))
        pass

    def set_values(self, pacman_position: list, ghosts_positions: list, food_positions: list, walls_positions: list, movable_positions: list = None):
        self.walls_positions = walls_positions
        self.pacman_position = pacman_position
        self.ghosts_positions = ghosts_positions
        self.food_positions = food_positions
        self.movable_positions = movable_positions

    def get_as_np_array(self):
        state_list = self.pacman_position
        [[state_list.append(pos) for pos in food_pos] for food_pos in self.food_positions]
        [[state_list.append(pos) for pos in ghost_pos] for ghost_pos in self.ghosts_positions]
        return np.array(state_list)

    def get_as_triple_one_hot(self):
        if self.movable_positions is None:
            raise ValueError("Movable positions cannot be None for getting state as one hot")
        walls_pos_one_hot = [1 if x in [wall for wall in self.walls_positions] else 0 for x in self.movable_positions]
        pacman_pos_one_hot = [1 if tuple(self.pacman_position) == x else 0 for x in self.movable_positions]
        ghosts_positions_one_hot = [1 if x in [tuple(ghost) for ghost in self.ghosts_positions] else 0 for x in self.movable_positions]
        food_positions_one_hot = [1 if x in [tuple(food) for food in self.food_positions] else 0 for x in self.movable_positions]
        # return [pacman_pos_one_hot, ghosts_positions_one_hot, food_positions_one_hot]
        return [walls_pos_one_hot, pacman_pos_one_hot, food_positions_one_hot]


    def __hash__(self):
        return hash(str(self.pacman_position) + str(self.ghosts_positions) + str(self.food_positions))

    def __eq__(self, other):
        return (str(self.pacman_position) + str(self.ghosts_positions) + str(self.food_positions)) == (
                str(other.pacman_position) + str(other.ghosts_positions) + str(other.food_positions))

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)
