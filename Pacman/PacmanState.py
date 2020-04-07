class PacmanState:

    def __init__(self, movable_positions=None, fields_setup=None):
        if fields_setup is None:
            fields_setup = []
        if movable_positions is None:
            movable_positions = []
        self.movable_positions = movable_positions
        self.fields_setup = fields_setup  # this field is not updated, watch out where you are using it
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

    def set_values(self, pacman_position: list, ghosts_positions: list, food_positions: list):
        self.pacman_position = pacman_position
        self.ghosts_positions = ghosts_positions
        self.food_positions = food_positions

    def __hash__(self):
        return hash(str(self.pacman_position) + str(self.ghosts_positions) + str(self.food_positions))

    def __eq__(self, other):
        return (str(self.pacman_position) + str(self.ghosts_positions) + str(self.food_positions)) == (
                str(other.pacman_position) + str(other.ghosts_positions) + str(other.food_positions))

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)
