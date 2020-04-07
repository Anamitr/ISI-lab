class PacmanState:
    # movable_positions = []
    # fields_setup = []
    # pacman_position = tuple
    # ghosts_positions = []
    # food_positions = []

    def __init__(self, movable_positions, fields_setup):
        self.movable_positions = movable_positions
        self.fields_setup = fields_setup
        self.pacman_position = tuple
        self.ghosts_positions = []
        self.food_positions = []

        for i in range(0, len(fields_setup)):
            if 'p' in fields_setup[i]:
                self.pacman_position = movable_positions[i]
            if 'g' in fields_setup[i]:
                self.ghosts_positions.append(movable_positions[i])
            if '*' in fields_setup[i]:
                self.food_positions.append(movable_positions[i])
        pass
