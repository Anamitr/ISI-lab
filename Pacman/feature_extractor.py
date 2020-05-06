from Pacman.PacmanState import PacmanState
from Pacman.finding_path_util import find_shortest_path


class FeatureExtractor:

    def __init__(self, movable_positions_graph):
        self.movable_positions_graph = movable_positions_graph

    def get_shortest_ghost_distance(self, state_with_moved_pacman: PacmanState):
        shortest_pacman_ghost_distance = len(
            find_shortest_path(self.movable_positions_graph, tuple(state_with_moved_pacman.pacman_position),
                               tuple(state_with_moved_pacman.ghosts_positions[0])))
        for i in range(1, len(state_with_moved_pacman.ghosts_positions)):
            pacman_ghost_distance = len(
                find_shortest_path(self.movable_positions_graph, tuple(state_with_moved_pacman.pacman_position),
                                   tuple(state_with_moved_pacman.ghosts_positions[i])))
            if pacman_ghost_distance < shortest_pacman_ghost_distance:
                shortest_pacman_ghost_distance = pacman_ghost_distance
        return float(shortest_pacman_ghost_distance / len(self.movable_positions_graph))

    def get_shortest_food_distance(self, state_with_moved_pacman: PacmanState):
        shortest_pacman_food_distance = len(
            find_shortest_path(self.movable_positions_graph, tuple(state_with_moved_pacman.pacman_position),
                               tuple(state_with_moved_pacman.food_positions[0])))
        for i in range(1, len(state_with_moved_pacman.food_positions)):
            pacman_food_distance = len(
                find_shortest_path(self.movable_positions_graph, tuple(state_with_moved_pacman.pacman_position),
                                   tuple(state_with_moved_pacman.food_positions[i])))
            if pacman_food_distance < shortest_pacman_food_distance:
                shortest_pacman_food_distance = pacman_food_distance
        return float(1 - shortest_pacman_food_distance / len(self.movable_positions_graph))  # should this be 1 - ?

    def get_is_ghost_close(self, state_with_moved_pacman: PacmanState):
        is_ghost_close = 0
        for ghost_pos in state_with_moved_pacman.ghosts_positions:
            if find_shortest_path(self.movable_positions_graph, tuple(state_with_moved_pacman.pacman_position),
                                  tuple(ghost_pos)) == 2:
                is_ghost_close = 1
                break
        return float(is_ghost_close)

    def get_is_food_close(self, state_with_moved_pacman: PacmanState):
        is_food_close = 0
        for food_pos in state_with_moved_pacman.food_positions:
            if find_shortest_path(self.movable_positions_graph, tuple(state_with_moved_pacman.pacman_position),
                                  tuple(food_pos)) == 2:
                is_food_close = 1
                break
        return float(is_food_close)
