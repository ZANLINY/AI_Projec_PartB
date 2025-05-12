import math

from referee.game.player import PlayerColor

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        se;f.parent = parent
        self.children = []
        self.total_reward = [0.0, 0.0]
        self.visited = 0
        self.untried_moves = state.get_valid_moves()
    
    def select_child(self):
        exploration_weight = math.sqrt(2)

        # get current player
        current_player_idx = 0 if self.state.board.turn_color == PlayerColor.RED else 1

        def ucb(child) -> float:
            if child.visited == 0:
                return float('inf') # first explore unvisited node
            
            # calculate ucb from player view
            exploit = child.total_reward[current_player_idx] / child.visited
            explore = exploration_weight + math.sqrt(math.log(self.visited) / child.visited)

            return exploit + explore   
        
        return max(self.children, key = ucb)
    
    def expand(self):
        action = self.untried_moves.pop()
        next_state = self.state.move(action)
        child = Node(next_state, parent=self)
        self.children.append(child)
        return child
    
    def update(self, reward):
        self.visited += 1
        if self.state.board.turn_color == PlayerColor.RED:
            self.total_reward[0] += reward
            self.total_reward[1] -= reward
        else:
            self.total_reward[0] -= reward
            self.total_reward[1] += reward



class MCTS:
    def __init__(self, state):
        pass

class GameState:
    def __init__(self, last_move, board):
        self.last_move = last_move
        self.board = board
        
if __name__ == "__main__":
    pass