import math
import random
from referee.game import PlayerColor, Board, Coord, Direction, \
    Action, MoveAction, GrowAction
from typing import List, Dict, Set, FrozenSet, Tuple, Optional
import copy

class Node:
    def __init__(self, state: 'GameState', parent: Optional['Node'] = None, action: Optional[Action] = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: List[Node] = []
        
        # Statistics for MCTS
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = state.get_legal_actions()
        
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state"""
        return self.state.is_terminal()
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried from this node"""
        return len(self.untried_actions) == 0
    
    def get_untried_action(self) -> Optional[Action]:
        """Get a random untried action"""
        if self.untried_actions:
            return random.choice(self.untried_actions)
        return None
    
    def add_child(self, action: Action, state: 'GameState') -> 'Node':
        """Add a child node for the given action"""
        child = Node(state, parent=self, action=action)
        self.children.append(child)
        self.untried_actions.remove(action)
        return child
    
    def update(self, reward: float):
        """Update node statistics after a simulation"""
        self.visits += 1
        self.total_reward += reward
    
    def get_ucb_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """Calculate UCB1 value for node selection"""
        if self.visits == 0:
            return float('inf')
        
        # UCB1 formula: win_rate + exploration_constant * sqrt(ln(parent_visits) / visits)
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def select_best_child(self, exploration_constant: float = math.sqrt(2)) -> 'Node':
        """Select child with highest UCB value"""
        return max(self.children, key=lambda child: child.get_ucb_value(exploration_constant))
    
    def get_most_visited_child(self) -> Optional['Node']:
        """Get the child with the most visits (for final move selection)"""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.visits)


class MCTS:
    def __init__(self, exploration_constant: float = math.sqrt(2), 
                 simulation_depth: int = 50,
                 use_heuristic: bool = True):
        """
        Initialize MCTS algorithm
        
        Args:
            exploration_constant: Balance between exploration and exploitation
            simulation_depth: Maximum depth for random simulation
            use_heuristic: Whether to use heuristic evaluation in simulations
        """
        self.exploration_constant = exploration_constant
        self.simulation_depth = simulation_depth
        self.use_heuristic = use_heuristic
    
    def search(self, root_state: 'GameState', iterations: int = 1000, 
               time_limit: Optional[float] = None) -> Optional[Action]:
        """
        Perform MCTS search from the given state
        
        Args:
            root_state: Current game state
            iterations: Number of MCTS iterations
            time_limit: Optional time limit in seconds
            
        Returns:
            Best action to take from root state
        """
        root = Node(root_state)
        
        # Keep track of time if time limit is set
        start_time = time.time() if time_limit else None
        
        for i in range(iterations):
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            # 1. Selection: traverse from root to leaf using UCB
            node = self._select(root)
            
            # 2. Expansion: add a new child node
            if not node.is_terminal() and node.is_fully_expanded():
                # If fully expanded but not terminal, select best child
                node = node.select_best_child(self.exploration_constant)
            elif not node.is_terminal():
                # Expand by trying an untried action
                node = self._expand(node)
            
            # 3. Simulation: play out randomly from the new node
            reward = self._simulate(node.state)
            
            # 4. Backpropagation: update statistics back to root
            self._backpropagate(node, reward)
        
        # Return the action of the most visited child
        best_child = root.get_most_visited_child()
        if best_child:
            return best_child.action
        return None
    
    def _select(self, node: Node) -> Node:
        """
        Selection phase: traverse tree using UCB until leaf is found
        """
        current = node
        
        while not current.is_terminal() and current.is_fully_expanded():
            current = current.select_best_child(self.exploration_constant)
        
        return current
    
    def _expand(self, node: Node) -> Node:
        """
        Expansion phase: add a new child to the tree
        """
        action = node.get_untried_action()
        if action is None:
            return node
        
        # Apply action to get new state
        new_state = node.state.make_move(action)
        
        # Create and return new child node
        return node.add_child(action, new_state)
    
    def _simulate(self, state: 'GameState') -> float:
        """
        Simulation phase: play out the game randomly or using heuristics
        
        Returns:
            Reward value from the perspective of the initial state's player
        """
        current_state = copy.deepcopy(state)
        initial_player = state.turn_color
        depth = 0
        
        # Play out until terminal or depth limit
        while not current_state.is_terminal() and depth < self.simulation_depth:
            if self.use_heuristic:
                # Use heuristic to guide simulation
                action = self._get_heuristic_action(current_state)
            else:
                # Random action selection
                actions = current_state.get_legal_actions()
                if not actions:
                    break
                action = random.choice(actions)
            
            current_state = current_state.make_move(action)
            depth += 1
        
        # Evaluate final state
        return self._evaluate_state(current_state, initial_player)
    
    def _get_heuristic_action(self, state: 'GameState') -> Action:
        """
        Get action using heuristic (prefer moves that advance toward goal)
        """
        actions = state.get_legal_actions()
        if not actions:
            return None
        
        # Evaluate each action and choose the best
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            score = state.evaluate_action(action)
            if score > best_score:
                best_score = score
                best_action = action
        
        # Add some randomness to avoid being too predictable
        if random.random() < 0.1:  # 10% chance of random action
            return random.choice(actions)
        
        return best_action
    
    def _evaluate_state(self, state: 'GameState', perspective_player: PlayerColor) -> float:
        """
        Evaluate terminal state or use heuristic evaluation
        
        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw, or heuristic value
        """
        # Check for terminal state
        if state.is_terminal():
            winner = state.get_winner()
            if winner == perspective_player:
                return 1.0
            elif winner is None:
                return 0.0  # Draw
            else:
                return -1.0  # Loss
        
        # Use heuristic evaluation for non-terminal states
        if self.use_heuristic:
            # Get current state evaluation
            current_player_score = self._heuristic_score(state, perspective_player)
            opponent_score = self._heuristic_score(state, 
                PlayerColor.BLUE if perspective_player == PlayerColor.RED else PlayerColor.RED)
            
            # Normalize to [-1, 1] range
            total = current_player_score + opponent_score
            if total == 0:
                return 0.0
            
            return (current_player_score - opponent_score) / total
        
        # Default draw for non-terminal states without heuristic
        return 0.0
    
    def _heuristic_score(self, state: 'GameState', player: PlayerColor) -> float:
        """
        Calculate heuristic score for a player in given state
        """
        score = 0.0
        frogs = state.red_frogs if player == PlayerColor.RED else state.blue_frogs
        
        # Score based on frog positions
        for frog in frogs:
            if player == PlayerColor.RED:
                # Red wants to maximize row (get to row 7)
                score += frog.r
                if frog.r == 7:
                    score += 10  # Bonus for reaching goal
            else:
                # Blue wants to minimize row (get to row 0)
                score += (7 - frog.r)
                if frog.r == 0:
                    score += 10  # Bonus for reaching goal
        
        # Add small bonus for having more lily pads
        score += len(state.lily_pads) * 0.1
        
        return score
    
    def _backpropagate(self, node: Node, reward: float):
        """
        Backpropagation phase: update statistics from leaf to root
        """
        current = node
        
        while current is not None:
            # Update from the perspective of the player who made the move
            if current.parent is not None:
                # Reward is from perspective of player who just moved
                parent_player = current.parent.state.turn_color
                current_reward = reward if parent_player == node.state.turn_color else -reward
            else:
                current_reward = reward
            
            current.update(current_reward)
            current = current.parent
    
    def get_action_probabilities(self, root_state: 'GameState', 
                                iterations: int = 1000) -> Dict[Action, float]:
        """
        Get probability distribution over actions based on visit counts
        
        Returns:
            Dictionary mapping actions to their selection probabilities
        """
        root = Node(root_state)
        
        # Run MCTS
        for _ in range(iterations):
            node = self._select(root)
            
            if not node.is_terminal():
                node = self._expand(node)
            
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        # Calculate probabilities based on visit counts
        total_visits = sum(child.visits for child in root.children)
        probabilities = {}
        
        for child in root.children:
            probabilities[child.action] = child.visits / total_visits
        
        return probabilities

# Additional utility functions
import time

class MCTSWithMemory(MCTS):
    """
    MCTS with transposition table to reuse previously computed states
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transposition_table: Dict[str, Node] = {}
    
    def _get_state_key(self, state: 'GameState') -> str:
        """Generate a unique key for the state"""
        # This is a simple implementation - you might want to use a better hash
        key_parts = []
        
        # Add frog positions
        for frog in sorted(state.red_frogs, key=lambda f: (f.r, f.c)):
            key_parts.append(f"R{frog.r}{frog.c}")
        for frog in sorted(state.blue_frogs, key=lambda f: (f.r, f.c)):
            key_parts.append(f"B{frog.r}{frog.c}")
        
        # Add turn
        key_parts.append(str(state.turn_color))
        
        return "_".join(key_parts)
    
    def search(self, root_state: 'GameState', iterations: int = 1000, 
               time_limit: Optional[float] = None) -> Optional[Action]:
        """Enhanced search with transposition table"""
        # Check if we've seen this state before
        state_key = self._get_state_key(root_state)
        
        if state_key in self.transposition_table:
            root = self.transposition_table[state_key]
        else:
            root = Node(root_state)
            self.transposition_table[state_key] = root
        
        # Continue with regular MCTS
        return super().search(root_state, iterations, time_limit)
    
    
    
class GameState:
    def __init__(self, board: Board):
        """Initialize game state from the referee board."""
        self.board = board  # Reference to the original board
        self.turn_color = board.turn_color
        
        # Extract positions of all frogs and lily pads
        self.red_frogs = []
        self.blue_frogs = []
        self.lily_pads = set()
        
        self._parse_board()
    
    def _parse_board(self):
        """Parse the board to extract frog and lily pad positions."""
        for r in range(8):
            for c in range(8):
                coord = Coord(r, c)
                cell_state = self.board[coord].state
                
                if cell_state == PlayerColor.RED:
                    self.red_frogs.append(coord)
                elif cell_state == PlayerColor.BLUE:
                    self.blue_frogs.append(coord)
                elif cell_state == "LilyPad":
                    self.lily_pads.add(coord)
    
    def valid_coord(self, r: int, c: int) -> bool:
        """Check if coordinates are within board bounds."""
        return 0 <= r < 8 and 0 <= c < 8
    
    def get_legal_actions(self) -> List[Action]:
        """Get all legal actions for the current player."""
        if self.turn_color == PlayerColor.RED:
            frogs = self.red_frogs
            directions = [
                Direction.Down, Direction.DownLeft, Direction.DownRight,
                Direction.Left, Direction.Right
            ]
        else:  # BLUE
            frogs = self.blue_frogs
            directions = [
                Direction.Up, Direction.UpLeft, Direction.UpRight,
                Direction.Left, Direction.Right
            ]
        
        actions = []
        
        # Generate MOVE actions for each frog
        for frog_coord in frogs:
            move_actions = self.get_moves_for_frog(frog_coord, directions)
            actions.extend(move_actions)
        
        # Add GROW action if there are empty cells adjacent to any frog
        if self.can_grow():
            actions.append(GrowAction())
        
        return actions
    
    def get_moves_for_frog(self, frog_coord: Coord, valid_directions: List[Direction]) -> List[MoveAction]:
        """Get all possible moves for a single frog."""
        moves = []
        visited = set()
        
        # Direct moves and single jumps
        for direction in valid_directions:
            # Direct move
            direct_move = self.try_direct_move(frog_coord, direction)
            if direct_move:
                moves.append(direct_move)
            
            # Single and multiple jumps
            jump_moves = self.find_jumps(frog_coord, [direction], valid_directions, visited.copy())
            moves.extend(jump_moves)
        
        return moves
    
    def try_direct_move(self, coord: Coord, direction: Direction) -> Optional[MoveAction]:
        """Try a direct move in the given direction."""
        next_r = coord.r + direction.r
        next_c = coord.c + direction.c
        
        if not self.valid_coord(next_r, next_c):
            return None
        
        next_coord = Coord(next_r, next_c)
        
        # Check if target is an empty lily pad
        if (next_coord in self.lily_pads and 
            next_coord not in self.red_frogs and 
            next_coord not in self.blue_frogs):
            return MoveAction(coord, (direction,))
        
        return None
    
    def find_jumps(self, start_coord: Coord, path: List[Direction], 
                   valid_directions: List[Direction], visited: Set[Coord]) -> List[MoveAction]:
        """Find all possible jump sequences from current position."""
        jumps = []
        current_coord = start_coord
        
        # Calculate current position after following the path
        for direction in path:
            jump_pos = self.get_jump_landing(current_coord, direction)
            if jump_pos is None:
                return jumps
            current_coord = jump_pos
        
        # If we've made at least one jump, this is a valid move
        if len(path) > 0 and current_coord != start_coord:
            jumps.append(MoveAction(start_coord, tuple(path)))
        
        visited.add(current_coord)
        
        # Try to continue jumping
        for direction in valid_directions:
            if current_coord in visited:
                continue
                
            landing = self.get_jump_landing(current_coord, direction)
            if landing and landing not in visited:
                new_path = path + [direction]
                extended_jumps = self.find_jumps(start_coord, new_path, valid_directions, visited.copy())
                jumps.extend(extended_jumps)
        
        return jumps
    
    def get_jump_landing(self, coord: Coord, direction: Direction) -> Optional[Coord]:
        """Get the landing position for a jump, or None if invalid."""
        # Check intermediate position (must have a frog)
        mid_r = coord.r + direction.r
        mid_c = coord.c + direction.c
        
        if not self.valid_coord(mid_r, mid_c):
            return None
        
        mid_coord = Coord(mid_r, mid_c)
        
        # Must jump over a frog
        if mid_coord not in self.red_frogs and mid_coord not in self.blue_frogs:
            return None
        
        # Check landing position
        land_r = mid_r + direction.r
        land_c = mid_c + direction.c
        
        if not self.valid_coord(land_r, land_c):
            return None
        
        land_coord = Coord(land_r, land_c)
        
        # Landing must be on an empty lily pad
        if (land_coord in self.lily_pads and 
            land_coord not in self.red_frogs and 
            land_coord not in self.blue_frogs):
            return land_coord
        
        return None
    
    def can_grow(self) -> bool:
        """Check if GROW action is possible."""
        frogs = self.red_frogs if self.turn_color == PlayerColor.RED else self.blue_frogs
        
        for frog in frogs:
            # Check all 8 adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    r, c = frog.r + dr, frog.c + dc
                    if self.valid_coord(r, c):
                        coord = Coord(r, c)
                        # Can grow if cell is empty (no lily pad and no frog)
                        if (coord not in self.lily_pads and 
                            coord not in self.red_frogs and 
                            coord not in self.blue_frogs):
                            return True
        return False
    
    def evaluate_action(self, action: Action) -> float:
        """Evaluate the quality of an action."""
        if isinstance(action, MoveAction):
            return self.evaluate_move(action)
        elif isinstance(action, GrowAction):
            return self.evaluate_grow()
        return 0
    
    def evaluate_move(self, action: MoveAction) -> float:
        """Evaluate a move action based on vertical progress."""
        score = 0
        start_coord = action.coord
        end_coord = self.calculate_end_position(action)
        
        if self.turn_color == PlayerColor.RED:
            # Red wants to maximize row (move down)
            progress = end_coord.r - start_coord.r
            # Bonus for reaching goal row
            if end_coord.r == 7:
                progress += 10
        else:  # BLUE
            # Blue wants to minimize row (move up)
            progress = start_coord.r - end_coord.r
            # Bonus for reaching goal row
            if end_coord.r == 0:
                progress += 10
        
        score += progress * 2
        
        # Penalty for lateral movement without vertical progress
        if progress == 0:
            score -= 0.5
        
        # Bonus for jump moves (more strategic)
        if len(action.directions) > 1:
            score += len(action.directions) * 0.5
        
        return score
    
    def evaluate_grow(self) -> float:
        """Evaluate a grow action."""
        score = 0
        frogs = self.red_frogs if self.turn_color == PlayerColor.RED else self.blue_frogs
        
        # Count potential new lily pads
        new_pads = 0
        for frog in frogs:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    r, c = frog.r + dr, frog.c + dc
                    if self.valid_coord(r, c):
                        coord = Coord(r, c)
                        if (coord not in self.lily_pads and 
                            coord not in self.red_frogs and 
                            coord not in self.blue_frogs):
                            # Check if this creates forward opportunities
                            if self.turn_color == PlayerColor.RED and dr > 0:
                                new_pads += 1.5  # Extra value for forward lily pads
                            elif self.turn_color == PlayerColor.BLUE and dr < 0:
                                new_pads += 1.5
                            else:
                                new_pads += 1
        
        score = new_pads * 0.3
        
        # Penalize grow if we're behind
        avg_position = self.get_average_position()
        if self.turn_color == PlayerColor.RED and avg_position < 3:
            score -= 1
        elif self.turn_color == PlayerColor.BLUE and avg_position > 4:
            score -= 1
        
        return score
    
    def calculate_end_position(self, action: MoveAction) -> Coord:
        """Calculate the final position after a move action."""
        current = action.coord
        
        for direction in action.directions:
            # For direct move
            next_r = current.r + direction.r
            next_c = current.c + direction.c
            
            if self.valid_coord(next_r, next_c):
                next_coord = Coord(next_r, next_c)
                
                # Check if it's a jump
                if (next_coord in self.red_frogs or next_coord in self.blue_frogs):
                    # It's a jump, calculate landing
                    land_r = next_r + direction.r
                    land_c = next_c + direction.c
                    if self.valid_coord(land_r, land_c):
                        current = Coord(land_r, land_c)
                else:
                    # Direct move
                    current = next_coord
        
        return current
    
    def get_average_position(self) -> float:
        """Get average row position of current player's frogs."""
        frogs = self.red_frogs if self.turn_color == PlayerColor.RED else self.blue_frogs
        if not frogs:
            return 0
        return sum(frog.r for frog in frogs) / len(frogs)
    
    def make_move(self, action: Action) -> 'GameState':
        """Apply an action and return a new game state."""
        # Create a deep copy of current state
        new_state = copy.deepcopy(self)
        
        if isinstance(action, MoveAction):
            new_state._apply_move(action)
        elif isinstance(action, GrowAction):
            new_state._apply_grow()
        
        # Switch turns
        new_state.turn_color = (PlayerColor.BLUE if self.turn_color == PlayerColor.RED 
                               else PlayerColor.RED)
        
        return new_state
    
    def _apply_move(self, action: MoveAction):
        """Apply a move action to this state."""
        start = action.coord
        end = self.calculate_end_position(action)
        
        # Remove frog from start position
        if self.turn_color == PlayerColor.RED:
            self.red_frogs.remove(start)
            self.red_frogs.append(end)
        else:
            self.blue_frogs.remove(start)
            self.blue_frogs.append(end)
        
        # Remove lily pad from start position
        self.lily_pads.discard(start)
    
    def _apply_grow(self):
        """Apply a grow action to this state."""
        frogs = self.red_frogs if self.turn_color == PlayerColor.RED else self.blue_frogs
        
        new_pads = set()
        for frog in frogs:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    r, c = frog.r + dr, frog.c + dc
                    if self.valid_coord(r, c):
                        coord = Coord(r, c)
                        if (coord not in self.lily_pads and 
                            coord not in self.red_frogs and 
                            coord not in self.blue_frogs):
                            new_pads.add(coord)
        
        self.lily_pads.update(new_pads)
    
    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        # Check if all red frogs reached row 7
        if all(frog.r == 7 for frog in self.red_frogs):
            return True
        
        # Check if all blue frogs reached row 0
        if all(frog.r == 0 for frog in self.blue_frogs):
            return True
        
        # Could also check for turn limit here if needed
        return False
    
    def get_winner(self) -> Optional[PlayerColor]:
        """Get the winner if the game has ended."""
        if all(frog.r == 7 for frog in self.red_frogs):
            return PlayerColor.RED
        if all(frog.r == 0 for frog in self.blue_frogs):
            return PlayerColor.BLUE
        return None
        
if __name__ == "__main__":
    pass