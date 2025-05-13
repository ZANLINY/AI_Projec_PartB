# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from typing import Optional

from referee.game.board import Board
from .MCTS import GameState, MCTS, MCTSWithMemory
import time


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        """Initialize the agent with MCTS"""
        self._color = color
        self.game_state = None
        self.turn_count = 0
        
        # Initialize board state tracking (fallback)
        self.init_board_state()
        
        # MCTS configuration - initialize later when we have game state
        self.mcts = None
        
        # Time management
        self.time_remaining = referee.get("time_remaining", 180.0)
        
        print(f"Agent initialized as {color}")
    
    def init_board_state(self):
        """Initialize board state for fallback strategy"""
        self.red_frogs = set()
        self.blue_frogs = set()
        self.lily_pads = set()
        
        # Set initial positions
        for c in range(1, 7):  # columns 1-6
            self.red_frogs.add(Coord(0, c))
            self.lily_pads.add(Coord(0, c))
            self.blue_frogs.add(Coord(7, c))
            self.lily_pads.add(Coord(7, c))
        
        # Corner lily pads
        for r, c in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            self.lily_pads.add(Coord(r, c))
        
        # Rows 1 and 6 lily pads
        for c in range(1, 7):
            self.lily_pads.add(Coord(1, c))
            self.lily_pads.add(Coord(6, c))
    
    def action(self, **referee: dict) -> Action:
        """
        Select action using MCTS
        """
        self.turn_count += 1
        self.time_remaining = referee.get("time_remaining", self.time_remaining)
        
        print(f"\n{self._color} is thinking... (Turn {self.turn_count})")
        print(f"Time remaining: {self.time_remaining:.1f}s")
        
        # Initialize game state if needed
        if self.game_state is None:
            # Create initial game state from board
            # Note: This is simplified - you'll need to get board from referee
            board = Board()  # This should come from referee context
            self.game_state = GameState(board)
        
        # Calculate time allocation for this turn
        time_for_turn = self._calculate_time_allocation()
        
        # Run MCTS with time limit
        start_time = time.time()
        
        # Adaptive iterations based on game phase
        iterations = self._calculate_iterations()
        
        try:
            # Run MCTS search
            best_action = self.mcts.search(
                self.game_state,
                iterations=iterations,
                time_limit=time_for_turn
            )
            
            if best_action is None:
                # Fallback to simple strategy
                best_action = self._get_fallback_action()
            
            # Debug information
            action_probs = self.mcts.get_action_probabilities(
                self.game_state, 
                iterations=min(100, iterations)
            )
            
            print(f"Action probabilities:")
            for action, prob in sorted(action_probs.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {action}: {prob:.3f}")
            
            elapsed = time.time() - start_time
            print(f"MCTS took {elapsed:.3f}s, chose: {best_action}")
            
            return best_action
            
        except Exception as e:
            print(f"Error in MCTS: {e}")
            return self._get_fallback_action()
    
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """Update internal game state"""
        print(f"Update: {color} played {action}")
        
        if self.game_state is not None:
            # Update game state
            self.game_state = self.game_state.make_move(action)
        
        # Update time remaining
        self.time_remaining = referee.get("time_remaining", self.time_remaining)
    
    def _calculate_time_allocation(self) -> float:
        """
        Calculate how much time to use for this turn
        """
        if self.time_remaining is None:
            return 5.0  # Default 5 seconds if no time limit
        
        # Simple time management: use more time in critical positions
        expected_turns_remaining = max(10, 75 - self.turn_count)
        base_time = self.time_remaining / expected_turns_remaining
        
        # Use more time in the middle game
        if 20 < self.turn_count < 60:
            base_time *= 1.5
        
        # Never use more than 10% of remaining time
        max_time = self.time_remaining * 0.1
        
        return min(base_time, max_time, 10.0)  # Cap at 10 seconds
    
    def _calculate_iterations(self) -> int:
        """
        Calculate MCTS iterations based on game phase
        """
        # Early game: fewer iterations (moves are more obvious)
        if self.turn_count < 10:
            return 500
        
        # Mid game: more iterations (critical decisions)
        elif self.turn_count < 50:
            return 1000
        
        # End game: moderate iterations (clearer objectives)
        else:
            return 750
    
    def _get_fallback_action(self) -> Action:
        """
        Simple fallback strategy when MCTS fails
        """
        if self.game_state:
            actions = self.game_state.get_legal_actions()
            if actions:
                # Choose action with best immediate evaluation
                return max(actions, key=lambda a: self.game_state.evaluate_action(a))
        
        # Ultimate fallback
        return GrowAction()

# Advanced version with opening book and endgame tables
class AdvancedMCTSAgent(Agent):
    def __init__(self, color: PlayerColor, **referee: dict):
        super().__init__(color, **referee)
        
        # Opening book (pre-computed good opening moves)
        self.opening_book = self._load_opening_book()
        
        # Endgame patterns
        self.endgame_patterns = self._load_endgame_patterns()
    
    def action(self, **referee: dict) -> Action:
        """Enhanced action selection with special cases"""
        
        # Check opening book for early game
        if self.turn_count < 5:
            book_move = self._check_opening_book()
            if book_move:
                print(f"Using opening book: {book_move}")
                return book_move
        
        # Check endgame patterns
        if self._is_endgame():
            endgame_move = self._check_endgame_patterns()
            if endgame_move:
                print(f"Using endgame pattern: {endgame_move}")
                return endgame_move
        
        # Otherwise use MCTS
        return super().action(**referee)
    
    def _load_opening_book(self) -> dict:
        """Load pre-computed opening moves"""
        # This would typically load from a file
        # For now, return some simple opening principles
        return {
            "RED_FIRST": MoveAction(Coord(0, 3), (Direction.Down,)),
            "BLUE_FIRST": GrowAction(),
        }
    
    def _load_endgame_patterns(self) -> dict:
        """Load endgame patterns"""
        # Simplified version
        return {}
    
    def _check_opening_book(self) -> Optional[Action]:
        """Check if current position is in opening book"""
        if self.turn_count == 0 and self._color == PlayerColor.RED:
            return self.opening_book.get("RED_FIRST")
        return None
    
    def _is_endgame(self) -> bool:
        """Determine if we're in endgame"""
        if not self.game_state:
            return False
        
        # Simple heuristic: if any frog is close to goal
        frogs = (self.game_state.red_frogs if self._color == PlayerColor.RED 
                else self.game_state.blue_frogs)
        
        for frog in frogs:
            if self._color == PlayerColor.RED and frog.r >= 5:
                return True
            elif self._color == PlayerColor.BLUE and frog.r <= 2:
                return True
        
        return False
    
    def _check_endgame_patterns(self) -> Optional[Action]:
        """Check for known endgame patterns"""
        # Implement pattern matching for endgame positions
        return None