from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import time
import math

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player
        """
        super().__init__(piece_type, name)
        self._max_depth = 2  # Profondeur maximale de recherche
        self._start_time = 0.0
        self._time_limit = 900.0  # 15 minutes en secondes

    def compute_action(self, current_state: GameState, remaining_time: float = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.
            remaining_time (float): time budget remaining for the agent (in seconds)

        Returns:
            Action: The best action as determined by minimax.
        """

        try:
            rem = float(remaining_time)
        except Exception:
            rem = self._time_limit
        self._time_limit = rem
        self._start_time = time.time()

        possible_actions = list(current_state.get_possible_light_actions())

        if not possible_actions:
            return None

        if len(possible_actions) == 1:
            return possible_actions[0]

        # 1) winning move -> Le jouer pour gagner immédiatement
        for action in possible_actions:
            try:
                next_state = current_state.apply_action(action)
            except Exception:
                continue
            if getattr(next_state, "is_done", lambda: False)():
                return action

        best_action = None
        best_value = -math.inf
        best_unsafe_action = None
        best_unsafe_value = -math.inf

        alpha = -math.inf
        beta = math.inf

        # 2) évalue les positions potentiel mais penalise ceux qui font gagner l'adversaire immédiattement
        for action in possible_actions:
            try:
                next_state = current_state.apply_action(action)
            except Exception:
                continue

            # detect if opponent has an immediate winning response
            opponent_moves = list(next_state.get_possible_light_actions())
            gives_opponent_win = False
            for op in opponent_moves:
                try:
                    resp = next_state.apply_action(op)
                except Exception:
                    continue
                if getattr(resp, "is_done", lambda: False)():
                    gives_opponent_win = True
                    break

            # compute minimax value
            value = self._minimax(next_state, 1, False, alpha, beta)

            if gives_opponent_win:
                # cherche le meilleur des move unsafe et pénalise
                if value > best_unsafe_value:
                    best_unsafe_value = value
                    best_unsafe_action = action
            else:
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)

            if time.time() - self._start_time > self._time_limit - 0.1:
                break

        # préfaire le move le plus safe; fallback to best unsafe if none safe
        return best_action if best_action is not None else best_unsafe_action

    

    def _minimax(self, state: GameState, depth: int, maximizing_player: bool, alpha: float, beta: float) -> float:
        # Condition d'arrêt
        if (depth >= self._max_depth or
            state.is_done() or
            time.time() - self._start_time > self._time_limit - 0.1):
            return self._evaluate_state(state)

        if maximizing_player:
            max_eval = -math.inf
            for action in state.get_possible_light_actions():
                next_state = state.apply_action(action)
                eval_score = self._minimax(next_state, depth + 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval

        else:
            min_eval = math.inf
            for action in state.get_possible_light_actions():
                next_state = state.apply_action(action)
                eval_score = self._minimax(next_state, depth + 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    
    def _evaluate_state(self, state: GameStateHex) -> float:
        if state.is_done():
            try:
                scores = state.get_scores()
                my_score = scores[self.get_id()]
                opponent_score = scores[1 - self.get_id()]
            except Exception:
                return 0.0

            if my_score > opponent_score:
                return 1000.0
            else:
                return -1000.0

        rep = state.get_rep()

        board_array = None
        get_grid = getattr(rep, "get_grid", None)
        if callable(get_grid):
            try:
                grid = get_grid()
                if isinstance(grid, list) and grid:
                    size = len(grid)
                    board_array = [[0 for _ in range(size)] for _ in range(size)]
                    for i in range(size):
                        for j in range(size):
                            cell = grid[i][j]
                            if isinstance(cell, tuple) and len(cell) >= 2:
                                board_array[i][j] = cell[1]
                            elif cell not in (None, " ", 0):
                                board_array[i][j] = cell
            except Exception:
                board_array = None

        # Heuristique
        score = 0.0

        for i in range(size):
            for j in range(size):
                piece = board_array[i][j]
                if piece == self.piece_type:
                    score += self._evaluate_position(i, j, size, board_array, True)
                elif piece != 0:
                    score -= self._evaluate_position(i, j, size, board_array, False)

        return score

    # Logique positionnelle
    def _evaluate_position(self, i: int, j: int, size: int, board_array: list, is_my_piece: bool) -> float:
        """
        Favorise positions qui connectent/bridgent des pions existants plutôt
        que remplir systématiquement la ligne d'arrivée.
        - réduit le poids 'forward' pur
        - ajoute un bonus fort si la case est entre deux de nos pions (bridge)
        - conserve un petit bonus pour connexions locales
        """
        base = 0.1

        # voisins directs (connectivité locale)
        neighbors = len(self._get_neighbors(i, j, size, board_array, is_my_piece))
        local_bonus = neighbors * 0.25

        # vecteurs forward/back selon la couleur
        if self.piece_type == "R":
            f_dir = (-1, 0)   # monter
            b_dir = (1, 0)    # descendre
        else:
            f_dir = (0, -1)   # aller à gauche
            b_dir = (0, 1)    # aller à droite

        # scan avant/arrière (détecte pions proches)
        def scan_dir(di, dj, max_steps=3):
            for k in range(1, max_steps + 1):
                ni, nj = i + di * k, j + dj * k
                if not (0 <= ni < size and 0 <= nj < size):
                    break
                if board_array[ni][nj] == (self.piece_type if is_my_piece else ("R" if self.piece_type == "B" else "B")):
                    return k
            return None

        f_dist = scan_dir(f_dir[0], f_dir[1])
        b_dist = scan_dir(b_dir[0], b_dir[1])

        # bridge / extension
        bridge_bonus = 0.0
        if is_my_piece:
            if f_dist is not None and b_dist is not None:
                bridge_bonus += 2.5 * (1.0 / f_dist + 1.0 / b_dist)
            else:
                if f_dist is not None:
                    bridge_bonus += 0.9 * (1.0 / f_dist) * (1.0 + 0.5 * neighbors)
                if b_dist is not None:
                    bridge_bonus += 0.6 * (1.0 / b_dist) * (1.0 + 0.3 * neighbors)
        else:
            if f_dist is not None and b_dist is not None:
                bridge_bonus -= 1.5 * (1.0 / f_dist + 1.0 / b_dist)
            else:
                if f_dist is not None:
                    bridge_bonus -= 0.6 * (1.0 / f_dist)
                if b_dist is not None:
                    bridge_bonus -= 0.4 * (1.0 / b_dist)

        # détecter si l'adversaire bloque en avant (1-2 cases)
        opponent = ("R" if self.piece_type == "B" else "B")
        forward_block = False
        for k in (1, 2):
            ni, nj = i + f_dir[0] * k, j + f_dir[1] * k
            if 0 <= ni < size and 0 <= nj < size and board_array[ni][nj] == opponent:
                forward_block = True
                break

        # compter voisins latéraux (directions autres que forward/back)
        lateral_dirs = [d for d in [(-1, 1), (0, -1), (0, 1), (1, -1)]]
        lateral_neighbors = 0
        for di, dj in lateral_dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < size and 0 <= nj < size and board_array[ni][nj] == (self.piece_type if is_my_piece else opponent):
                lateral_neighbors += 1

        # si blocage en avant, favoriser positions latérales (contournement)
        contour_bonus = 0.0
        if forward_block and is_my_piece:
            contour_bonus += lateral_neighbors * 1.2 + 0.8  # favorise les positions qui construisent autour du bloc
        elif forward_block and not is_my_piece:
            contour_bonus -= lateral_neighbors * 0.8  # pénaliser si l'adversaire contournne vers nous

      
        return base + local_bonus + bridge_bonus + contour_bonus 

    def _get_neighbors(self, i: int, j: int, size: int, board_array: list, is_my_piece: bool) -> list:
        neighbors = []
        target_piece = self.piece_type if is_my_piece else ("R" if self.piece_type == "B" else "B")

        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < size and 0 <= nj < size:
                if board_array[ni][nj] == target_piece:
                    neighbors.append((ni, nj))

        return neighbors
