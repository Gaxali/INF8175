from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import random
import time
import math
from collections import deque
import heapq

class MyPlayer(PlayerHex):
    """
    Player class for Hex game
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._max_depth = 2
        self._start_time = 0
        self._time_limit = 900.0

        self._dirs = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
        self._bridge_patterns = {
            (-1, -1): [(-1, 0), (0, -1)],
            (-2, 1): [(-1, 0), (-1, 1)],
            (-1, 2): [(-1, 1), (0, 1)],
            (1, 1): [(1, 0), (0, 1)],
            (2, -1): [(1, -1), (1, 0)],
            (1, -2): [(0, -1), (1, -1)],
        }

    def time_is_up(self):
        return (time.time() - self._start_time) >= self._time_limit
    
    def extract_board(self, rep):
        # Essaye d'obtenir la grille depuis rep.get_grid()
        get_grid = getattr(rep, "get_grid", None)
        if callable(get_grid):
            try:
                g = get_grid()
                if isinstance(g, list) and g:
                    size = len(g)
                    board = [[0 for _ in range(size)] for _ in range(size)]
                    for i in range(size):
                        for j in range(size):
                            cell = g[i][j]
                            if isinstance(cell, tuple) and len(cell) >= 2:
                                board[i][j] = cell[1]
                            elif cell not in (None, " ", 0):
                                board[i][j] = cell
                    return board, size
            except:
                pass

        # Sinon fallback via rep.env
        env = getattr(rep, "env", None) or getattr(rep, "_env", None)
        if env:
            max_i = max(k[0] for k in env.keys())
            max_j = max(k[1] for k in env.keys())
            size = max(max_i, max_j) + 1
            board = [[0 for _ in range(size)] for _ in range(size)]
            for (i, j), piece in env.items():
                if piece is None:
                    continue
                val = None
                for attr in ("get_type", "piece_type", "type", "value"):
                    a = getattr(piece, attr, None)
                    try:
                        val = a() if callable(a) else a
                    except:
                        val = None
                    if val is not None:
                        break
                if val is None:
                    try:
                        val = str(piece)[0]
                    except:
                        val = 0
                board[i][j] = val
            return board, size

        raise RuntimeError("Impossible d'extraire le board Hex")

    def evaluate_state(self, state):
        rep = state.get_rep()
        board, size = self.extract_board(rep)

        my = self.piece_type
        opp = "R" if my == "B" else "B"

        return - self.hex_distance(board, size, my) + self.hex_distance(board, size, opp)

    def hex_distance(self, board, size, player):
        opponent = "R" if player == "B" else "B"

        def heuristic(x, y):
            # Distance jusqu'au bord cible
            if player == "R":
                return size - 1 - x
            else:  # player == "B"
                return size - 1 - y

        dist = [[float("inf")] * size for _ in range(size)]
        pq = []

        # Initialisation depuis le bord du joueur
        if player == "R":
            for y in range(size):
                if board[0][y] != opponent:
                    cost = 0 if board[0][y] == player else 1
                    dist[0][y] = cost
                    heapq.heappush(pq, (cost + heuristic(0, y), cost, (0, y)))
        else:  # player == "B"
            for x in range(size):
                if board[x][0] != opponent:
                    cost = 0 if board[x][0] == player else 1
                    dist[x][0] = cost
                    heapq.heappush(pq, (cost + heuristic(x, 0), cost, (x, 0)))

        while pq:
            f, d, (x, y) = heapq.heappop(pq)
            if d != dist[x][y]:
                continue

            for dx, dy in self._dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if board[nx][ny] == opponent:
                        continue
                    cost = 0 if board[nx][ny] == player else 1
                    nd = d + cost
                    if nd < dist[nx][ny]:
                        dist[nx][ny] = nd
                        heapq.heappush(pq, (nd + heuristic(nx, ny), nd, (nx, ny)))

            # Bridges
            for (dx, dy), bridge_intermediate in self._bridge_patterns.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if board[x][y] == player and board[nx][ny] == player:
                        x1, y1 = x + bridge_intermediate[0][0], y + bridge_intermediate[0][1]
                        x2, y2 = x + bridge_intermediate[1][0], y + bridge_intermediate[1][1]
                        if board[x1][y1] == 0 and board[x2][y2] == 0:
                            nd = d + 0.8
                            if nd < dist[nx][ny]:
                                dist[nx][ny] = nd
                                heapq.heappush(pq, (nd + heuristic(nx, ny), nd, (nx, ny)))

        # Distance au bord opposé
        best = float("inf")
        if player == "R":
            for y in range(size):
                best = min(best, dist[size-1][y])
        else:  # player == "B"
            for x in range(size):
                best = min(best, dist[x][size-1])

        return best
    
    def minimax(self, state, depth, alpha, beta, maximizing):
        if self.time_is_up():
            raise TimeoutError
        
        if depth == 0 or state.is_done():
            return self.evaluate_state(state), None

        actions = state.get_possible_light_actions()
        if not actions:
            return self.evaluate_state(state), None

        if maximizing:
            best_value = -float("inf")
            best_action = None

            for action in actions:
                next_state = state.apply_action(action)
                value, _ = self.minimax(next_state, depth - 1, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_action = action

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            return best_value, best_action

        else:
            best_value = float("inf")
            best_action = None

            for action in actions:
                next_state = state.apply_action(action)
                value, _ = self.minimax(next_state, depth - 1, alpha, beta, True)

                if value < best_value:
                    best_value = value
                    best_action = action

                beta = min(beta, best_value)
                if beta <= alpha:
                    break

            return best_value, best_action
    
    def compute_action(self, current_state: GameState, remaining_time: float = 1e9, **kwargs) -> Action:
        self._start_time = time.time()

        actions = current_state.get_possible_light_actions()
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]
        
        best_action = None

        # appel du minimax
        try:
            _, best_action = self.minimax(current_state, depth=self._max_depth, alpha=-float("inf"), beta=float("inf"), maximizing=True)
        except TimeoutError:
            pass
        return best_action if best_action  else random.choice(list(actions))
