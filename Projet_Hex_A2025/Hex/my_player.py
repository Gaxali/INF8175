from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import random
import time
import math
from collections import deque

class MyPlayer(PlayerHex):
    """
    Player class for Hex game
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self._max_depth = 2
        self._start_time = 0.0
        self._time_limit = 900.0
        self._center_played = False        # ouverture centrale tentée
        self._sides_played = set()         # côtés déjà utilisés (top/bottom ou left/right)
        self._last_side = None             # dernier côté choisi (pour alternance)
        self._alternate_fill_threshold = 4 # seuil pour considérer un côté "suffisamment rempli"
        self._center_needed = 3            # nb de pions dans corridor pour quitter phase centre
        self._focus_column = None
        self._focus_row = None

    def compute_action(self, current_state: GameState, remaining_time: float = 1e9, **kwargs) -> Action:
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

        # --- construire board simple ---
        rep = current_state.get_rep()
        board = None
        size = 0

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
            except Exception:
                board = None

        if board is None:
            env = None
            get_env = getattr(rep, "get_env", None)
            if callable(get_env):
                try:
                    env = get_env()
                except Exception:
                    env = None
            else:
                env = getattr(rep, "env", None) or getattr(rep, "_env", None)

            if isinstance(env, dict) and env:
                max_i = max(k[0] for k in env.keys())
                max_j = max(k[1] for k in env.keys())
                size = max(max_i, max_j) + 1
                board = [[0 for _ in range(size)] for _ in range(size)]
                for (i, j), piece in env.items():
                    if piece is None:
                        continue
                    ptype = None
                    for attr in ("get_type", "piece_type", "type", "value"):
                        a = getattr(piece, attr, None)
                        try:
                            ptype = a() if callable(a) else a
                        except Exception:
                            ptype = None
                        if ptype is not None:
                            break
                    if ptype is None:
                        try:
                            s = str(piece)
                            ptype = s[0] if s else 0
                        except Exception:
                            ptype = 0
                    board[i][j] = ptype

        if board is None or size == 0:
            return random.choice(possible_actions)

        my = self.piece_type
        opp = "R" if my == "B" else "B"
        occupied = sum(1 for row in board for c in row if c != 0)

        # helper : vérifie si une action place sur coord (renvoie True si oui)
        def action_places_on(action, coord):
            try:
                ns = current_state.apply_action(action)
            except Exception:
                return False
            rep2 = ns.get_rep()
            get_grid2 = getattr(rep2, "get_grid", None)
            if callable(get_grid2):
                try:
                    g2 = get_grid2()
                    if isinstance(g2, list) and 0 <= coord[0] < len(g2) and 0 <= coord[1] < len(g2):
                        cell = g2[coord[0]][coord[1]]
                        val = cell[1] if isinstance(cell, tuple) and len(cell) >= 2 else cell
                        return val == my
                except Exception:
                    pass
            get_env2 = getattr(rep2, "get_env", None)
            env2 = None
            if callable(get_env2):
                try:
                    env2 = get_env2()
                except Exception:
                    env2 = None
            else:
                env2 = getattr(rep2, "env", None) or getattr(rep2, "_env", None)
            if isinstance(env2, dict):
                val = env2.get(coord)
                if val is None:
                    return False
                for attr in ("get_type", "piece_type", "type", "value"):
                    a = getattr(val, attr, None)
                    try:
                        p = a() if callable(a) else a
                    except Exception:
                        p = None
                    if p is not None:
                        return p == my
                try:
                    return str(val).find(my) != -1
                except Exception:
                    return False
            return False

        # --- cellules centrales utilitaire ---
        def get_central_cells(sz, target=12):
            ci = (sz - 1) / 2.0
            cj = (sz - 1) / 2.0
            cells = [(i, j, abs(i - ci) + abs(j - cj)) for i in range(sz) for j in range(sz)]
            cells.sort(key=lambda x: (x[2], x[0], x[1]))
            return [(i, j) for (i, j, _) in cells[:min(target, sz*sz)]]

        center_cells = get_central_cells(size, target=min(12, size*size))
        central_empty = [c for c in center_cells if board[c[0]][c[1]] == 0]

        # --- ouverture : R commence au centre (préférence), B réagit au premier R (si applicable) ---
        if (not self._center_played) and central_empty and occupied <= 2:
            # R premier coup -> choisir une cellule "centrée mais orientée" vers notre objectif
            if my == "R" and occupied == 0:
                mid = size // 2
                # préférence pour avancer rapidement : 4 lignes vers notre bord (bas)
                pref = (min(size - 1, mid + 4), mid)
                if 0 <= pref[0] < size and 0 <= pref[1] < size and board[pref[0]][pref[1]] == 0:
                    for a in possible_actions:
                        if action_places_on(a, pref):
                            self._center_played = True
                            return a
                # fallback : cellule centrale la plus proche de pref
                empties = [c for c in center_cells if board[c[0]][c[1]] == 0]
                if empties:
                    empties.sort(key=lambda c: abs(c[0] - pref[0]) + abs(c[1] - pref[1]))
                    cand = empties[0]
                    for a in possible_actions:
                        if action_places_on(a, cand):
                            self._center_played = True
                            return a
                    if board[cand[0]][cand[1]] != 0:
                        self._center_played = True

            # B reacting to single R move: place on R's column ~4 towards center
            elif my == "B" and occupied == 1:
                opp_cells = [(i, j) for i in range(size) for j in range(size) if board[i][j] == opp]
                if opp_cells:
                    oi, oj = opp_cells[0]
                    ci = (size - 1) / 2.0
                    dir_i = 1 if oi < ci else -1
                    target_i = oi + 4 * dir_i
                    target_i = max(0, min(size - 1, int(round(target_i))))
                    pref = (target_i, oj)
                    if 0 <= pref[0] < size and 0 <= pref[1] < size and board[pref[0]][pref[1]] == 0:
                        for a in possible_actions:
                            if action_places_on(a, pref):
                                self._center_played = True
                                self._focus_column = oj
                                self._focus_row = oi
                                return a
                    # fallback: central cell closest to pref
                    if central_empty:
                        central_empty.sort(key=lambda c: abs(c[0] - pref[0]) + abs(c[1] - pref[1]))
                        cand = central_empty[0]
                        for a in possible_actions:
                            if action_places_on(a, cand):
                                self._center_played = True
                                return a
                        if board[cand[0]][cand[1]] != 0:
                            self._center_played = True

            else:
                # general fallback: place in center as before
                desired_center = random.choice(central_empty)
                for a in possible_actions:
                    if action_places_on(a, desired_center):
                        self._center_played = True
                        return a
                if board[desired_center[0]][desired_center[1]] != 0:
                    self._center_played = True

        # --- 0-1 BFS helpers (nos pions coût 0, vides coût 1, adversaire bloqué) ---
        dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        def bfs_from_starts(starts, is_target):
            INF = 10**9
            dist = {}
            prev = {}
            dq = deque()
            for s in starts:
                i, j = s
                if not (0 <= i < size and 0 <= j < size):
                    continue
                if board[i][j] == opp:
                    continue
                cost = 0 if board[i][j] == my else 1
                if dist.get((i, j), INF) > cost:
                    dist[(i, j)] = cost
                    prev[(i, j)] = None
                    if cost == 0:
                        dq.appendleft((i, j))
                    else:
                        dq.append((i, j))
            reached_targets = []
            best_target_cost = INF
            while dq:
                u = dq.popleft()
                if dist.get(u, INF) > best_target_cost:
                    continue
                if is_target(u):
                    c = dist.get(u, INF)
                    if c < best_target_cost:
                        best_target_cost = c
                        reached_targets = [u]
                    elif c == best_target_cost:
                        reached_targets.append(u)
                    continue
                ui, uj = u
                du = dist.get(u, INF)
                for di, dj in dirs:
                    vi, vj = ui + di, uj + dj
                    if not (0 <= vi < size and 0 <= vj < size):
                        continue
                    if board[vi][vj] == opp:
                        continue
                    w = 0 if board[vi][vj] == my else 1
                    nd = du + w
                    if nd < dist.get((vi, vj), INF):
                        dist[(vi, vj)] = nd
                        prev[(vi, vj)] = (ui, uj)
                        if w == 0:
                            dq.appendleft((vi, vj))
                        else:
                            dq.append((vi, vj))
            return reached_targets, best_target_cost, prev, dist

        # --- préparer deux recherches (sens A et B) ---
        my_cells = [(i, j) for i in range(size) for j in range(size) if board[i][j] == my]

        if my == "R":
            borderA = [(0, j) for j in range(size)]
            borderB = [(size - 1, j) for j in range(size)]
            is_targetA = lambda cell: cell[0] == size - 1
            is_targetB = lambda cell: cell[0] == 0
            sideA_name, sideB_name = "top", "bottom"
        else:
            borderA = [(i, 0) for i in range(size)]
            borderB = [(i, size - 1) for i in range(size)]
            is_targetA = lambda cell: cell[1] == size - 1
            is_targetB = lambda cell: cell[1] == 0
            sideA_name, sideB_name = "left", "right"

        startsA = list(dict.fromkeys(my_cells + borderA))
        startsB = list(dict.fromkeys(my_cells + borderB))

        reachedA, costA, prevA, distA = bfs_from_starts(startsA, is_targetA)
        reachedB, costB, prevB, distB = bfs_from_starts(startsB, is_targetB)

        def first_empty_cells(reached, prev):
            set_cells = set()
            for t in reached:
                cur = t
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev.get(cur)
                path.reverse()
                for (pi, pj) in path:
                    if board[pi][pj] == 0:
                        set_cells.add((pi, pj))
                        break
            return set_cells

        candidatesA = first_empty_cells(reachedA, prevA) if reachedA else set()
        candidatesB = first_empty_cells(reachedB, prevB) if reachedB else set()

        # --- alternance forcée des côtés (sauf si côté "bien rempli") ---
        def side_fill_count(side_name):
            border = borderA if side_name == sideA_name else borderB
            return sum(1 for (i, j) in border if board[i][j] == my)

        threshold = getattr(self, "_alternate_fill_threshold", 4)
        last = getattr(self, "_last_side", None)

        # choisir côté préféré en alternance
        if last is None:
            if sideA_name not in self._sides_played and candidatesA:
                preferred = sideA_name
            elif sideB_name not in self._sides_played and candidatesB:
                preferred = sideB_name
            else:
                preferred = sideA_name if random.random() < 0.5 else sideB_name
        else:
            preferred = sideB_name if last == sideA_name else sideA_name

        def preferred_has_candidates(pref):
            return (pref == sideA_name and bool(candidatesA)) or (pref == sideB_name and bool(candidatesB))

        if not preferred_has_candidates(preferred) or side_fill_count(preferred) >= threshold:
            other = sideB_name if preferred == sideA_name else sideA_name
            if preferred_has_candidates(other) and side_fill_count(other) < threshold:
                preferred = other

        desired = None
        if preferred == sideA_name and candidatesA:
            desired = random.choice(list(candidatesA))
            self._sides_played.add(sideA_name)
            self._last_side = sideA_name
        elif preferred == sideB_name and candidatesB:
            desired = random.choice(list(candidatesB))
            self._sides_played.add(sideB_name)
            self._last_side = sideB_name
        else:
            # fallback : choisir parmi meilleurs coûts A/B
            best_cost = None
            if reachedA and reachedB:
                best_cost = min(costA, costB)
            elif reachedA:
                best_cost = costA
            elif reachedB:
                best_cost = costB
            pool = set()
            if best_cost is not None:
                if reachedA and costA == best_cost:
                    pool |= candidatesA
                if reachedB and costB == best_cost:
                    pool |= candidatesB
            if not pool:
                pool = candidatesA | candidatesB
            if pool:
                desired = random.choice(list(pool))

        if not desired:
            return random.choice(possible_actions)

        # retrouver action correspondant à desired
        def action_plays_on_coord(action, coord):
            try:
                ns = current_state.apply_action(action)
            except Exception:
                return False
            rep2 = ns.get_rep()
            get_grid2 = getattr(rep2, "get_grid", None)
            if callable(get_grid2):
                try:
                    g2 = get_grid2()
                    if isinstance(g2, list) and 0 <= coord[0] < len(g2) and 0 <= coord[1] < len(g2):
                        cell = g2[coord[0]][coord[1]]
                        val = cell[1] if isinstance(cell, tuple) and len(cell) >= 2 else cell
                        return val == my
                except Exception:
                    pass
            get_env2 = getattr(rep2, "get_env", None)
            env2 = None
            if callable(get_env2):
                try:
                    env2 = get_env2()
                except Exception:
                    env2 = None
            else:
                env2 = getattr(rep2, "env", None) or getattr(rep2, "_env", None)
            if isinstance(env2, dict) and env2.get(coord) is not None:
                val = env2.get(coord)
                for attr in ("get_type", "piece_type", "type", "value"):
                    a = getattr(val, attr, None)
                    try:
                        p = a() if callable(a) else a
                    except Exception:
                        p = None
                    if p is not None:
                        return p == my
                try:
                    return str(val).find(my) != -1
                except Exception:
                    return False
            return False

        for a in possible_actions:
            if action_plays_on_coord(a, desired):
                return a

        # brute-force fallback
        for a in possible_actions:
            try:
                ns = current_state.apply_action(a)
            except Exception:
                continue
            rep2 = ns.get_rep()
            get_grid2 = getattr(rep2, "get_grid", None)
            if callable(get_grid2):
                try:
                    g2 = get_grid2()
                    if isinstance(g2, list) and 0 <= desired[0] < len(g2) and 0 <= desired[1] < len(g2):
                        cell = g2[desired[0]][desired[1]]
                        val = cell[1] if isinstance(cell, tuple) and len(cell) >= 2 else cell
                        if val == my:
                            return a
                except Exception:
                    pass
            get_env2 = getattr(rep2, "get_env", None)
            env2 = None
            if callable(get_env2):
                try:
                    env2 = get_env2()
                except Exception:
                    env2 = None
            else:
                env2 = getattr(rep2, "env", None) or getattr(rep2, "_env", None)
            if isinstance(env2, dict) and env2.get(desired) is not None:
                val = env2.get(desired)
                try:
                    for attr in ("get_type", "piece_type", "type", "value"):
                        a = getattr(val, attr, None)
                        p = a() if callable(a) else a
                        if p == my:
                            return a
                except Exception:
                    pass

        return random.choice(possible_actions)
    

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

