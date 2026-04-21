# =============================================================================
# GridWorld Environment (no gym/gymnasium dependency)
# Replicates GridWorld_Fox2016 exactly
# =============================================================================
import numpy as np
import pandas as pd

class GridWorldEnv:
    """
    8x8 GridWorld environment from Fox (2016).
    Terminal state: [4, 4] (index 36).
    Blocked (invalid) states: inv_state list below.

    Actions:
        0 - North       (row-1)
        1 - North-East  (row-1, col+1)
        2 - East        (col+1)
        3 - South-East  (row+1, col+1)
        4 - South       (row+1)
        5 - South-West  (row+1, col-1)
        6 - West        (col-1)
        7 - North-West  (row-1, col-1)
        8 - No change
    """

    def __init__(self):
        print('GridWorldEnv (Fox2016) Loaded...')
        self.state = None

        # Blocked grid cells
        self.inv_state = [
            [1,1], [1,4],
            [2,1], [2,4],
            [3,1], [3,4], [3,5], [3,6],
            [4,1], [4,2], [4,6],
            [5,2], [5,6],
            [6,2], [6,6]
        ]

        # Map: flat index -> (row, col)
        self.states_ind = {}
        k = 0
        for i in range(8):
            for j in range(8):
                self.states_ind[k] = (i, j)
                k += 1

        self.n_invalid = len(self.inv_state)

    # ------------------------------------------------------------------
    def _is_blocked(self, row, col):
        return [row, col] in self.inv_state

    def _is_out(self, row, col):
        return row < 0 or row > 7 or col < 0 or col > 7

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Returns: (obs, cost, done, info)
            obs  : np.array([row, col])
            cost : float
            done : bool
            info : dict (empty)
        """
        row, col = self.state
        row_old, col_old = row, col
        done = False
        info = {}
        cost = 0
        sigma = 0.2

        # Already at terminal state
        if [row, col] == [4, 4]:
            return np.array([row, col]), cost, True, info

        # --- Apply chosen action ---
        deltas = {
            0: (-1,  0), 1: (-1,  1), 2: ( 0,  1), 3: ( 1,  1),
            4: ( 1,  0), 5: ( 1, -1), 6: ( 0, -1), 7: (-1, -1),
            8: ( 0,  0)
        }
        dr, dc = deltas[action]
        row += dr
        col += dc
        cost += 1
        cost += np.random.normal(0, sigma)
        # Check if terminal after intended move
        if [row, col] == [4, 4]:
            self.state = np.array([row, col])
            return self.state, cost, True, info

        # Revert if out-of-bounds or blocked
        if self._is_out(row, col) or self._is_blocked(row, col):
            row, col = row_old, col_old
            cost += 1000
            cost += np.random.normal(0, sigma)


        row_old, col_old = row, col  # save post-action position

        # --- Stochastic drift ---
        r = np.random.uniform(0.0, 1.0)
        if   r <= 0.050:               row -= 1
        elif r <= 0.075:               row -= 1; col += 1
        elif r <= 0.125:               col += 1
        elif r <= 0.150:               row += 1; col += 1
        elif r <= 0.200:               row += 1
        elif r <= 0.225:               row += 1; col -= 1
        elif r <= 0.275:               col -= 1
        elif r <= 0.300:               row -= 1; col -= 1
        # else r > 0.3 : no drift

        # Drift cost only applies when drift actually moves agent
        if (row, col) != (row_old, col_old):
            cost += 1
            cost += np.random.normal(0, sigma)


        # Check terminal after drift
        if [row, col] == [4, 4]:
            self.state = np.array([row, col])
            return self.state, cost, True, info

        # Revert drift if out-of-bounds or blocked (undo the +1 cost)
        if self._is_out(row, col) or self._is_blocked(row, col):
            row, col = row_old, col_old
            cost -= 1
            cost -= np.random.normal(0, sigma)

        self.state = np.array([row, col])
        return self.state, cost, done, info

    # ------------------------------------------------------------------
    def reset(self):
        """
        Randomly pick a valid (non-blocked, non-terminal) start state.
        Returns: np.array([row, col])
        """
        while True:
            idx = np.random.randint(64)
            row, col = self.states_ind[idx]
            if [row, col] not in self.inv_state and [row, col] != [4, 4]:
                break
        self.state = np.array([row, col])
        return self.state

    # ------------------------------------------------------------------
    def reset_previous(self, st):
        """Reset to a specific (row, col) state."""
        self.state = np.array(st)
        return self.state
