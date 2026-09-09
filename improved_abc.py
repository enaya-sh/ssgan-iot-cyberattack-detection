from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BeeState:
    x: np.ndarray
    fitness: float
    trials: int = 0


class ImprovedABC:
    """
    Research-aligned reference implementation of the thesis' improved
    Artificial Bee Colony (ABC) optimizer.

    It includes:
    - random initialization (Equation 6)
    - employed-bee search
    - fitness-proportional onlooker selection (Equation 8)
    - scout replacement after stagnation
    - a mutual-learning move inspired by Equation 9

    The thesis does not specify every implementation detail required for a
    byte-for-byte reconstruction (e.g., exact minimization/maximization
    transformation, clipping/tie rules, and all stopping details).
    Consequently this class should be described as "research-aligned",
    not as the exact original experimental source.
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dimension: int,
        colony_size: int = 20,
        max_iter: int = 50,
        limit: int = 10,
        seed: int = 42,
    ):
        if colony_size < 2:
            raise ValueError("colony_size must be >= 2")
        self.objective = objective
        self.dimension = dimension
        self.colony_size = colony_size
        self.max_iter = max_iter
        self.limit = limit
        self.rng = np.random.default_rng(seed)

    def _random_key(self):
        # Equation 6 with normalized random-key bounds [0,1].
        return self.rng.random(self.dimension)

    def _evaluate(self, x):
        return float(self.objective(np.clip(x, 0.0, 1.0)))

    def _mutual_learning_candidate(self, states, i):
        choices = [j for j in range(len(states)) if j != i]
        k = int(self.rng.choice(choices))
        xi = states[i].x
        xk = states[k].x

        j = int(self.rng.integers(0, self.dimension))
        phi = float(self.rng.uniform(0.0, 1.0))

        v = xi.copy()

        # Higher fitness is treated as better.
        if states[i].fitness < states[k].fitness:
            v[j] = xi[j] + phi * (xk[j] - xi[j])
        else:
            v[j] = xk[j] + phi * (xi[j] - xk[j])

        return np.clip(v, 0.0, 1.0)

    def _greedy_update(self, state, candidate):
        score = self._evaluate(candidate)
        if score > state.fitness:
            return BeeState(candidate, score, 0)
        state.trials += 1
        return state

    def optimize(self):
        states = [
            BeeState(x := self._random_key(), self._evaluate(x))
            for _ in range(self.colony_size)
        ]

        for _ in range(self.max_iter):
            # Employed bees
            for i in range(self.colony_size):
                candidate = self._mutual_learning_candidate(states, i)
                states[i] = self._greedy_update(states[i], candidate)

            # Onlooker bees - Equation 8 style fitness proportional selection.
            raw = np.array([s.fitness for s in states], dtype=float)
            shifted = raw - raw.min() + 1e-12
            probs = shifted / shifted.sum()

            for _ in range(self.colony_size):
                i = int(self.rng.choice(self.colony_size, p=probs))
                candidate = self._mutual_learning_candidate(states, i)
                states[i] = self._greedy_update(states[i], candidate)

            # Scout bees
            for i, state in enumerate(states):
                if state.trials >= self.limit:
                    x = self._random_key()
                    states[i] = BeeState(x, self._evaluate(x), 0)

        return max(states, key=lambda s: s.fitness)
