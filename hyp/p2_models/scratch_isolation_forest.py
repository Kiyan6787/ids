import logging

import numpy as np

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329


def average_path_length(n) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    c = np.zeros_like(n)
    c[n == 2] = 1.0
    big = n > 2
    
    c[big] = 2.0 * (np.log(n[big] - 1.0) + EULER_GAMMA) - 2.0 * (n[big] - 1.0) / n[big]
    return c


class _IsolationTree:


    def __init__(self, rows: np.ndarray, height_limit: int, rng: np.random.RandomState):
        self.feature: list[int] = []      # split feature, -1 for a leaf
        self.threshold: list[float] = []  # split value p
        self.left: list[int] = []         # child node for x[feature] < p
        self.right: list[int] = []        # child node for x[feature] >= p
        self.size: list[int] = []         # training rows that reached this node
        self.depth: list[int] = [] 
        self._grow(rows, 0, height_limit, rng)
        # Freeze to arrays for vectorised scoring
        self.feature = np.array(self.feature)
        self.threshold = np.array(self.threshold)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.size = np.array(self.size)
        self.depth = np.array(self.depth)

    def _grow(self, rows: np.ndarray, depth: int, height_limit: int,
              rng: np.random.RandomState) -> int:
        node = len(self.feature)
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.size.append(len(rows))
        self.depth.append(depth)

        if depth >= height_limit or len(rows) <= 1:
            return node
        lo, hi = rows.min(axis=0), rows.max(axis=0)
        candidates = np.flatnonzero(hi > lo)
        if len(candidates) == 0:
            return node

        q = rng.choice(candidates)
        p = rng.uniform(lo[q], hi[q])
        goes_left = rows[:, q] < p

        self.feature[node] = q
        self.threshold[node] = p
        self.left[node] = self._grow(rows[goes_left], depth + 1, height_limit, rng)
        self.right[node] = self._grow(rows[~goes_left], depth + 1, height_limit, rng)
        return node

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        node = np.zeros(len(X), dtype=np.int64)
        active = self.feature[node] >= 0

        while active.any():
            rows = np.flatnonzero(active)
            n = node[rows]
            goes_left = X[rows, self.feature[n]] < self.threshold[n]
            node[rows] = np.where(goes_left, self.left[n], self.right[n])
            active[rows] = self.feature[node[rows]] >= 0
        return self.depth[node] + average_path_length(self.size[node])


class ScratchIsolationForestDetector(AnomalyDetector):

    model_name = "ScratchIsolationForest"

    def __init__(self, n_estimators: int = 100, max_samples: int = 256, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self.trees: list[_IsolationTree] = []
        self._sample_size = 0

    # API
    def train(self, train_data: np.ndarray) -> None:
        rng = np.random.RandomState(self.seed)
        X = np.asarray(train_data, dtype=np.float64)
        self._sample_size = min(self.max_samples, len(X))

        height_limit = int(np.ceil(np.log2(max(self._sample_size, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            sample = X[rng.choice(len(X), self._sample_size, replace=False)]
            self.trees.append(_IsolationTree(sample, height_limit, rng))
        logger.info("ScratchIF trained: %d trees on %d-row subsamples of %d rows",
                    self.n_estimators, self._sample_size, len(X))

    def score(self, data: np.ndarray) -> np.ndarray:
        X = np.asarray(data, dtype=np.float64)
        total_path = np.zeros(len(X))
        for tree in self.trees:
            total_path += tree.path_lengths(X)
        mean_path = total_path / len(self.trees)
        return 2.0 ** (-mean_path / average_path_length(self._sample_size))
