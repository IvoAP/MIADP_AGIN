
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass, field

try:
    # sklearn is commonly available; used only for MI estimation
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
except Exception as e:
    mutual_info_classif = None
    mutual_info_regression = None


@dataclass
class MIADPAnonymizer:
    """
    MIADP (Mutual Information Adaptive Differential Privacy) with Laplace noise.

    Parameters
    ----------
    epsilon : float
        Global privacy budget. Lower => more noise; must be > 0.
    random_state : Optional[int]
        Seed for reproducibility.
    task_type : str
        "classification" or "regression" — controls MI estimator.
    """
    epsilon: float = 1.0
    random_state: Optional[int] = 42
    task_type: str = "classification"

    # internal fields (populated on fit)
    _colnames: list = field(default_factory=list, init=False, repr=False)
    _mins: np.ndarray = field(default=None, init=False, repr=False)
    _maxs: np.ndarray = field(default=None, init=False, repr=False)
    _mi: np.ndarray = field(default=None, init=False, repr=False)
    _weights: np.ndarray = field(default=None, init=False, repr=False)
    _epsilons: np.ndarray = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)
    _numeric_mask: np.ndarray = field(default=None, init=False, repr=False)

    def _validate_inputs(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Tuple[pd.DataFrame, np.ndarray]:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas DataFrame or numpy array.")
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if y is None:
            raise ValueError("y must be provided to compute mutual information.")

        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if self.task_type not in ("classification", "regression"):
            raise ValueError("task_type must be 'classification' or 'regression'.")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0.")

        return X.copy(), y_array

    def _scale_minmax(self, X_num: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mins = np.nanmin(X_num, axis=0)
        maxs = np.nanmax(X_num, axis=0)
        ranges = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
        X_scaled = (X_num - mins) / ranges
        # clip potential numerical issues
        X_scaled = np.clip(X_scaled, 0.0, 1.0)
        return X_scaled, mins, maxs

    def _estimate_mi(self, X_scaled: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        if mutual_info_classif is None or mutual_info_regression is None:
            raise ImportError("scikit-learn is required for mutual information estimation.")
        # Choose estimator based on task
        if self.task_type == "classification":
            # sklearn expects float X, discrete y
            mi = mutual_info_classif(X_scaled, y_array, random_state=self.random_state, discrete_features=False)
        else:
            mi = mutual_info_regression(X_scaled, y_array, random_state=self.random_state, discrete_features=False)
        # sanitize MI
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        mi = np.maximum(mi, 0.0)
        return mi

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the anonymizer: compute MI, normalize weights, and per-feature epsilons.
        """
        X_df, y_array = self._validate_inputs(X, y)
        self._colnames = list(X_df.columns)

        # Mask numeric columns; leave non-numeric untouched in transform
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        self._numeric_mask = X_df.columns.isin(numeric_cols)

        if not np.any(self._numeric_mask):
            raise ValueError("No numeric columns found in X. Provide numeric features to anonymize.")

        X_num = X_df.loc[:, numeric_cols].to_numpy(dtype=float)

        # Scale to [0,1] for sensitivity=1
        X_scaled, mins, maxs = self._scale_minmax(X_num)
        self._mins, self._maxs = mins, maxs

        # Mutual Information per numeric feature
        self._mi = self._estimate_mi(X_scaled, y_array)

        # Normalize MI to weights (if all zero, uniform)
        mi_sum = float(np.sum(self._mi))
        if mi_sum <= 1e-12:
            weights = np.ones_like(self._mi, dtype=float) / len(self._mi)
        else:
            weights = self._mi / mi_sum

        # avoid zero weights
        weights = np.maximum(weights, 1e-12)
        weights = weights / np.sum(weights)
        self._weights = weights

        # Allocate per-feature epsilon
        self._epsilons = self.epsilon * self._weights

        # RNG
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Laplace noise per feature using the learned epsilons.
        Non-numeric columns are passed through unchanged.
        """
        if self._epsilons is None or self._numeric_mask is None:
            raise RuntimeError("Call fit before transform.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._colnames)

        X_out = X.copy()
        numeric_cols = [c for c, m in zip(self._colnames, self._numeric_mask) if m]
        X_num = X_out.loc[:, numeric_cols].to_numpy(dtype=float)

        # Rescale to [0,1] using mins/maxs from fit
        mins = self._mins
        maxs = self._maxs
        ranges = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
        X_scaled = (X_num - mins) / ranges
        X_scaled = np.clip(X_scaled, 0.0, 1.0)

        # Sensitivity=1 in [0,1], Laplace scale b_i = 1 / epsilon_i
        b = 1.0 / self._epsilons  # per-feature
        # sample noise for each column
        for j in range(X_scaled.shape[1]):
            noise = self._rng.laplace(loc=0.0, scale=b[j], size=X_scaled.shape[0])
            X_scaled[:, j] = X_scaled[:, j] + noise

        # invert scaling
        X_anon_num = X_scaled * ranges + mins
        # clip to original bounds
        X_anon_num = np.clip(X_anon_num, mins, maxs)

        # write back
        X_out.loc[:, numeric_cols] = X_anon_num

        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def export_report(self) -> pd.DataFrame:
        """
        Return a DataFrame with MI, weights, and per-feature epsilons for numeric columns.
        """
        if self._epsilons is None:
            raise RuntimeError("Call fit first.")
        numeric_cols = [c for c, m in zip(self._colnames, self._numeric_mask) if m]
        df = pd.DataFrame({
            "feature": numeric_cols,
            "MI": self._mi,
            "weight": self._weights,
            "epsilon_i": self._epsilons,
        })
        return df
