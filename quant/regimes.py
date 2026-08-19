from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


@dataclass
class CausalRegimeModel:
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            min_covar=1e-4,
            random_state=self.random_state,
        )
        self.columns: tuple[str, ...] = ()
        self.component_order: tuple[int, ...] = ()
        self.location = np.array([], dtype=float)
        self.scale = np.array([], dtype=float)

    def fit(self, training_features: pd.DataFrame, trend_column: str) -> "CausalRegimeModel":
        clean = training_features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 30:
            raise ValueError("At least 30 complete training rows are required for the HMM")
        self.columns = tuple(clean.columns)
        values = clean.to_numpy(dtype=float)
        self.location = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        self.model.fit((values - self.location) / self.scale)
        trend_index = self.columns.index(trend_column)
        self.component_order = tuple(int(i) for i in np.argsort(self.model.means_[:, trend_index]))
        return self

    def forward_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Forward-filter probabilities without future-aware smoothing."""
        if not self.columns or not self.component_order:
            raise RuntimeError("HMM is not fitted")
        values = (features.loc[:, self.columns].to_numpy(dtype=float) - self.location) / self.scale
        output = np.full((len(values), 3), np.nan, dtype=float)
        log_start = np.log(np.clip(self.model.startprob_, 1e-300, None))
        log_transition = np.log(np.clip(self.model.transmat_, 1e-300, None))
        alpha: np.ndarray | None = None
        for row_index, row in enumerate(values):
            if not np.isfinite(row).all():
                continue
            emission = np.asarray(
                [multivariate_normal.logpdf(row, mean=mean, cov=covariance, allow_singular=True) for mean, covariance in zip(self.model.means_, self.model.covars_)],
                dtype=float,
            )
            alpha = emission + (log_start if alpha is None else logsumexp(alpha[:, None] + log_transition, axis=0))
            alpha -= logsumexp(alpha)
            output[row_index] = np.exp(alpha)[list(self.component_order)]
        return pd.DataFrame(output, index=features.index, columns=("regime_bear", "regime_neutral", "regime_bull"))

    def parameters(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "input_location": self.location.tolist(),
            "input_scale": self.scale.tolist(),
            "component_order": list(self.component_order),
            "start_probability": self.model.startprob_.tolist(),
            "transition_matrix": self.model.transmat_.tolist(),
            "means": self.model.means_.tolist(),
            "covariances": self.model.covars_.tolist(),
        }
