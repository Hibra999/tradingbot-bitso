from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

@dataclass
class CausalRegimeModel:
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, min_covar=1e-3, random_state=self.random_state)
        self.columns: tuple[str, ...] = ()
        self.component_order: tuple[int, ...] = ()
        self.location = np.array([], dtype=float)
        self.scale = np.array([], dtype=float)

    def fit(self, training_features: pd.DataFrame, trend_column: str) -> "CausalRegimeModel":
        clean = training_features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 30: raise ValueError("At least 30 complete training rows are required for the HMM")
        self.columns = tuple(clean.columns)
        values = clean.to_numpy(dtype=float)
        self.location = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        self.model.fit((values - self.location) / self.scale)
        self.component_order = tuple(int(i) for i in np.argsort(self.model.means_[:, self.columns.index(trend_column)]))
        return self

    def forward_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self.component_order: raise RuntimeError("HMM is not fitted")
        values = (features.loc[:, self.columns].to_numpy(dtype=float) - self.location) / self.scale
        N, D = values.shape
        K = len(self.model.means_)
        output = np.full((N, 3), np.nan, dtype=float)
        inv_covs = [np.linalg.pinv(c) for c in self.model.covars_]
        log_dets = [np.linalg.slogdet(c)[1] for c in self.model.covars_]
        log_2pi_d = D * np.log(2 * np.pi)
        emissions = np.empty((N, K), dtype=float)
        for k in range(K):
            diff = values - self.model.means_[k]
            emissions[:, k] = -0.5 * (log_2pi_d + log_dets[k] + np.einsum("ij,jk,ik->i", diff, inv_covs[k], diff))
        log_start = np.log(np.clip(self.model.startprob_, 1e-300, None))
        log_trans = np.log(np.clip(self.model.transmat_, 1e-300, None))
        alpha: np.ndarray | None = None
        order = list(self.component_order)
        for i in range(N):
            if not np.isfinite(values[i]).all(): continue
            pred = log_start if alpha is None else logsumexp(alpha[:, None] + log_trans, axis=0)
            alpha_raw = emissions[i] + pred
            lse = logsumexp(alpha_raw)
            if np.isfinite(lse):
                alpha = alpha_raw - lse
            else:
                pred_lse = logsumexp(pred)
                alpha = pred - pred_lse if np.isfinite(pred_lse) else (log_start - logsumexp(log_start))
            output[i] = np.exp(alpha)[order]
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
