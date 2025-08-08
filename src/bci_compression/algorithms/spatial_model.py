"""Spatial decorrelation and cross-channel predictive models.

Components:
  - OnlineDecorrelator: incremental PCA (covariance update), DCT fallback, graph Laplacian smoothing option.
  - PredictiveCrossChannel: linear predictors using neighbor channels (adjacency based on correlation or geometric layout).

Goal: Reduce inter-channel redundancy prior to temporal compression.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


def dct_basis(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    basis = np.cos(np.pi * (2 * t + 1) * k / (2 * n))
    basis[0] /= np.sqrt(2)
    return basis * np.sqrt(2 / n)


def graph_laplacian_filter(data: np.ndarray, adjacency: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    d = np.sum(adjacency, axis=1)
    L = np.diag(d) - adjacency
    return data - alpha * (L @ data)


@dataclass
class OnlineDecorrelatorState:
    mean: np.ndarray
    cov: np.ndarray
    count: int
    basis: Optional[np.ndarray]


class OnlineDecorrelator:
    """Incremental PCA-like decorrelator with fallback strategies.

    update(): consume new batch of shape (channels, samples)
    transform(): project data into decorrelated space
    inverse(): reconstruct (approximate) original space
    """

    def __init__(self, channels: int, max_components: Optional[int] = None, use_graph: bool = False):
        self.channels = channels
        self.max_components = max_components or channels
        self.use_graph = use_graph
        self.state = OnlineDecorrelatorState(
            mean=np.zeros(channels, dtype=np.float64),
            cov=np.zeros((channels, channels), dtype=np.float64),
            count=0,
            basis=None,
        )
        self._dct = dct_basis(channels)

    def update(self, batch: np.ndarray, adjacency: Optional[np.ndarray] = None) -> None:
        if batch.ndim == 2 and batch.shape[0] != self.channels:
            raise ValueError("Batch channel dimension mismatch")
        x = batch.astype(np.float64)
        if self.use_graph and adjacency is not None:
            x = graph_laplacian_filter(x, adjacency)
        # Compute batch mean
        batch_mean = x.mean(axis=1)
        # Update global mean (online)
        total = self.state.count + x.shape[1]
        delta = batch_mean - self.state.mean
        self.state.mean += delta * (x.shape[1] / total)
        # Center batch
        xc = x - batch_mean[:, None]
        # Update covariance (Bessel corrected)
        self.state.cov += xc @ xc.T
        self.state.count += x.shape[1]
        # Periodically (re)compute basis
        if self.state.count >= self.channels * 50 and (self.state.count // x.shape[1]) % 5 == 0:
            # Eigen decomposition
            c = self.state.cov / max(self.state.count - 1, 1)
            try:
                w, v = np.linalg.eigh(c)
                idx = np.argsort(w)[::-1]
                v = v[:, idx[: self.max_components]]
                self.state.basis = v.T  # shape (k, channels)
            except np.linalg.LinAlgError:
                self.state.basis = self._dct[: self.max_components]
        if self.state.basis is None:
            # DCT fallback until enough samples
            self.state.basis = self._dct[: self.max_components]

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] != self.channels:
            raise ValueError("Channel mismatch in transform")
        centered = data - self.state.mean[:, None]
        return (self.state.basis @ centered)  # type: ignore

    def inverse(self, coeffs: np.ndarray) -> np.ndarray:
        if self.state.basis is None:
            raise ValueError("Basis not initialized")
        recon = self.state.basis.T @ coeffs
        return recon + self.state.mean[:, None]

    def info(self) -> Dict[str, Any]:
        return {
            'channels': self.channels,
            'components': self.state.basis.shape[0] if self.state.basis is not None else 0,
            'samples_seen': self.state.count,
        }


class PredictiveCrossChannel:
    """Cross-channel linear predictors using neighbor weighting.

    Maintains simple linear regression weights per channel predicted from a
    set of neighbor channels (selected by correlation). Updated incrementally
    using a forgetting factor.
    """

    def __init__(self, channels: int, neighbors: int = 4, lr: float = 0.01, forgetting: float = 0.999):
        self.channels = channels
        self.neighbors = neighbors
        self.lr = lr
        self.forgetting = forgetting
        self.weights = np.zeros((channels, neighbors))
        self.neighbor_idx = np.zeros((channels, neighbors), dtype=np.int32)
        self.running_corr = np.zeros((channels, channels))
        self.count = 0

    def _update_correlations(self, batch: np.ndarray) -> None:
        # Incremental correlation estimate (cov normalization deferred)
        self.running_corr += (batch @ batch.T)

    def _select_neighbors(self) -> None:
        corr = self.running_corr.copy()
        np.fill_diagonal(corr, -np.inf)
        for ch in range(self.channels):
            idx = np.argsort(corr[ch])[::-1][: self.neighbors]
            self.neighbor_idx[ch] = idx

    def update(self, batch: np.ndarray) -> None:
        if batch.shape[0] != self.channels:
            raise ValueError("Channel mismatch in update")
        self._update_correlations(batch)
        self.count += batch.shape[1]
        if self.count % (self.channels * 20) == 0:
            self._select_neighbors()
        # Online weight update (simple LMS per channel)
        for ch in range(self.channels):
            neigh = self.neighbor_idx[ch]
            if np.all(neigh == 0) and self.count < self.channels * 20:
                continue  # not enough info yet
            X = batch[neigh]
            y = batch[ch]
            pred = (self.weights[ch][:, None] * X).sum(axis=0)
            err = y - pred
            grad = (err * X).mean(axis=1)
            self.weights[ch] = self.forgetting * self.weights[ch] + self.lr * grad

    def predict(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] != self.channels:
            raise ValueError("Channel mismatch in predict")
        preds = np.zeros_like(data)
        for ch in range(self.channels):
            neigh = self.neighbor_idx[ch]
            X = data[neigh]
            preds[ch] = (self.weights[ch][:, None] * X).sum(axis=0)
        return preds

    def residual(self, data: np.ndarray) -> np.ndarray:
        return data - self.predict(data)

    def info(self) -> Dict[str, Any]:
        return {
            'channels': self.channels,
            'neighbors': self.neighbors,
            'samples_seen': self.count,
        }


__all__ = [
    'OnlineDecorrelator',
    'PredictiveCrossChannel',
]
