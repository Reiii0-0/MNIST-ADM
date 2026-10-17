"""
Test-Time Prior Adaptation (TTPA) under Joint Covariate and Label Shift.
Final Project — Analisis Data Multivariat (ADM), Teknik Informatika ITS

Based on:
1. Saerens, Latinne, & Decaestecker (Neural Computation 2002):
   "Adjusting the Outputs of a Classifier to New a Priori Probabilities"
2. Sun, Murphy, Ebrahimi, & D'Amour (NeurIPS 2023):
   "Beyond Invariance: Test-Time Label-Shift Adaptation"
3. Park, Yang, Choo, & Yun (ICCV 2023):
   "Label Shift Adapter for Test-Time Adaptation under Covariate and Label Shifts"
4. Zhao, Chen, & Xia (ICLR 2023):
   "DELTA: Degradation-Free Fully Test-Time Adaptation"

This module provides a fully vectorized, mathematically sound Expectation-Maximization (EM)
algorithm with symmetric Dirichlet MAP smoothing to adapt pre-trained classifier posteriors
to unknown target class distributions without human labels or hardcoded zero masking.
"""

from typing import Union, Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestTimePriorAdapter:
    """
    Expectation-Maximization (EM) Prior Shift Adapter with Dirichlet MAP Smoothing.

    Adjusts classifier posterior probabilities P_s(y | x) (trained under source prior P_s(y))
    to unknown target domain prior P_t(y) by maximizing the marginal test likelihood:
        L(pi) = sum_{i=1}^N log [ sum_{k=0}^{K-1} P_s(y=k | x_i) * (pi_k / P_s(y=k)) ]

    Equipped with symmetric Dirichlet prior Dir(alpha) to prevent probability collapse
    and guarantee 100% mathematical safety for unseen evaluation phases.
    """

    def __init__(
        self,
        alpha_dirichlet: float = 1.05,
        temperature: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        source_priors: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        eps: float = 1e-12,
    ):
        """
        Args:
            alpha_dirichlet: Dirichlet prior parameter (alpha >= 1.0).
                             alpha = 1.0 corresponds to standard MLE (flat prior).
                             alpha > 1.0 adds symmetric Laplace/Dirichlet pseudo-counts
                             gamma = alpha - 1.0, guaranteeing non-zero support for all classes.
            temperature: Calibration temperature T > 0 applied to logits/probs before EM.
                         T = 1.0 leaves probabilities unchanged.
            max_iter: Maximum EM iterations before termination.
            tol: Convergence tolerance threshold for max absolute change in class priors.
            source_priors: Array of source domain prior probabilities P_s(y).
                           Defaults to uniform prior (1/K) across all classes.
            eps: Small epsilon constant for numerical stability and zero-division protection.
        """
        if alpha_dirichlet < 1.0:
            raise ValueError(f"alpha_dirichlet must be >= 1.0, got {alpha_dirichlet}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0.0, got {temperature}")

        self.alpha_dirichlet = float(alpha_dirichlet)
        self.temperature = float(temperature)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.source_priors = source_priors
        self.eps = float(eps)

        # Fitted attributes
        self.target_prior_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.history_prior_: list = []
        self.history_log_likelihood_: list = []
        self.converged_: bool = False

    def _prepare_inputs(
        self, probs: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[np.ndarray, bool, Any]:
        """Validates input array/tensor and converts to NumPy array."""
        is_torch = False
        torch_device = None
        torch_dtype = None

        if HAS_TORCH and isinstance(probs, torch.Tensor):
            is_torch = True
            torch_device = probs.device
            torch_dtype = probs.dtype
            probs_np = probs.detach().cpu().numpy().astype(np.float64)
        else:
            probs_np = np.asarray(probs, dtype=np.float64)

        if probs_np.ndim != 2:
            raise ValueError(f"Expected 2D probability array (N, K), got shape {probs_np.shape}")

        N, K = probs_np.shape
        if K < 2:
            raise ValueError(f"Number of classes K must be >= 2, got {K}")

        # Normalize rows to sum to 1 if needed
        row_sums = probs_np.sum(axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            probs_np = np.clip(probs_np, self.eps, None)
            probs_np /= probs_np.sum(axis=1, keepdims=True)

        return probs_np, is_torch, (torch_device, torch_dtype)

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Applies temperature scaling to probabilities in log-space."""
        if self.temperature == 1.0:
            return probs.copy()
        log_p = np.log(np.clip(probs, self.eps, 1.0)) / self.temperature
        log_p -= np.max(log_p, axis=1, keepdims=True)
        scaled_p = np.exp(log_p)
        return scaled_p / np.sum(scaled_p, axis=1, keepdims=True)

    def fit(self, probs: Union[np.ndarray, "torch.Tensor"]) -> "TestTimePriorAdapter":
        """
        Estimates target domain prior P_t(y) using EM with Dirichlet MAP smoothing.

        Args:
            probs: (N, K) predicted posterior probabilities from source model.

        Returns:
            self
        """
        probs_np, _, _ = self._prepare_inputs(probs)
        N, K = probs_np.shape

        # Source priors
        if self.source_priors is None:
            p_source = np.ones(K, dtype=np.float64) / K
        else:
            if HAS_TORCH and isinstance(self.source_priors, torch.Tensor):
                p_source = self.source_priors.detach().cpu().numpy().astype(np.float64)
            else:
                p_source = np.asarray(self.source_priors, dtype=np.float64)
            p_source = p_source / p_source.sum()

        # Apply temperature calibration if requested
        p_calibrated = self._apply_temperature(probs_np)

        # Precompute likelihood ratio matrix: q_ik = P_s(y=k | x_i) / P_s(y=k)
        q = p_calibrated / (p_source[np.newaxis, :] + self.eps)

        # Initialize target prior as uniform (or source prior)
        pi = np.ones(K, dtype=np.float64) / K
        gamma = max(0.0, self.alpha_dirichlet - 1.0)

        self.history_prior_ = [pi.copy()]
        self.history_log_likelihood_ = []
        self.converged_ = False

        for it in range(1, self.max_iter + 1):
            pi_prev = pi.copy()

            # -------------------------------------------------------------
            # E-step: Compute posterior responsibilities w_ik = P_t(y=k | x_i)
            # w_ik = (pi_k * q_ik) / sum_j (pi_j * q_ij)
            # -------------------------------------------------------------
            numerator = q * pi[np.newaxis, :]  # Shape (N, K)
            marginal_x = np.sum(numerator, axis=1, keepdims=True) + self.eps
            w = numerator / marginal_x

            # Marginal test log-likelihood
            log_lik = float(np.sum(np.log(marginal_x)))
            self.history_log_likelihood_.append(log_lik)

            # -------------------------------------------------------------
            # M-step: Maximize expected log-likelihood with Dirichlet MAP
            # pi_k^{(t+1)} = ( sum_{i=1}^N w_ik + gamma ) / ( N + K * gamma )
            # -------------------------------------------------------------
            class_weights_sum = np.sum(w, axis=0)  # Shape (K,)
            pi = (class_weights_sum + gamma) / (N + K * gamma)
            pi /= pi.sum()  # Guard against precision drift

            self.history_prior_.append(pi.copy())

            # Check convergence
            delta = float(np.max(np.abs(pi - pi_prev)))
            if delta < self.tol:
                self.converged_ = True
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.target_prior_ = pi
        return self

    def transform(
        self, probs: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Adjusts probabilities using the fitted target prior via Bayes' rule:
            P_t(y=k | x_i) = P_s(y=k | x_i) * (pi_k / P_s(y=k)) / Z_i

        Args:
            probs: (N, K) predicted posterior probabilities.

        Returns:
            Adjusted posterior probabilities (same type and device as input).
        """
        if self.target_prior_ is None:
            raise RuntimeError("TestTimePriorAdapter is not fitted yet! Call fit() first.")

        probs_np, is_torch, (device, dtype) = self._prepare_inputs(probs)
        K = probs_np.shape[1]

        if self.source_priors is None:
            p_source = np.ones(K, dtype=np.float64) / K
        else:
            if HAS_TORCH and isinstance(self.source_priors, torch.Tensor):
                p_source = self.source_priors.detach().cpu().numpy().astype(np.float64)
            else:
                p_source = np.asarray(self.source_priors, dtype=np.float64)
            p_source = p_source / p_source.sum()

        p_calibrated = self._apply_temperature(probs_np)
        q = p_calibrated / (p_source[np.newaxis, :] + self.eps)

        numerator = q * self.target_prior_[np.newaxis, :]
        denominator = np.sum(numerator, axis=1, keepdims=True) + self.eps
        adjusted_probs_np = numerator / denominator

        if is_torch:
            return torch.from_numpy(adjusted_probs_np).to(device=device, dtype=dtype)
        return adjusted_probs_np

    def fit_transform(
        self, probs: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Fits the target prior and returns adjusted probabilities."""
        return self.fit(probs).transform(probs)

    def predict(
        self, probs: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Predicts class labels (argmax) after prior adaptation."""
        adjusted_probs = self.transform(probs)
        if HAS_TORCH and isinstance(adjusted_probs, torch.Tensor):
            return torch.argmax(adjusted_probs, dim=1)
        return np.argmax(adjusted_probs, axis=1)

    def get_diagnostics(
        self,
        probs_orig: Union[np.ndarray, "torch.Tensor"],
        class_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Computes diagnostic statistics comparing original vs adapted predictions.
        """
        probs_orig_np, _, _ = self._prepare_inputs(probs_orig)
        probs_adj_np = self.transform(probs_orig_np)

        preds_orig = np.argmax(probs_orig_np, axis=1)
        preds_adj = np.argmax(probs_adj_np, axis=1)

        K = probs_orig_np.shape[1]
        counts_orig = np.bincount(preds_orig, minlength=K)
        counts_adj = np.bincount(preds_adj, minlength=K)

        flipped_mask = preds_orig != preds_adj
        n_flipped = int(np.sum(flipped_mask))

        class_shifts = []
        for c in range(K):
            name = class_names.get(c, str(c)) if class_names else str(c)
            c_orig = int(counts_orig[c])
            c_adj = int(counts_adj[c])
            prior_pct = float(self.target_prior_[c] * 100.0) if self.target_prior_ is not None else 0.0
            class_shifts.append({
                "class_id": c,
                "class_name": name,
                "orig_count": c_orig,
                "adj_count": c_adj,
                "diff": c_adj - c_orig,
                "estimated_prior_pct": prior_pct,
            })

        return {
            "n_samples": len(probs_orig_np),
            "n_classes": K,
            "n_iter": self.n_iter_,
            "converged": self.converged_,
            "n_flipped": n_flipped,
            "pct_flipped": float(n_flipped / len(probs_orig_np) * 100.0),
            "class_shifts": class_shifts,
            "target_prior": self.target_prior_.copy() if self.target_prior_ is not None else None,
        }


def adapt_test_time_priors(
    probs: Union[np.ndarray, "torch.Tensor"],
    alpha_dirichlet: float = 1.05,
    temperature: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    source_priors: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> Tuple[Union[np.ndarray, "torch.Tensor"], np.ndarray]:
    """
    Functional interface for Test-Time Prior Adaptation.

    Args:
        probs: (N, K) array or tensor of predicted class probabilities.
        alpha_dirichlet: Dirichlet prior hyperparameter (>= 1.0).
        temperature: Temperature calibration parameter (> 0.0).
        max_iter: Max EM iterations.
        tol: Convergence tolerance.
        source_priors: Source training prior distribution.

    Returns:
        (adjusted_probs, estimated_target_prior)
    """
    adapter = TestTimePriorAdapter(
        alpha_dirichlet=alpha_dirichlet,
        temperature=temperature,
        max_iter=max_iter,
        tol=tol,
        source_priors=source_priors,
    )
    adjusted_probs = adapter.fit_transform(probs)
    return adjusted_probs, adapter.target_prior_
