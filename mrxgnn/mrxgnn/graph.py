"""Population-level anatomical prior graph over ROIs.

Nodes = ROIs. Prior edges = top-k structural-covariance neighbours (|Pearson r| of MRI W-scores
across TRAINING subjects of the fold) plus homotopic (left-right) edges. The model adds its own
learned, subject-specific attention on top of this prior (see model.DenseGraphAttention).
"""
import numpy as np
import torch

from .atlas import hemisphere_pairs


def covariance_prior(W_mri, m_mri, rois, k=8):
    X = W_mri[m_mri > 0]
    C = np.corrcoef(X.T)
    C = np.nan_to_num(np.abs(C))
    np.fill_diagonal(C, 0)
    R = C.shape[0]
    A = np.zeros_like(C)
    nn_idx = np.argsort(-C, 1)[:, :k]
    for i in range(R):
        A[i, nn_idx[i]] = C[i, nn_idx[i]]
    try:
        for i, j in hemisphere_pairs(rois):
            A[i, j] = A[j, i] = max(A[i, j], 0.5)
    except KeyError:
        pass  # custom atlas without lh_/rh_ naming
    A = np.maximum(A, A.T) + np.eye(R)
    d = A.sum(1)
    A = A / np.sqrt(d[:, None] * d[None, :])
    return torch.tensor(A, dtype=torch.float32)
