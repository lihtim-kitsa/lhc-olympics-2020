"""Permutation-invariant Deep Sets autoencoder for LHCO particle records.

Input tensors have shape ``(batch, particles, 5)`` and a boolean validity mask
of shape ``(batch, particles)``. Use :func:`encode_particles` on the raw
``(pt, eta, phi)`` triples first. The model's event score is masked per-particle
reconstruction MSE; it is a ranking score, not a calibrated likelihood.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


PARTICLE_FEATURES = ("pt_fraction", "log_pt_ratio", "log_pt_gev", "eta_scaled", "sin_phi", "cos_phi")


def encode_particles(raw: np.ndarray, eta_limit: float = 6.0):
    """Convert one raw LHCO event's flattened triples to features and mask.

    Positive finite-pT entries are particles; zero-padded and malformed entries
    are masked. Features retain relative pT and angular structure while being
    insensitive to a common pT rescaling. Returns ``(features, mask)``.
    """
    row = np.asarray(raw, dtype=np.float32)
    if row.ndim != 1 or row.size % 3:
        raise ValueError("raw event must be a flat array of (pt, eta, phi) triples")
    triples = row.reshape(-1, 3)
    valid = np.isfinite(triples).all(axis=1) & (triples[:, 0] > 0)
    selected = triples[valid]
    out = np.zeros((len(triples), len(PARTICLE_FEATURES)), dtype=np.float32)
    mask = valid.astype(np.bool_)
    if not len(selected):
        return out, mask
    pt = selected[:, 0]
    total_pt = float(pt.sum(dtype=np.float64))
    max_pt = float(pt.max())
    eta = np.clip(selected[:, 1], -eta_limit, eta_limit) / eta_limit
    phi = selected[:, 2]
    out[valid] = np.column_stack((pt / max(total_pt, 1e-12),
                                  np.log(np.maximum(pt / max_pt, 1e-12)),
                                  np.log1p(pt), eta, np.sin(phi), np.cos(phi))).astype(np.float32)
    return out, mask


class ParticleDeepSetsAutoencoder(nn.Module):
    """Masked permutation-equivariant reconstruction with invariant pooling."""

    def __init__(self, input_dim: int = len(PARTICLE_FEATURES), hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.particle_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.event_projection = nn.Sequential(nn.Linear(2 * hidden_dim, latent_dim), nn.SiLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have shape (batch, particles, {self.input_dim})")
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape (batch, particles)")
        mask = mask.bool()
        m = mask.unsqueeze(-1)
        h = self.particle_encoder(x) * m
        count = m.sum(dim=1).clamp_min(1)
        mean = h.sum(dim=1) / count
        # A masked max is stable for empty events; empty events are excluded by
        # the data builder and should not be used for training/scoring.
        maximum = h.masked_fill(~m, torch.finfo(h.dtype).min).amax(dim=1)
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        z = self.event_projection(torch.cat((mean, maximum), dim=-1))
        z_particles = z.unsqueeze(1).expand(-1, x.shape[1], -1)
        reconstruction = self.decoder(torch.cat((h, z_particles), dim=-1)) * m
        return reconstruction, z

    def anomaly_score(self, x: torch.Tensor, mask: torch.Tensor):
        """Return masked event-wise reconstruction MSE (higher is stranger)."""
        reconstruction, _ = self(x, mask)
        m = mask.bool().unsqueeze(-1)
        per_particle = ((x - reconstruction) ** 2).mean(dim=-1)
        return (per_particle * mask.bool()).sum(dim=1) / mask.bool().sum(dim=1).clamp_min(1)
