# -*- coding: utf-8 -*-
"""
Standalone CNN first guess for SGD_jax.py: copy this file and first_guess_net_jax.pt next to it.
cnn_first_guess(noisy_psf) turns the batch of patches of the loop into starting values for the
gradient descent, in its units and conventions. Needs torch, numpy and jax only.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_guess_net_jax.pt")
_cache = {}


class FirstGuessNet(nn.Module):
    """ Same architecture as first_guess_network.FirstGuessNet, so the trained weights load. """

    def __init__(self, n_channels=6, n_outputs=8, patch_size=13):
        super().__init__()
        self.n_outputs = n_outputs
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
        )
        s = (patch_size // 2) // 2  # two maxpool(2): 13 -> 6 -> 3
        self.head = nn.Sequential(
            nn.Linear(128 * s * s, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 2 * n_outputs),  # mean and log-variance per output
        )

    def forward(self, x):
        out = self.head(self.conv(x).flatten(1))
        return out[:, :self.n_outputs], out[:, self.n_outputs:]


def load(path=CHECKPOINT):
    """ Returns (model, checkpoint dict), loaded once on CPU then cached. """
    if path not in _cache:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = FirstGuessNet(n_outputs=len(ckpt["label_names"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _cache[path] = (model, ckpt)
    return _cache[path]


def cnn_first_guess(noisy_psf, return_sigma=False, path=CHECKPOINT):
    """ Takes noisy_psf (NPSF, 3, 2, 13, 13) in photons, returns (x0, y0, z0, N0, rho0, eta0, delta0), each a
    (NPSF,) float32 jnp array in SGD_jax.py units: um, photons per plane, degrees. With return_sigma, also
    the (NPSF, 8) predicted 1-sigma uncertainties [x_um, y_um, z_um, mx, my, mz, delta_deg, N_per_plane]. """
    model, ckpt = load(path)
    p = np.asarray(noisy_psf, dtype=np.float32)
    x = torch.from_numpy(np.ascontiguousarray(p.reshape(len(p), 6, p.shape[-2], p.shape[-1]))) / ckpt["input_scale"]
    with torch.no_grad():
        mean, log_var = model(x)
    lm, ls = torch.as_tensor(ckpt["label_mean"]), torch.as_tensor(ckpt["label_std"])
    guess = (mean * ls + lm).numpy()
    sigma = (torch.exp(0.5 * log_var) * ls).numpy()

    # dipole axis -> angles of PSF_jax, m = (cos rho sin eta, sin rho sin eta, cos eta)
    m = guess[:, 3:6] / np.linalg.norm(guess[:, 3:6], axis=1, keepdims=True)
    rho = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
    eta = np.degrees(np.arccos(np.clip(m[:, 2], -1., 1.)))
    # kept away from the exp(100*overshoot) penalties of SGD_jax.py (delta in [1, 180], N >= 0)
    delta = np.clip(guess[:, 6], 5., 175.)
    n = np.clip(guess[:, 7], 50., None)

    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    out = (f(guess[:, 0]), f(guess[:, 1]), f(guess[:, 2]), f(n), f(rho), f(eta), f(delta))
    return out + (sigma,) if return_sigma else out
