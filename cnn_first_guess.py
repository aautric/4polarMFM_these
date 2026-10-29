# -*- coding: utf-8 -*-
"""
Standalone CNN first guess for SGD_jax_CNN.py: copy this file and first_guess_net_jax.pt next to it.
cnn_first_guess(noisy_psf) turns the batch of patches of the loop into starting values for the
gradient descent, in its units and conventions. Needs torch, numpy and jax only.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_guess_net_jax.pt")
# settings of the training data (train_settings of train_cnn_jax.py), for a checkpoint that does not store them (the
# first one, 2026-10-29): middle plane (um, middle_plane of SGD_jax.py) and z range (um) of train_cnn_jax.py of that
# day. The z of the network is the z whose PSF at the middle plane of the training looks like the PSF of the data
# (shifted by about 0.74*(1.4 - middle_plane) otherwise, see z_table); the network is wrong outside the z range
DEFAULT_TRAIN_SETTINGS = {'middle_plane': 1.4, 'z_um': (0., 2.)}
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


def training_settings(path=CHECKPOINT):
    """ Settings of the training data of the checkpoint (train_settings of train_cnn_jax.py): middle_plane, z_um, ... """
    return dict(DEFAULT_TRAIN_SETTINGS, **load(path)[1].get("train_settings", {}))


def cnn_first_guess(noisy_psf, return_sigma=False, path=CHECKPOINT, z_table=None):
    """ Takes noisy_psf (NPSF, 3, 2, 13, 13) in photons, returns (x0, y0, z0, N0, rho0, eta0, delta0, bg0), each a
    (NPSF,) float32 jnp array in SGD_jax.py units: um, photons of the whole molecule (3 planes x 2 polarisations,
    convention of PSF_jax since 2026-10-28), degrees, mean background level of the 6 channels (photons/pixel; None
    for a checkpoint without the background output). With return_sigma, also the predicted 1-sigma uncertainties of
    the labels of the checkpoint (label_names; z at the middle plane of the training, N of the molecule).
    z_table: (z, z_train), increasing: depths of the run and the depth whose PSF at the middle plane of the training
    matches each (cnn_z_equivalence of SGD_jax_CNN.py); the z of the network is converted to the run with it (clamped
    to the table). None: z of the network as it is (right for data at the middle plane of the training only). """
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
    # photons of the molecule. The first checkpoint was trained with N per plane (PSF_jax normalised each plane until
    # 2026-10-28): x3 for the molecule
    names = list(ckpt["label_names"])
    k = names.index("n_photons_molecule") if "n_photons_molecule" in names else names.index("n_photons_per_plane")
    per_plane = 3. if names[k] == "n_photons_per_plane" else 1.
    n = per_plane*np.clip(guess[:, k], 50., None)
    sigma[:, k] *= per_plane
    bg = np.clip(guess[:, names.index("background")], 0., None) if "background" in names else None

    z = guess[:, 2] if z_table is None else np.interp(guess[:, 2], z_table[1], z_table[0])

    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    out = (f(guess[:, 0]), f(guess[:, 1]), f(z), f(n), f(rho), f(eta), f(delta), None if bg is None else f(bg))
    return out + (sigma,) if return_sigma else out
