# -*- coding: utf-8 -*-
"""
First-guess CNN trained on PSFs from the reference JAX model (calibrated Jones matrices, SAF,
acquisition geometry), to initialize the gradient descent of SGD_jax.py. A fixed set of
noiseless unit PSFs is generated once (--generate); photons, background and noise are drawn
on the fly at training time (--train). Patches use the SGD_jax.py layout, noisy_psf.reshape(N, 6, 13, 13).
Needs simu_PSF_polarMFM_JAX.py and cnn_first_guess.py (network architecture) next to this file, jax and torch.
Writes dataset_jax_slb.npz (~850 MB, not to be committed) and first_guess_net_jax.pt (read by SGD_jax.py) next to it.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT_DIR, "dataset_jax_slb.npz")
CHECKPOINT = os.path.join(ROOT_DIR, "first_guess_net_jax.pt")  # --train overwrites the weights used by SGD_jax.py
sys.path.insert(0, ROOT_DIR)
INPUT_SCALE = 1000.  # divides the photon patches before the network, saved in the checkpoint

# labels, units of SGD_jax.py: x, y, z in um, dipole axis (mz >= 0), wobble cone delta (deg), photons per plane
LABEL_NAMES = ['x_um', 'y_um', 'z_um', 'mx', 'my', 'mz', 'delta_deg', 'n_photons_per_plane']

# acquisition (SGD_jax.py, SLB data, interplane from pre_loc_deltaz_389.csv): must stay equal to the values of
# SGD_jax.py, otherwise the CNN no longer matches the model of the fit and has to be trained again
LAMBDA_NM = 620.
MIDDLE_PLANE = 1.4
INTERPLANE = 0.389
J1 = np.array([[0.89097912 + 0.j, -0.25009759 - 0.43946158j], [0.43033225 - 0.14481147j, 0.6335862 + 0.58557087j]])
J2 = np.array([[0.31676515 + 0.j, -0.93297585 - 0.07330244j], [-0.58381747 + 0.74754063j, -0.26412184 + 0.23328625j]])
ROTATION, ROTATION2 = 7, 2
N_GRID, L_PIXEL, NA, MAG_OBJ, F_TUBE, MAG_RELAY = 80, 16, 1.4, 100, 200, 200 / 150
HALF = 6  # 13x13 patches

# sampling ranges, wide around the fitted values of the SLB run (z 0.35-1.24 um, N 230-650 per plane, 5-95 %)
XY_UM = 0.30                 # uniform in +-0.30 um (~2.5 px)
Z_UM = (0., 2.0)
DELTA_DEG = (1., 180.)
N_PER_PLANE = (100., 3000.)  # log-uniform
BACKGROUND = (1., 8.)        # photons/pixel above the camera baseline, per patch, x U(0.85, 1.15) per channel
GAIN = (1.2, 2.4)            # noise variance = gain * (value - baseline)
BASELINE = (2.5, 4.5)        # camera baseline, photons


def rot(angle):
    a = np.radians(angle)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def build_renderer():
    """ Returns render(xp, yp, zp, rho, eta, delta) -> (B, 3, 2, 13, 13) unit PSFs (1 photon per plane), and the pixel size in um. """
    import jax
    import jax.numpy as jnp
    from simu_PSF_polarMFM_JAX import (vectorial_BFP_perfect_focus_jax, padding_jax, pad_jax,
                                       generate_zernike_base_jax, compute_M_jax, PSF_jax)

    J_dichroic = np.array([J1 @ rot(ROTATION), J2 @ rot(-ROTATION2), J1 @ rot(ROTATION)])
    d = jnp.array([MIDDLE_PLANE - INTERPLANE, MIDDLE_PLANE, MIDDLE_PLANE + INTERPLANE])
    second_plane = jnp.array([d[1] - d[0], 0., d[1] - d[2]])
    lambd = jnp.array(LAMBDA_NM)
    N = jnp.array(N_GRID)
    # same calls as SGD_jax.py: SAF in the BFP fields, compute_M_jax with its defaults
    xx, yy, th1, phi, (Ex0, Ex1, Ex2), (Ey0, Ey1, Ey2), r, r_cut, _, k_, f_o, _ = vectorial_BFP_perfect_focus_jax(
        N, NA=NA, mag=MAG_OBJ, lambd_nm=lambd, f_tube_mm=F_TUBE, J_dichroic=J_dichroic, SAF=True)
    u, v, Npadding = padding_jax(r, r_cut, k_, f_o, N=N, l_pixel=jnp.array(L_PIXEL), NA=NA, mag=MAG_OBJ,
                                 lambd=lambd, f_tube=F_TUBE, MAG=MAG_RELAY)
    zb = generate_zernike_base_jax(r_cut=r_cut, N=N_GRID, zernike_order=4, skip_indices={0, 1, 2, 3})
    P = lambda a: pad_jax(a, Npadding).astype(jnp.complex64)
    xx, yy, th1, phi = P(xx), P(yy), P(th1), P(phi)
    Ex0, Ex1, Ex2, Ey0, Ey1, Ey2 = P(Ex0), P(Ex1), P(Ex2), P(Ey0), P(Ey1), P(Ey2)
    zb = P(zb)
    zc = jnp.zeros((3, 15)).astype(jnp.complex64)
    pixel_um = float((u[0, 1] - u[0, 0]) / (MAG_OBJ * MAG_RELAY))

    @jax.jit
    def render(xp, yp, zp, rho, eta, delta):
        B = xp.shape[0]
        M = compute_M_jax(xp=xp, yp=yp, zp=zp, d=jnp.full(B, -d[1]), x=xx, y=yy, th1=th1, phi=phi,
                          Ex0=Ex0, Ex1=Ex1, Ex2=Ex2, Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zb,
                          zernike_coefs_x=zc, zernike_coefs_y=zc, second_plane=second_plane,
                          polar_projections=jnp.array([0., 45., 0.]), lambd=lambd, f_tube=F_TUBE)
        h = PSF_jax(rho=rho, eta=eta, delta=delta, M=M, N_photons=jnp.ones(B))
        c = h.shape[-1] // 2
        return h[:, :, :, c - HALF:c + HALF + 1, c - HALF:c + HALF + 1]

    return render, pixel_um


def sample_geometry(n, rng):
    """ Returns (n, 6) [x_um, y_um, z_um, rho_deg, eta_deg, delta_deg], dipole axes uniform on the half sphere mz >= 0. """
    return np.stack([rng.uniform(-XY_UM, XY_UM, n), rng.uniform(-XY_UM, XY_UM, n), rng.uniform(*Z_UM, n),
                     rng.uniform(0., 360., n), np.degrees(np.arccos(rng.uniform(0., 1., n))),
                     rng.uniform(*DELTA_DEG, n)], axis=1).astype(np.float32)


def generate(n_train, n_val, chunk, seed=0):
    """ Renders n_train + n_val unit PSFs with the JAX model and saves them with their geometry to DATASET. """
    import jax.numpy as jnp
    render, pixel_um = build_renderer()
    rng = np.random.default_rng(seed)
    geo = sample_geometry(n_train + n_val, rng)
    psf = np.zeros((len(geo), 6, 2 * HALF + 1, 2 * HALF + 1), np.float32)
    t0 = time.time()
    for k in range(0, len(geo), chunk):
        g = jnp.asarray(geo[k:k + chunk])
        h = render(g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5])
        psf[k:k + chunk] = np.asarray(h).reshape(-1, 6, 2 * HALF + 1, 2 * HALF + 1)
        if (k // chunk) % 50 == 0:
            el = time.time() - t0
            print(f"  {k + len(g)}/{len(geo)}  ({el / 60:.1f} min, ~{el / (k + len(g)) * (len(geo) - k - len(g)) / 60:.0f} min left)", flush=True)
    np.savez(DATASET, psf=psf, geometry=geo, n_train=n_train, pixel_um=pixel_um)
    print(f"-> {DATASET} ({len(geo)} PSF, pixel {pixel_um * 1000:.1f} nm)")


def geometry_to_labels(geo):
    """ (n, 6) geometry -> (n, 7) labels without photons: x, y, z, (mx, my, mz) with mz >= 0, delta. """
    rho, eta = np.radians(geo[:, 3]), np.radians(geo[:, 4])
    m = np.stack([np.cos(rho) * np.sin(eta), np.sin(rho) * np.sin(eta), np.cos(eta)], 1)
    return np.concatenate([geo[:, :3], m, geo[:, 5:6]], 1).astype(np.float32)


def draw_noisy(unit, gen):
    """ Takes (B, 6, 13, 13) unit PSFs (torch), draws photons, background, gain and baseline per patch, returns (noisy, n_photons). """
    import torch
    B, dev = unit.shape[0], unit.device
    U = lambda lo, hi, *s: torch.rand(*s, generator=gen, device=dev) * (hi - lo) + lo
    n = torch.exp(U(np.log(N_PER_PLANE[0]), np.log(N_PER_PLANE[1]), B))
    bg = U(*BACKGROUND, B, 1) * U(0.85, 1.15, B, 6)
    gain, base = U(*GAIN, B, 1, 1, 1), U(*BASELINE, B, 1, 1, 1)
    mean = unit * n[:, None, None, None] + bg[:, :, None, None]
    return base + gain * torch.poisson(mean / gain, generator=gen), n


def heteroscedastic_loss(mean, log_var, target, min_log_var, max_log_var):
    """ Gaussian negative log-likelihood with a learned variance per output, log-variance clamped. """
    import torch
    log_var = torch.clamp(log_var, min_log_var, max_log_var)
    return (0.5 * torch.exp(-log_var) * (mean - target) ** 2 + 0.5 * log_var).mean()


def train(epochs=200, batch_size=256, lr=1e-3, patience=12, min_log_var=-8., max_log_var=4., seed=0):
    """ Trains FirstGuessNet on DATASET with noise drawn on the fly, saves the best model to CHECKPOINT. """
    import copy
    import torch
    from cnn_first_guess import FirstGuessNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(DATASET)
    n_train = int(data["n_train"])
    unit = torch.as_tensor(data["psf"], device=device)
    geo_labels = torch.as_tensor(geometry_to_labels(data["geometry"]), device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    # label standardization from the sampling distribution, photons drawn as in draw_noisy
    n_draw = torch.exp(torch.rand(len(geo_labels), generator=gen, device=device) * np.log(N_PER_PLANE[1] / N_PER_PLANE[0]) + np.log(N_PER_PLANE[0]))
    all_labels = torch.cat([geo_labels, n_draw[:, None]], 1)
    lmean, lstd = all_labels[:n_train].mean(0), all_labels[:n_train].std(0)

    # fixed validation set: one noise draw per PSF
    val_x, val_n = draw_noisy(unit[n_train:], torch.Generator(device=device).manual_seed(seed + 1))
    val_x = val_x / INPUT_SCALE
    val_y = (torch.cat([geo_labels[n_train:], val_n[:, None]], 1) - lmean) / lstd

    model = FirstGuessNet(n_channels=6, n_outputs=len(LABEL_NAMES), patch_size=2 * HALF + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, stall = float("inf"), None, 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, generator=gen, device=device)
        losses = []
        for k in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[k:k + batch_size]
            xb, nb = draw_noisy(unit[idx], gen)
            yb = (torch.cat([geo_labels[idx], nb[:, None]], 1) - lmean) / lstd
            mean, log_var = model(xb / INPUT_SCALE)
            loss = heteroscedastic_loss(mean, log_var, yb, min_log_var, max_log_var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            outs = [model(val_x[k:k + 1024]) for k in range(0, len(val_x), 1024)]
            vm, vlv = torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
            val_loss = heteroscedastic_loss(vm, vlv, val_y, min_log_var, max_log_var).item()
            rmse = (((vm - val_y) ** 2).mean(0).sqrt() * lstd).cpu().numpy()
        print(f"epoch {epoch} ({time.time() - t0:.0f} s): train {np.mean(losses):.4f}  val {val_loss:.4f}")
        print("  RMSE: " + "  ".join(f"{name}={v:.4g}" for name, v in zip(LABEL_NAMES, rmse)), flush=True)

        if val_loss < best:
            best, best_state, stall = val_loss, copy.deepcopy(model.state_dict()), 0
            torch.save({'model_state_dict': best_state, 'label_mean': lmean.cpu().numpy(),
                        'label_std': lstd.cpu().numpy(), 'label_names': LABEL_NAMES, 'input_scale': INPUT_SCALE,
                        'pixel_um': float(data["pixel_um"]), 'patch_layout': 'SGD_jax noisy_psf.reshape(N, 6, 13, 13)'},
                       CHECKPOINT)
        else:
            stall += 1
            if stall >= patience:
                print(f"early stopping at epoch {epoch}, best val {best:.4f}")
                break
    print(f"-> {CHECKPOINT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-val", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    if args.generate:
        generate(args.n_train, args.n_val, args.chunk)
    if args.train:
        train()
