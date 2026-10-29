# -*- coding: utf-8 -*-
"""
First-guess CNN trained on PSFs from the reference JAX model (calibrated Jones matrices, SAF,
acquisition geometry), to initialize the fit of SGD_jax_CNN.py. A fixed set of
noiseless unit PSFs is generated once (--generate); photons, background and noise are drawn
on the fly at training time (--train). Patches use the SGD_jax.py layout, noisy_psf.reshape(N, 6, 13, 13).
Needs simu_PSF_polarMFM_JAX.py and cnn_first_guess.py (network architecture) next to this file, jax and torch.
Writes the dataset (--dataset, ~850 MB, not to be committed) and the checkpoint (--checkpoint, by default
first_guess_net_jax.pt next to this file, read by SGD_jax_CNN.py). The settings of the training are saved in the
checkpoint (train_settings) and read by cnn_first_guess.py.
Outputs: position, dipole axis, wobble, photons of the whole molecule and mean background level (photons/pixel,
start of the 6 background values fitted by SGD_jax_CNN.py). Camera noise: EMCCD or Poisson, with background
fluctuation and read noise, as the simulations of SGD_precision_claude.py.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT_DIR, "dataset_jax_slb.npz")  # default of --dataset
CHECKPOINT = os.path.join(ROOT_DIR, "first_guess_net_jax.pt")  # default of --checkpoint: the weights used by SGD_jax_CNN.py
sys.path.insert(0, ROOT_DIR)
INPUT_SCALE = 1000.  # divides the photon patches before the network, saved in the checkpoint

# labels, units of SGD_jax.py: x, y, z in um, dipole axis (mz >= 0), wobble cone delta (deg), photons of the whole
# molecule (3 planes x 2 polarisations, convention of PSF_jax since 2026-10-28), mean background level of the 6
# channels (photons/pixel, residual baseline included: the level the fit sees). The first checkpoint (2026-10-29)
# had 8 labels with 'n_photons_per_plane' and no background, cnn_first_guess.py reads both
LABEL_NAMES = ['x_um', 'y_um', 'z_um', 'mx', 'my', 'mz', 'delta_deg', 'n_photons_molecule', 'background']

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

# sampling ranges, from SLB-like to actin-like data (fits of 2026-10: SLB N ~1170 per molecule, background ~3.4;
# actin N ~2000-3600, background 20-35; photons/pixel, baseline removed by the reconstruction)
XY_UM = 0.30                 # uniform in +-0.30 um (~2.5 px)
Z_UM = (0., 2.0)
DELTA_DEG = (1., 180.)
N_MOLECULE = (300., 15000.)  # photons of the whole molecule, log-uniform
BACKGROUND = (1., 80.)       # photons/pixel, per patch, log-uniform, x U(0.85, 1.15) per channel
BASELINE = (-0.5, 0.5)       # residual of the baseline removed by the reconstruction, photons
# camera noise, as the simulations of SGD_precision_claude.py (noise_jax and emccd_jax of simu_PSF_polarMFM_JAX.py):
# a fraction P_EMCCD of the patches with the EMCCD chain (photoelectrons, EM register, read noise, ADC), the others
# with Poisson noise only; per patch a background fluctuation SIGMA_B (photons, per pixel, inside the shot noise) and
# a read noise SIGMA_R (photons). Presets of the simulations: SLB sigma_b 0.5, actin 3, read noise 0.4
P_EMCCD = 0.75
QE, EM, SENSITIVITY = 0.92, 200., 4.4  # quantum efficiency, EM gain, electrons per ADU (iXon 897, 2026-10-22)
SIGMA_B = (0., 4.)           # uniform
SIGMA_R = (0., 1.)           # uniform


def rot(angle):
    a = np.radians(angle)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def build_renderer():
    """ Returns render(xp, yp, zp, rho, eta, delta) -> (B, 3, 2, 13, 13) unit PSFs (1 photon for the whole molecule on
    the whole simulated field, PSF_jax since 2026-10-28: the 13 x 13 patches hold part of it), and the pixel size in um. """
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


def generate(n_train, n_val, chunk, dataset=DATASET, seed=0):
    """ Renders n_train + n_val unit PSFs with the JAX model and saves them with their geometry to dataset. """
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
    np.savez(dataset, psf=psf, geometry=geo, n_train=n_train, pixel_um=pixel_um, psf_normalisation='molecule')
    print(f"-> {dataset} ({len(geo)} PSF, pixel {pixel_um * 1000:.1f} nm)")


def geometry_to_labels(geo):
    """ (n, 6) geometry -> (n, 7) labels without photons: x, y, z, (mx, my, mz) with mz >= 0, delta. """
    rho, eta = np.radians(geo[:, 3]), np.radians(geo[:, 4])
    m = np.stack([np.cos(rho) * np.sin(eta), np.sin(rho) * np.sin(eta), np.cos(eta)], 1)
    return np.concatenate([geo[:, :3], m, geo[:, 5:6]], 1).astype(np.float32)


def draw_conditions(B, gen, dev):
    """ Draws per patch the photons of the molecule (B,), the background of the 6 channels (B, 6), the residual
    baseline (B, 1, 1, 1) and the camera: EMCCD or not (B, 1, 1, 1), background fluctuation and read noise (photons,
    (B, 1, 1, 1)). """
    import torch
    U = lambda lo, hi, *s: torch.rand(*s, generator=gen, device=dev) * (hi - lo) + lo
    n = torch.exp(U(np.log(N_MOLECULE[0]), np.log(N_MOLECULE[1]), B))
    bg = torch.exp(U(np.log(BACKGROUND[0]), np.log(BACKGROUND[1]), B, 1)) * U(0.85, 1.15, B, 6)
    emccd = U(0., 1., B, 1, 1, 1) < P_EMCCD
    return n, bg, U(*BASELINE, B, 1, 1, 1), emccd, U(*SIGMA_B, B, 1, 1, 1), U(*SIGMA_R, B, 1, 1, 1)


def camera_noise(expected, emccd, sigma_b, sigma_r, gen):
    """ Noisy values in photons (no baseline) of the expected photons (B, 6, 13, 13), as emccd_jax (emccd True) or
    noise_jax (False) of simu_PSF_polarMFM_JAX.py. The EM register (Gamma) uses the global torch generator. """
    import torch
    randn = lambda: torch.randn(expected.shape, generator=gen, device=expected.device)
    light = torch.clamp(expected + sigma_b * randn(), min=0.)
    poisson_only = torch.poisson(light, generator=gen) + sigma_r * randn()
    electrons = torch.poisson(QE * light, generator=gen)
    amplified = torch.where(electrons > 0, torch._standard_gamma(torch.clamp(electrons, min=1.)) * EM, torch.zeros_like(electrons))
    adu = torch.round((amplified + sigma_r * QE * EM * randn()) / SENSITIVITY)
    return torch.where(emccd, adu * SENSITIVITY / (QE * EM), poisson_only)


def draw_noisy(unit, gen):
    """ Takes (B, 6, 13, 13) unit PSFs (torch), draws photons, background, baseline and camera noise per patch,
    returns (noisy, n_photons of the molecule, background level = baseline + mean background of the 6 channels). """
    n, bg, base, emccd, sigma_b, sigma_r = draw_conditions(unit.shape[0], gen, unit.device)
    expected = unit * n[:, None, None, None] + bg[:, :, None, None]
    return base + camera_noise(expected, emccd, sigma_b, sigma_r, gen), n, base[:, 0, 0, 0] + bg.mean(1)


def heteroscedastic_loss(mean, log_var, target, min_log_var, max_log_var):
    """ Gaussian negative log-likelihood with a learned variance per output, log-variance clamped. """
    import torch
    log_var = torch.clamp(log_var, min_log_var, max_log_var)
    return (0.5 * torch.exp(-log_var) * (mean - target) ** 2 + 0.5 * log_var).mean()


def train_settings():
    """ Settings of the training data, saved in the checkpoint: cnn_first_guess.py converts z with middle_plane, and
    they tell where the network can be trusted. """
    return {'middle_plane': MIDDLE_PLANE, 'interplane': INTERPLANE, 'lambda_nm': LAMBDA_NM, 'rotation': ROTATION,
            'rotation2': ROTATION2, 'xy_um': XY_UM, 'z_um': Z_UM, 'delta_deg': DELTA_DEG, 'n_photons_molecule': N_MOLECULE,
            'background': BACKGROUND, 'baseline': BASELINE, 'p_emccd': P_EMCCD, 'QE': QE, 'EM': EM,
            'sensitivity': SENSITIVITY, 'sigma_b': SIGMA_B, 'sigma_r': SIGMA_R, 'aberrations': None,
            'registration_z': None, 'y_gain': None}


def train(epochs=200, batch_size=256, lr=1e-3, patience=12, min_log_var=-8., max_log_var=4., seed=0,
          dataset=DATASET, checkpoint=CHECKPOINT):
    """ Trains FirstGuessNet on dataset with noise drawn on the fly, saves the best model to checkpoint. """
    import copy
    import torch
    from cnn_first_guess import FirstGuessNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)  # global generator: EM register of camera_noise
    data = np.load(dataset)
    if "psf_normalisation" not in data.files or str(data["psf_normalisation"]) != "molecule":
        raise ValueError(f"{dataset}: unit PSFs not normalised on the whole molecule (made before 2026-10-28?), run --generate again")
    n_train = int(data["n_train"])
    unit = torch.as_tensor(data["psf"], device=device)
    geo_labels = torch.as_tensor(geometry_to_labels(data["geometry"]), device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    # label standardization from the sampling distribution, photons and background drawn as in draw_noisy
    n_draw, bg_draw, base_draw = draw_conditions(len(geo_labels), gen, device)[:3]
    all_labels = torch.cat([geo_labels, n_draw[:, None], (base_draw[:, 0, 0, 0] + bg_draw.mean(1))[:, None]], 1)
    lmean, lstd = all_labels[:n_train].mean(0), all_labels[:n_train].std(0)

    # fixed validation set: one noise draw per PSF
    val_x, val_n, val_bg = draw_noisy(unit[n_train:], torch.Generator(device=device).manual_seed(seed + 1))
    val_x = val_x / INPUT_SCALE
    val_y = (torch.cat([geo_labels[n_train:], val_n[:, None], val_bg[:, None]], 1) - lmean) / lstd

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
            xb, nb, bgb = draw_noisy(unit[idx], gen)
            yb = (torch.cat([geo_labels[idx], nb[:, None], bgb[:, None]], 1) - lmean) / lstd
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
                        'pixel_um': float(data["pixel_um"]), 'patch_layout': 'SGD_jax noisy_psf.reshape(N, 6, 13, 13)',
                        'train_settings': train_settings(), 'epoch': epoch, 'val_loss': val_loss},
                       checkpoint)
        else:
            stall += 1
            if stall >= patience:
                print(f"early stopping at epoch {epoch}, best val {best:.4f}")
                break
    print(f"-> {checkpoint}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-val", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--dataset", default=DATASET, help="npz of the unit PSFs (written by --generate, read by --train)")
    ap.add_argument("--checkpoint", default=CHECKPOINT, help="weights written by --train (default: the ones of SGD_jax_CNN.py)")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    if args.generate:
        generate(args.n_train, args.n_val, args.chunk, dataset=args.dataset)
    if args.train:
        train(epochs=args.epochs, dataset=args.dataset, checkpoint=args.checkpoint)
