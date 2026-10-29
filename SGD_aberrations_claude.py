# -*- coding: utf-8 -*-
"""
Copy of SGD_aberrations.py: aberrations of the 4polarMFM measured on the z-stacks of beads, with the model of SGD_jax.py.

Each isolated bead is fitted on its whole z-stack (all the frames together, focus of frame k = d0 + dz_scale x stage
z of frame k), in the reconstruction of the data (same calibration.npz, same channels and orientation as SGD_jax.py),
with the calibration of a real run (J1, J2, rotations, y gains, planes) read from its config.txt.

Bead model (the dipole model with delta = 180 alone does not describe the beads: the measured PSF of each polarisation
channel is rounder than the model, see the bead fit of 2026-10-27), options of the second step:
    - isotropic emission: delta = 180 (always)
    - conv: gaussian blur of the image of each channel, sigma free (size of the bead, residual vibrations, ...)
    - na: effective numerical aperture, soft edge of the pupil (light of the high angles lost in the relays/grating)
    - leak: fraction of the light of the other polarisation in each channel, per plane
    - xyframe: lateral position free in each frame (drift of the stage during the stack)
    - regz: lateral offset of each plane and channel vs the stage z measured on the beads (registration_z.npz, as
      registration_z in SGD_jax.py), fixed
    - n2: index of the medium of the beads (SAF light), fixed
    - zernike: coefficients of the Noll modes of `noll` in each plane and channel (x and y separately)
Step 1: position, focus, photons and background with the bead model at its start values, no aberration.
Step 2: everything together. Minimisation: L-BFGS (scipy) with the gradients of jax, Gaussian likelihood with the
noise of the EMCCD var = gain mu + V (V: variance without light of each reconstructed channel).

Headless:  python SGD_aberrations_claude.py --options conv na --tag conv_na
@author: Amaury (copy modified by Claude)
"""
#%%
import os, sys, glob, time, argparse, logging
from pathlib import Path
import numpy as np
import cv2, tifffile
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
logging.getLogger('tifffile').setLevel(logging.ERROR)
import SGD_precision_claude as S
import simu_PSF_polarMFM_JAX as SP
from density_experimental import run_parameters
from reconstruction_juliette import subcanals, measure_baseline, extract_raw_ome, dark_variance, interpolation_factor

# %% PARAMETERS TO BE DEFINED
calibration_file = '/mnt/d/AMAURY/DATA/0_papier/actin2/calibr/calibration.npz' # reconstruction of the data
bead_files = sorted(glob.glob('/mnt/d/AMAURY/DATA/2026_09_18_calibration/beads/*/images/RAW_DATA/image_Pos0.ome.tif'))
registration_file = '/mnt/d/AMAURY/DATA/0_papier/actin2/calibr/registration_z.npz' # option regz
run_config = '/mnt/d/AMAURY/DATA/0_papier/actin2/NPZ_2026-10-22_17h02/config.txt' # calibration of the model
out_folder = '/mnt/d/AMAURY/DATA/2026_09_18_calibration/beads/aberrations_beads'
QE = 0.92
sensitivity = 4.4 # e-/ADU, see SGD_jax.py
EM = None # EM gain of the bead stacks: None = Micro-Manager metadata
dz_step = 0.1 # um, stage step of the z-stacks
frames_around = 12 # frames kept on each side of the focus of the middle plane (+-1.2 um)
half = 8 # patch of (2 half + 1)^2 pixels
n_beads = 10 # brightest isolated beads fitted
isolation = 16 # px, no other bead closer than this
bead_z = 0.05 # um, emitter (centre of the bead) above the coverslip
n2_beads = None # index of the medium of the beads, None = n2 of simu_PSF_polarMFM_JAX.py (1.33)
options = ['conv', 'na'] # bead model, see above
noll = [5, 6, 7, 8, 9, 10, 11] # fitted Zernike modes (Noll), the same list in the 3 planes and both channels
maxiter1, maxiter2 = 300, 1500

def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--options', nargs='*', default=options, choices=['conv', 'na', 'leak', 'xyframe', 'regz'])
    p.add_argument('--noll', nargs='*', type=int, default=noll)
    p.add_argument('--n2', type=float, default=n2_beads)
    p.add_argument('--n-beads', type=int, default=n_beads)
    p.add_argument('--frames-around', type=int, default=frames_around)
    p.add_argument('--tag', default='')
    return p.parse_args(argv)
a = parse(sys.argv[1:] if __name__ == '__main__' else [])

#%% model of SGD_jax.py with the calibration of the run
ov, interplane, _ = run_parameters(run_config)
ns, _, _ = S.build_pipeline(dict(ov, loss2='huber'), out_folder + '/sgd_tmp', 'huber', False,
                            S.parse_arguments(['--interplane', str(interplane)]))
if a.n2 is not None: # pupil fields and SAF recomputed with the index of the beads
    sim_args = argparse.Namespace(data_n2=a.n2, data_saf_dz=0.)
    medium = S.data_physics(ns, sim_args).__enter__() # kept for the whole script
    fields = medium.fields
else:
    fields = {k: ns[k] for k in ('xx', 'yy', 'th1', 'phi', 'Ex0', 'Ex1', 'Ex2', 'Ey0', 'Ey1', 'Ey2')}
print('model: planes', ns['d'], 'interplane', ns['interplane'], 'n1', SP.n1, 'n2', SP.n2, 'NA', float(ns['NA']))
dim = int(ns['dim_simu'])
cg = ns['channel_gain']()
sin_th = jnp.sin(jnp.real(fields['th1'])) * SP.n1 # n1 sin(theta) in the pupil, for the effective NA
inside = jnp.sum(jnp.abs(fields['Ex0']) + jnp.abs(fields['Ey0']) + jnp.abs(fields['Ex2']), 0) > 0 # pupil
modes = jnp.array([n - 1 for n in a.noll], dtype=jnp.int32) # index of the Noll modes in zernike_base

#%% bead stacks: reconstruction as reconstruction_stack, patches as SGD_jax.py (photons)
def read_em(f):
    if EM is not None:
        return float(EM)
    import json, re
    meta = Path(f).parent / 'metadata.json'
    m = re.search(r'"Andor1-Gain": *"?([0-9.]+)', meta.read_text()) if meta.exists() else None
    return float(m.group(1)) if m else 200.

def load_beads():
    cal = np.load(calibration_file, allow_pickle=True)
    X, Y, vec, norm = cal['X'], cal['Y'], cal['vecteurs'].tolist(), cal['normalisation']
    M = [cal[f'M_{i}'] for i in range(6)]
    beads, V = [], []
    for f in bead_files:
        conv = sensitivity / (QE * read_em(f))
        raw = extract_raw_ome(f).astype(np.float32)
        base = measure_baseline(raw)
        rec = np.array([[cv2.warpPerspective(ch[i], M[i], ch[i].shape[:2][::-1], flags=cv2.INTER_CUBIC) / norm[i]
                         for i in range(6)] for ch in (subcanals(fr - base, X, Y, vec) for fr in raw)]) * conv
        v = dark_variance(raw) * interpolation_factor(M, rec.shape[2:]) / np.asarray(norm)**2 * conv**2
        V.append(v)
        mp = ndimage.gaussian_filter(rec[:, 1].max(0), 1)
        pk = (mp == ndimage.maximum_filter(mp, 11)) & (mp > np.percentile(mp, 99.5))
        by, bx = np.nonzero(pk)
        m = half + 4
        keep = (bx > m) & (bx < mp.shape[1] - m) & (by > m) & (by < mp.shape[0] - m)
        for b in np.nonzero(keep)[0]:
            if np.min(np.hypot(np.delete(bx, b) - bx[b], np.delete(by, b) - by[b]), initial=99) <= isolation:
                continue
            A = rec[:, 1, by[b]-1:by[b]+2, bx[b]-1:bx[b]+2].sum((1, 2))
            k = int(np.argmax(A))
            ks = np.arange(k - a.frames_around, k + a.frames_around + 1)
            if ks[0] < 0 or ks[-1] >= len(rec):
                continue
            pa = rec[ks][:, :, by[b]-half:by[b]+half+1, bx[b]-half:bx[b]+half+1] # frame, canal, rows, cols
            # SGD_jax.py: plane P, polarisation i = canal (2-P)+3i, rows reversed
            ps = np.stack([np.stack([pa[:, (2-P)+3*i, ::-1] for i in range(2)], 1) for P in range(3)], 1)
            beads.append(dict(data=ps.astype(np.float32), file=f, col=int(bx[b]), row=int(by[b]),
                              dz=(ks - k)*dz_step, peak=float(A[k])))
    V = np.mean(V, 0)
    V = np.array([[V[(2-P)+3*i] for i in range(2)] for P in range(3)])
    beads.sort(key=lambda b: -b['peak'])
    return beads[:a.n_beads], V

#%% model of a bead z-stack
GAIN = 1.9 / QE # EM excess noise (F^2 at EM 10) in photon units
def plane_shift(dz):
    # offsets of registration_z.npz (canal, stage dz, (col, row) nm, relative to canal 1) at the stage z of the frames,
    # in the model: plane P, polarisation i = canal (2-P)+3i, (x, y) = (col, -row), um. (frame, plane, channel)
    reg = np.load(registration_file)
    off = np.array([[np.interp(dz*1000, reg['dz'], reg['offset'][c, :, j]) for j in range(2)] for c in range(6)]) # canal, col/row, frame
    sx = np.stack([np.stack([off[(2-P)+3*i, 0] for i in range(2)], -1) for P in range(3)], 1)/1000
    sy = -np.stack([np.stack([off[(2-P)+3*i, 1] for i in range(2)], -1) for P in range(3)], 1)/1000
    return jnp.array(sx, jnp.float32), jnp.array(sy, jnp.float32)

def psf_stack(p, dz, shift=None):
    n = dz.shape[0]
    mask = jnp.ones_like(sin_th)
    if 'na' in a.options: # soft edge of the pupil at n1 sin(theta) = NA_eff
        mask = jax.nn.sigmoid((p['na'] - sin_th) / 0.01)
    mask = jnp.where(inside, mask, 0.).astype(jnp.complex64)
    mask = jnp.stack([mask]*3)
    zx = jnp.zeros((3, 15)).at[:, modes].set(p['zern'][:, 0]).astype(jnp.complex64)
    zy = jnp.zeros((3, 15)).at[:, modes].set(p['zern'][:, 1]).astype(jnp.complex64)
    F = fields
    M = ns['compute_M_jax'](xp=p['x'] + p['dxy'][:, 0], yp=p['y'] + p['dxy'][:, 1], zp=jnp.full(n, bead_z),
                            d=p['d0'] + p['dz_scale']*dz, x=F['xx'], y=F['yy'], th1=F['th1'], phi=F['phi'],
                            Ex0=F['Ex0'], Ex1=F['Ex1'], Ex2=F['Ex2'], Ey0=F['Ey0'], Ey1=F['Ey1'], Ey2=F['Ey2'],
                            u=ns['u'], v=ns['v'], phase_maskx=mask, phase_masky=mask, zernike_base=ns['zernike_base'],
                            zernike_coefs_x=zx, zernike_coefs_y=zy, second_plane=ns['second_plane'],
                            polar_projections=ns['polar_projections'], lambd=ns['lambd'], f_tube=ns['f_tube'],
                            plane_shift=shift)
    h = ns['PSF_jax'](rho=jnp.zeros(n), eta=jnp.zeros(n), delta=jnp.full(n, 180.), M=M, N_photons=jnp.exp(p['logN']))
    if 'conv' in a.options: # gaussian blur, sigma in pixels, on the whole simulated image
        H = h.shape[-1]
        f = jnp.fft.fftfreq(H)
        otf = jnp.exp(-2*jnp.pi**2*p['sigma']**2*(f[:, None]**2 + f[None, :]**2))
        h = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(h) * otf))
    h = h[:, :, :, dim-half:dim+half+1, dim-half:dim+half+1] * cg[None, :, :, None, None]
    if 'leak' in a.options:
        l = p['leak'][None, :, None, None, None]
        h = (1 - l)*h + l*h[:, :, ::-1]
    return h + p['bg'][:, :, :, None, None]

def chi2(p, data, dz, V, shift=None):
    mu = psf_stack(p, dz, shift)
    var = GAIN*jnp.maximum(mu, 0.) + V[None, :, :, None, None]
    return (data - mu)**2 / var, jnp.log(var), mu

def loss(p, data, dz, V, shift=None):
    c, lv, _ = chi2(p, data, dz, V, shift)
    return jnp.sum(c + lv)

def start_params(data):
    n = data.shape[0]
    bg = np.median(np.concatenate([data[..., 0, :], data[..., -1, :]], -1), -1) # frame, plane, channel
    N = np.maximum(data.sum((1, 2, 3, 4)) - bg.sum((1, 2))*(2*half+1)**2, 1e3)
    return dict(x=jnp.float32(0.), y=jnp.float32(0.), d0=jnp.float32(0.), dz_scale=jnp.float32(1.),
                logN=jnp.array(np.log(N), jnp.float32), bg=jnp.array(bg, jnp.float32),
                dxy=jnp.zeros((n, 2), jnp.float32), sigma=jnp.float32(0.5), na=jnp.float32(float(ns['NA'])), leak=jnp.zeros(3, jnp.float32),
                zern=jnp.zeros((3, 2, len(a.noll)), jnp.float32))

# scales of the parameters for L-BFGS (one unit of the optimiser ~ a meaningful change)
SCALE = dict(dxy=0.01, x=0.02, y=0.02, d0=0.02, dz_scale=0.02, logN=0.05, bg=1., sigma=0.05, na=0.01, leak=0.01, zern=0.05)

def fit(p0, free, data, dz, V, maxiter, shift=None):
    fixed = {k: v for k, v in p0.items() if k not in free}
    scaled0 = {k: p0[k] / SCALE[k] for k in free}
    vec0, unravel = ravel_pytree(scaled0)
    def f(vec):
        q = unravel(vec)
        return loss({**fixed, **{k: q[k]*SCALE[k] for k in free}}, data, dz, V, shift)
    vg = jax.jit(jax.value_and_grad(f))
    def fun(v):
        val, g = vg(jnp.array(v, jnp.float32))
        return float(val), np.asarray(g, np.float64)
    r = minimize(fun, np.asarray(vec0, np.float64), jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
    q = unravel(jnp.array(r.x, jnp.float32))
    return {**fixed, **{k: q[k]*SCALE[k] for k in free}}

def fit_bead(bead, V):
    data, dz = jnp.array(bead['data']), jnp.array(bead['dz'], jnp.float32)
    Vj = jnp.array(V, jnp.float32)
    shift = plane_shift(bead['dz']) if 'regz' in a.options else None
    p = start_params(bead['data'])
    # focus and direction of the stack: grid on d0 for both signs of the stage, model at the start values
    best = None
    for s in (1., -1.):
        for d0 in np.linspace(-1.0, 0.8, 19):
            q = {**p, 'd0': jnp.float32(d0), 'dz_scale': jnp.float32(s)}
            L = float(loss(q, data, dz, Vj, shift))
            if best is None or L < best[0]:
                best = (L, q)
    p = best[1]
    # step 1: position, focus, photons, background
    p = fit(p, ['x', 'y', 'd0', 'dz_scale', 'logN', 'bg'], data, dz, Vj, maxiter1, shift)
    c1, _, _ = chi2(p, data, dz, Vj, shift)
    # step 2: bead model and aberrations
    free = ['x', 'y', 'd0', 'dz_scale', 'logN', 'bg'] + [k for k, o in (('sigma', 'conv'), ('na', 'na'), ('leak', 'leak'), ('dxy', 'xyframe')) if o in a.options]
    if a.noll:
        free.append('zern')
    p = fit(p, free, data, dz, Vj, maxiter2, shift)
    c2, _, mu = chi2(p, data, dz, Vj, shift)
    return p, float(jnp.mean(c1)), float(jnp.mean(c2)), np.array(mu), np.array(jnp.mean(c2, (2, 3, 4)))

#%% main loop
if __name__ == '__main__':
    tag = a.tag or '_'.join(a.options) or 'iso'
    out = Path(out_folder) / tag
    out.mkdir(parents=True, exist_ok=True)
    print('options', a.options, 'noll', a.noll, 'n2', SP.n2, '->', out)
    beads, V = load_beads()
    print(len(beads), 'beads, dark variance per channel (photons^2)', V.round(1).tolist())
    results = []
    for i, bead in enumerate(beads):
        t0 = time.time()
        p, c1, c2, mu, c_frame = fit_bead(bead, V)
        p = {k: np.array(v) for k, v in p.items()}
        zern_ml = p['zern'] * 1000 / (2*np.pi) # m-lambda
        print(f'bead {i} ({Path(bead["file"]).parts[-4]}, col {bead["col"]}, row {bead["row"]}): chi2/pixel '
              f'{c1:.2f} -> {c2:.2f}  d0 {p["d0"]:.3f} dz_scale {p["dz_scale"]:.3f}'
              + (f'  sigma {p["sigma"]:.2f} px' if 'conv' in a.options else '')
              + (f'  NA_eff {p["na"]:.3f}' if 'na' in a.options else '')
              + (f'  leak {np.round(p["leak"], 3).tolist()}' if 'leak' in a.options else '')
              + f'  [{time.time()-t0:.0f} s]', flush=True)
        if a.noll:
            for P in range(3):
                print(f'    plane {P} zernike (m-lambda) Noll {a.noll}: x {zern_ml[P, 0].round(0).tolist()}  y {zern_ml[P, 1].round(0).tolist()}')
        np.savez_compressed(out / f'fit_{i}.npz', data=bead['data'], model=mu, dz=bead['dz'], chi2_frame=c_frame,
                            chi2_step1=c1, chi2_step2=c2, noll=np.array(a.noll), zern_mlambda=zern_ml,
                            col=bead['col'], row=bead['row'], bead_file=bead['file'], **{'p_'+k: v for k, v in p.items()})
        results.append(dict(c1=c1, c2=c2, p=p, zern=zern_ml, data=bead['data'], mu=mu, dz=bead['dz']))
    # summary
    zz = np.array([r['zern'] for r in results])
    lines = [f'options {a.options}  noll {a.noll}  n2 {SP.n2}  {len(results)} beads',
             f'chi2/pixel step 1 (no aberration) median {np.median([r["c1"] for r in results]):.3f}, '
             f'step 2 median {np.median([r["c2"] for r in results]):.3f}']
    if 'conv' in a.options:
        lines.append(f'sigma conv (px): median {np.median([r["p"]["sigma"] for r in results]):.3f}, '
                     f'spread {np.std([r["p"]["sigma"] for r in results]):.3f}')
    if 'na' in a.options:
        lines.append(f'NA_eff: median {np.median([r["p"]["na"] for r in results]):.3f}, spread {np.std([r["p"]["na"] for r in results]):.3f}')
    if 'leak' in a.options:
        lines.append(f'leak per plane: median {np.median([r["p"]["leak"] for r in results], 0).round(3).tolist()}')
    lines.append(f'dz_scale median {np.median([r["p"]["dz_scale"] for r in results]):.3f}')
    if a.noll:
        lines.append('zernike (m-lambda), median over the beads +- spread (std):')
        for P in range(3):
            for c, lab in enumerate('xy'):
                lines.append(f'  plane {P} {lab}: ' + '  '.join(f'Z{n} {np.median(zz[:, P, c, j]):+5.0f}+-{np.std(zz[:, P, c, j]):3.0f}'
                                                             for j, n in enumerate(a.noll)))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    # figure: mean data and residual at the sharpest frame of each plane, all the beads
    fig, ax = plt.subplots(3, 4, figsize=(10, 8))
    for P in range(3):
        for c in range(2):
            dn, rn = [], []
            for r in results:
                k = int(np.argmax(r['data'][:, P, c, half-1:half+2, half-1:half+2].sum((1, 2))))
                bg = np.median(r['mu'][k, P, c][[0, -1]])
                s = (r['mu'][k, P, c] - bg).sum()
                dn.append((r['data'][k, P, c] - bg) / s); rn.append((r['data'][k, P, c] - r['mu'][k, P, c]) / s)
            dn, rn = np.mean(dn, 0), np.mean(rn, 0)
            v = np.abs(rn).max()
            ax[P, 2*c].imshow(dn, cmap='magma'); ax[P, 2*c].set_title(f'plane {P} ch {c} data', fontsize=8)
            ax[P, 2*c+1].imshow(rn, cmap='RdBu_r', vmin=-v, vmax=v)
            ax[P, 2*c+1].set_title(f'data - model ({v/dn.max()*100:.0f}% of peak)', fontsize=8)
    for x_ in ax.ravel():
        x_.set_xticks([]); x_.set_yticks([])
    fig.suptitle(f'{tag}: mean over {len(results)} beads, sharpest frame of each plane')
    fig.tight_layout(); fig.savefig(out / 'residual.png', dpi=110); plt.close(fig)
