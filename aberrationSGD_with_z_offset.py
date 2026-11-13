# -*- coding: utf-8 -*-
"""
2026-11-10 (Claude), side script on purpose (the z registration may be removed from the pipeline soon).
Copy of SGD_aberrations_claude.py (removed in fb633a7, recovered from git) with, in the same fit as the aberrations, a
FREE LATERAL OFFSET of each plane and polarisation channel as a function of the stage z (instead of the fixed
registration_z.npz of option regz). The offsets are those the model of SGD_jax.py needs, i.e. the position of the
channels in the sense of the SMLM fit (not a gaussian / circle / centroid centre of the bead image).

Each isolated bead is fitted on its z-stack (frames_around frames on each side of the focus of the middle plane, focus
of frame k = d0 + dz_scale x stage dz of frame k), in the reconstruction of the data (calibration.npz), with the model
of SGD_jax.py and the calibration of a real run (J, planes, ...) read from its config.txt.
Bead model (options): delta = 180 always; conv (gaussian blur, sigma free); na (soft pupil edge, NA_eff free);
na6 (one NA_eff per plane and channel); leak; xyframe (lateral position of the bead free in each frame: stage drift,
common to all the channels); regz (fixed offsets of registration_z.npz, for comparison); n2 (index of the bead medium);
zernike (Noll modes `noll`, per plane and channel).
OFFSET MODEL (--offset):
    frame: one (x, y) offset per frame for each plane and channel, channel 1 of the reconstruction (model plane 1,
           polarisation 0) = reference, fixed at 0 (the common motion is in xyframe)
    poly2: offset = c0 + c1 dz + c2 dz^2 (dz = stage z, um) per plane and channel, channel 1 = 0
    none : no offset (baseline)
Step 1: position, focus, photons, background (no aberration, no offset). Step 2: everything together (L-BFGS, jax
gradients, Gaussian likelihood with var = gain mu + V).
Outputs in <out>/<tag>/: fit_<i>.npz per bead (p_off_nm: (canal, frame, col/row) nm vs canal 1), summary.txt,
offsets_vs_z.png (median over the beads vs registration_z.npz), residual.png.

python aberrationSGD_with_z_offset.py --offset frame --options xyframe conv na --n2 1.47 --tag ... \
    --calibration <calibration.npz> --beads "<glob of the bead stacks>" --registration <registration_z.npz> \
    --run-config <config.txt> --out <folder> --dz-step 0.07 --frames-around 15
@author: Amaury (copy modified by Claude)
"""
#%%
import os, sys, glob, time, argparse, logging
from pathlib import Path
import numpy as np
import cv2
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
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simu_precision'))  # SGD_precision.py (was SGD_precision_claude.py)
import SGD_precision as S
import simu_PSF_polarMFM_JAX as SP
from density_experimental import run_parameters
from reconstruction_functions import subcanals, measure_baseline, extract_raw_ome, dark_variance, interpolation_factor

# %% PARAMETERS (defaults; the command line overrides them)
calibration_file = '/mnt/d/AMAURY/DATA/2026_11_04_calibration/calibration.npz'
bead_glob = '/mnt/d/AMAURY/DATA/2026_11_04_beads_aberrations/Calib_Polar_2026-11-04*/images/RAW_DATA/image_Pos0.ome.tif'
registration_file = '/mnt/d/AMAURY/DATA/2026_11_04_calibration/registration_z.npz'
run_config = '/mnt/d/AMAURY/DATA/2026_11_04_beads_aberrations/aberrations_beads/model_config.txt'
out_folder = '/mnt/d/AMAURY/DATA/2026_11_04_beads_aberrations/aberrations_beads'
QE = 0.92
sensitivity = 4.4
EM = None
dz_step = 0.07
frames_around = 15
half = 8
n_beads = 1000
isolation = 12
bead_z = 0.05
n2_beads = 1.47
options = ['xyframe', 'conv', 'na']
noll = [5, 6, 7, 8, 9, 10, 11]
maxiter1, maxiter2 = 300, 1500

def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--options', nargs='*', default=options, choices=['conv', 'na', 'na6', 'leak', 'xyframe', 'regz'])
    p.add_argument('--offset', default='frame', choices=['frame', 'poly2', 'none'])
    p.add_argument('--noll', nargs='*', type=int, default=noll)
    p.add_argument('--n2', type=float, default=n2_beads)
    p.add_argument('--n-beads', type=int, default=n_beads)
    p.add_argument('--frames-around', type=int, default=frames_around)
    p.add_argument('--isolation', type=float, default=isolation)
    p.add_argument('--threshold-sigma', type=float, default=8.)
    p.add_argument('--margin', type=int, default=8)
    p.add_argument('--bead-z', type=float, default=bead_z)
    p.add_argument('--model-na', type=float, default=None)
    p.add_argument('--dz-step', type=float, default=dz_step)
    p.add_argument('--calibration', default=calibration_file)
    p.add_argument('--beads', default=bead_glob)
    p.add_argument('--registration', default=registration_file)
    p.add_argument('--run-config', default=run_config)
    p.add_argument('--out', default=out_folder)
    p.add_argument('--tag', default='')
    p.add_argument('--maxiter', nargs=2, type=int, default=[maxiter1, maxiter2], help='L-BFGS iterations of steps 1 and 2')
    return p.parse_args(argv)
a = parse(sys.argv[1:] if __name__ == '__main__' else [])
maxiter1, maxiter2 = a.maxiter
bead_files = sorted(glob.glob(a.beads))
half = max(half, 0)

#%% model of SGD_jax.py with the calibration of the run
ov, interplane, _ = run_parameters(a.run_config)
if a.model_na is not None:
    ov = dict(ov, NA=f'jax.device_put(jnp.array({a.model_na}))')
ns, _, _ = S.build_pipeline(dict(ov, loss2='huber'), a.out + '/sgd_tmp', 'huber', False,
                            S.parse_arguments(['--interplane', str(interplane)]))
if a.n2 is not None:
    sim_args = argparse.Namespace(data_n2=a.n2, data_saf_dz=0.)
    medium = S.data_physics(ns, sim_args).__enter__()
    fields = medium.fields
else:
    fields = {k: ns[k] for k in ('xx', 'yy', 'th1', 'phi', 'Ex0', 'Ex1', 'Ex2', 'Ey0', 'Ey1', 'Ey2')}
print('model: planes', ns['d'], 'interplane', ns['interplane'], 'n1', SP.n1, 'n2', SP.n2, 'NA', float(ns['NA']))
dim = int(ns['dim_simu'])
cg = ns['channel_gain']()
sin_th = jnp.sin(jnp.real(fields['th1'])) * SP.n1
inside = jnp.sum(jnp.abs(fields['Ex0']) + jnp.abs(fields['Ey0']) + jnp.abs(fields['Ex2']), 0) > 0
modes = jnp.array([n - 1 for n in a.noll], dtype=jnp.int32)
REF = jnp.ones((3, 2, 2)).at[1, 0].set(0.)   # (plane, channel, xy): canal 1 = model plane 1, channel 0 = reference
CANAL = [[2 - P + 3*c for c in (0, 1)] for P in range(3)]

#%% bead stacks
def read_em(f):
    if EM is not None:
        return float(EM)
    import re
    meta = Path(f).parent / 'metadata.json'
    m = re.search(r'"Andor1-Gain": *"?([0-9.]+)', meta.read_text()) if meta.exists() else None
    return float(m.group(1)) if m else 200.

def load_beads():
    cal = np.load(a.calibration, allow_pickle=True)
    X, Y, vec, norm = cal['X'], cal['Y'], cal['vecteurs'].tolist(), cal['normalisation']
    M = [cal[f'M_{i}'] for i in range(6)]
    beads, V = [], []
    for f in bead_files:
        conv = sensitivity / (QE * read_em(f))
        raw = extract_raw_ome(f).astype(np.float32)
        base = measure_baseline(raw)
        rec = np.array([[cv2.warpPerspective(ch[i], M[i], ch[i].shape[:2][::-1], flags=cv2.INTER_CUBIC) / norm[i]
                         for i in range(6)] for ch in (subcanals(fr - base, X, Y, vec) for fr in raw)]) * conv
        V.append(dark_variance(raw) * interpolation_factor(M, rec.shape[2:]) / np.asarray(norm)**2 * conv**2)
        mp = ndimage.gaussian_filter(rec[:, 1].max(0), 1)
        if a.threshold_sigma is None:
            thr = np.percentile(mp, 99.5)
        else:
            med = np.median(mp); thr = med + a.threshold_sigma*1.4826*np.median(np.abs(mp - med))
        pk = (mp == ndimage.maximum_filter(mp, 11)) & (mp > thr)
        by, bx = np.nonzero(pk)
        m = max(a.margin, half)
        keep = (bx > m) & (bx < mp.shape[1] - m) & (by > m) & (by < mp.shape[0] - m)
        for b in np.nonzero(keep)[0]:
            if np.min(np.hypot(np.delete(bx, b) - bx[b], np.delete(by, b) - by[b]), initial=99) <= a.isolation:
                continue
            A = rec[:, 1, by[b]-1:by[b]+2, bx[b]-1:bx[b]+2].sum((1, 2))
            k = int(np.argmax(A))
            ks = np.arange(k - a.frames_around, k + a.frames_around + 1)
            if ks[0] < 0 or ks[-1] >= len(rec):
                continue
            pa = rec[ks][:, :, by[b]-half:by[b]+half+1, bx[b]-half:bx[b]+half+1]
            ps = np.stack([np.stack([pa[:, (2-P)+3*i, ::-1] for i in range(2)], 1) for P in range(3)], 1)
            beads.append(dict(data=ps.astype(np.float32), file=f, col=int(bx[b]), row=int(by[b]),
                              dz=(ks - k)*a.dz_step, peak=float(A[k])))
    V = np.mean(V, 0)
    V = np.array([[V[(2-P)+3*i] for i in range(2)] for P in range(3)])
    beads.sort(key=lambda b: -b['peak'])
    return beads[:a.n_beads], V

#%% model
GAIN = 1.9 / QE
def regz_shift(dz):
    reg = np.load(a.registration)
    off = np.array([[np.interp(dz*1000, reg['dz'], reg['offset'][c, :, j]) for j in range(2)] for c in range(6)])
    sx = np.stack([np.stack([off[(2-P)+3*i, 0] for i in range(2)], -1) for P in range(3)], 1)/1000
    sy = -np.stack([np.stack([off[(2-P)+3*i, 1] for i in range(2)], -1) for P in range(3)], 1)/1000
    return jnp.array(sx, jnp.float32), jnp.array(sy, jnp.float32)

def offsets(p, dz):
    """free offsets (frame, plane, channel, xy) in um (model frame: x = col, y = -row), reference channel at 0"""
    if a.offset == 'frame':
        return p['off'] * REF
    if a.offset == 'poly2':
        c = p['offc']
        return (c[0][None] + c[1][None]*dz[:, None, None, None] + c[2][None]*dz[:, None, None, None]**2) * REF
    return jnp.zeros((dz.shape[0], 3, 2, 2))

def psf_stack(p, dz, regz=None):
    n = dz.shape[0]
    mask = jnp.ones_like(sin_th)
    if 'na' in a.options:
        mask = jax.nn.sigmoid((p['na'] - sin_th) / 0.01)
    mask = jnp.where(inside, mask, 0.).astype(jnp.complex64)
    mask = jnp.stack([mask]*3)
    maskx = masky = mask
    if 'na6' in a.options:
        soft = lambda na_: jnp.where(inside, jax.nn.sigmoid((na_[:, None, None] - sin_th) / 0.01), 0.).astype(jnp.complex64)
        maskx, masky = soft(p['na'][:, 0]), soft(p['na'][:, 1])
    zx = jnp.zeros((3, 15)).at[:, modes].set(p['zern'][:, 0]).astype(jnp.complex64)
    zy = jnp.zeros((3, 15)).at[:, modes].set(p['zern'][:, 1]).astype(jnp.complex64)
    o = offsets(p, dz)
    sx, sy = o[..., 0], o[..., 1]
    if regz is not None:
        sx, sy = sx + regz[0], sy + regz[1]
    F = fields
    M = ns['compute_M_jax'](xp=p['x'] + p['dxy'][:, 0], yp=p['y'] + p['dxy'][:, 1], zp=jnp.full(n, a.bead_z),
                            d=p['d0'] + p['dz_scale']*dz, x=F['xx'], y=F['yy'], th1=F['th1'], phi=F['phi'],
                            Ex0=F['Ex0'], Ex1=F['Ex1'], Ex2=F['Ex2'], Ey0=F['Ey0'], Ey1=F['Ey1'], Ey2=F['Ey2'],
                            u=ns['u'], v=ns['v'], phase_maskx=maskx, phase_masky=masky, zernike_base=ns['zernike_base'],
                            zernike_coefs_x=zx, zernike_coefs_y=zy, second_plane=ns['second_plane'],
                            polar_projections=ns['polar_projections'], lambd=ns['lambd'], f_tube=ns['f_tube'],
                            plane_shift=(sx, sy))
    h = ns['PSF_jax'](rho=jnp.zeros(n), eta=jnp.zeros(n), delta=jnp.full(n, 180.), M=M, N_photons=jnp.exp(p['logN']))
    if 'conv' in a.options:
        H = h.shape[-1]
        f = jnp.fft.fftfreq(H)
        otf = jnp.exp(-2*jnp.pi**2*p['sigma']**2*(f[:, None]**2 + f[None, :]**2))
        h = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(h) * otf))
    h = h[:, :, :, dim-half:dim+half+1, dim-half:dim+half+1] * cg[None, :, :, None, None]
    if 'leak' in a.options:
        l = p['leak'][None, :, None, None, None]
        h = (1 - l)*h + l*h[:, :, ::-1]
    return h + p['bg'][:, :, :, None, None]

def chi2(p, data, dz, V, regz=None):
    mu = psf_stack(p, dz, regz)
    var = GAIN*jnp.maximum(mu, 0.) + V[None, :, :, None, None]
    return (data - mu)**2 / var, jnp.log(var), mu

def loss(p, data, dz, V, regz=None):
    c, lv, _ = chi2(p, data, dz, V, regz)
    return jnp.sum(c + lv)

def start_params(data):
    n = data.shape[0]
    bg = np.median(np.concatenate([data[..., 0, :], data[..., -1, :]], -1), -1)
    N = np.maximum(data.sum((1, 2, 3, 4)) - bg.sum((1, 2))*(2*half+1)**2, 1e3)
    return dict(x=jnp.float32(0.), y=jnp.float32(0.), d0=jnp.float32(0.), dz_scale=jnp.float32(1.),
                logN=jnp.array(np.log(N), jnp.float32), bg=jnp.array(bg, jnp.float32),
                dxy=jnp.zeros((n, 2), jnp.float32), sigma=jnp.float32(0.5),
                na=jnp.full((3, 2), float(ns['NA']), jnp.float32) if 'na6' in a.options else jnp.float32(float(ns['NA'])),
                leak=jnp.zeros(3, jnp.float32), zern=jnp.zeros((3, 2, len(a.noll)), jnp.float32),
                off=jnp.zeros((n, 3, 2, 2), jnp.float32), offc=jnp.zeros((3, 3, 2, 2), jnp.float32))

SCALE = dict(dxy=0.01, x=0.02, y=0.02, d0=0.02, dz_scale=0.02, logN=0.05, bg=1., sigma=0.05, na=0.01, leak=0.01,
             zern=0.05, off=0.005, offc=0.005)

def fit(p0, free, data, dz, V, maxiter, regz=None):
    fixed = {k: v for k, v in p0.items() if k not in free}
    scaled0 = {k: p0[k] / SCALE[k] for k in free}
    vec0, unravel = ravel_pytree(scaled0)
    def f(vec):
        q = unravel(vec)
        return loss({**fixed, **{k: q[k]*SCALE[k] for k in free}}, data, dz, V, regz)
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
    regz = regz_shift(bead['dz']) if 'regz' in a.options else None
    p = start_params(bead['data'])
    best = None
    for s in (1., -1.):
        for d0 in np.linspace(-1.0, 0.8, 19):
            q = {**p, 'd0': jnp.float32(d0), 'dz_scale': jnp.float32(s)}
            L = float(loss(q, data, dz, Vj, regz))
            if best is None or L < best[0]:
                best = (L, q)
    p = best[1]
    p = fit(p, ['x', 'y', 'd0', 'dz_scale', 'logN', 'bg'], data, dz, Vj, maxiter1, regz)
    c1, _, _ = chi2(p, data, dz, Vj, regz)
    free = ['x', 'y', 'd0', 'dz_scale', 'logN', 'bg'] + [k for k, o in (('sigma', 'conv'), ('na', 'na'), ('leak', 'leak'), ('dxy', 'xyframe')) if o in a.options]
    if 'na6' in a.options and 'na' not in free:
        free.append('na')
    if a.noll:
        free.append('zern')
    if a.offset == 'frame':
        free.append('off')
    elif a.offset == 'poly2':
        free.append('offc')
    p = fit(p, free, data, dz, Vj, maxiter2, regz)
    c2, _, mu = chi2(p, data, dz, Vj, regz)
    o = np.array(offsets(p, dz))                                     # (frame, plane, channel, xy) um
    off_nm = np.zeros((6, len(bead['dz']), 2))
    for P in range(3):
        for c in (0, 1):
            off_nm[CANAL[P][c], :, 0] = o[:, P, c, 0]*1000           # column
            off_nm[CANAL[P][c], :, 1] = -o[:, P, c, 1]*1000          # row
    return p, float(jnp.mean(c1)), float(jnp.mean(c2)), np.array(mu), np.array(jnp.mean(c2, (2, 3, 4))), off_nm

#%% main
if __name__ == '__main__':
    tag = a.tag or ('offset_' + a.offset + '_' + '_'.join(a.options))
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    print('options', a.options, 'offset', a.offset, 'noll', a.noll, 'n2', SP.n2, '->', out, flush=True)
    beads, V = load_beads()
    print(len(beads), 'beads, dark variance per channel (photons^2)', V.round(1).tolist(), flush=True)
    results = []
    for i, bead in enumerate(beads):
        t0 = time.time()
        p, c1, c2, mu, c_frame, off_nm = fit_bead(bead, V)
        p = {k: np.array(v) for k, v in p.items()}
        zern_ml = p['zern'] * 1000 / (2*np.pi)
        k0 = int(np.argmin(np.abs(bead['dz'])))
        print(f'bead {i} ({Path(bead["file"]).parts[-4]}, col {bead["col"]}, row {bead["row"]}): chi2/pixel '
              f'{c1:.2f} -> {c2:.2f}  d0 {p["d0"]:.3f} dz_scale {p["dz_scale"]:.3f}'
              + (f'  sigma {p["sigma"]:.2f} px' if 'conv' in a.options else '')
              + (f'  NA_eff {p["na"]:.3f}' if 'na' in a.options else '')
              + (f'  offsets at dz 0 (col,row nm) ' + ' '.join(f'c{c}({off_nm[c, k0, 0]:+.0f},{off_nm[c, k0, 1]:+.0f})' for c in (0, 3, 4, 2, 5)) if a.offset != 'none' else '')
              + f'  [{time.time()-t0:.0f} s]', flush=True)
        np.savez_compressed(out / f'fit_{i}.npz', data=bead['data'], model=mu, dz=bead['dz'], chi2_frame=c_frame,
                            chi2_step1=c1, chi2_step2=c2, noll=np.array(a.noll), zern_mlambda=zern_ml, p_off_nm=off_nm,
                            col=bead['col'], row=bead['row'], bead_file=bead['file'], **{'p_'+k: v for k, v in p.items()})
        results.append(dict(c1=c1, c2=c2, p=p, zern=zern_ml, data=bead['data'], mu=mu, dz=bead['dz'], off=off_nm))
    # summary
    zz = np.array([r['zern'] for r in results])
    lines = [f'options {a.options}  offset {a.offset}  noll {a.noll}  n2 {SP.n2}  {len(results)} beads  (beads {a.beads}, '
             f'calibration {a.calibration})',
             f'chi2/pixel step 1 (no aberration, no offset) median {np.median([r["c1"] for r in results]):.4f}, '
             f'step 2 median {np.median([r["c2"] for r in results]):.4f}, mean {np.mean([r["c2"] for r in results]):.4f}']
    if 'conv' in a.options:
        lines.append(f'sigma conv (px): median {np.median([r["p"]["sigma"] for r in results]):.3f}')
    if 'na' in a.options:
        lines.append(f'NA_eff: median {np.median([r["p"]["na"] for r in results]):.3f}')
    lines.append(f'dz_scale median {np.median([r["p"]["dz_scale"] for r in results]):.3f}')
    if a.noll:
        lines.append('zernike (m-lambda), median over the beads +- spread (std):')
        for P in range(3):
            for c, lab in enumerate('xy'):
                lines.append(f'  plane {P} {lab}: ' + '  '.join(f'Z{n} {np.median(zz[:, P, c, j]):+5.0f}+-{np.std(zz[:, P, c, j]):3.0f}'
                                                             for j, n in enumerate(a.noll)))
    reg = np.load(a.registration)
    dzs = results[0]['dz']
    if a.offset != 'none':
        OFF = np.array([r['off'] for r in results])                 # (bead, canal, frame, col/row) nm
        lines.append('free offsets vs canal 1 (nm, col / row), median over the beads [p25, p75], and registration_z.npz (gaussian):')
        for c in (0, 3, 4, 2, 5):
            for zi in range(0, len(dzs), 3):
                v = OFF[:, c, zi]
                lines.append(f'  canal {c} dz {dzs[zi]*1000:+5.0f}: col {np.median(v[:, 0]):+6.1f} [{np.percentile(v[:, 0], 25):+6.1f},{np.percentile(v[:, 0], 75):+6.1f}]'
                             f'  row {np.median(v[:, 1]):+6.1f} [{np.percentile(v[:, 1], 25):+6.1f},{np.percentile(v[:, 1], 75):+6.1f}]'
                             f'  | regz col {np.interp(dzs[zi]*1000, reg["dz"], reg["offset"][c, :, 0]):+6.1f} row {np.interp(dzs[zi]*1000, reg["dz"], reg["offset"][c, :, 1]):+6.1f}')
        fig, ax = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
        for j, c in enumerate((0, 3, 4, 2, 5)):
            for xy in range(2):
                A_ = ax[xy, j]
                med = np.median(OFF[:, c, :, xy], 0)
                A_.fill_between(dzs*1000, np.percentile(OFF[:, c, :, xy], 25, 0), np.percentile(OFF[:, c, :, xy], 75, 0), alpha=.25)
                A_.plot(dzs*1000, med, 'o-', ms=3, label='model fit (median, p25-p75)')
                A_.plot(reg['dz'], reg['offset'][c, :, xy], 'k--', label='registration_z.npz (gaussian)')
                A_.axhline(0, color='0.5', lw=.8); A_.set_ylim(-60, 60); A_.grid(alpha=.3)
                A_.set_title(f'canal {c} (plane {"0 0 1 1 2 2".split()[[0,3,1,4,2,5].index(c)]}): {"col (x)" if xy == 0 else "row (y)"} vs canal 1', fontsize=10)
                if xy == 1: A_.set_xlabel('stage dz vs middle-plane focus (nm)')
                if j == 0: A_.set_ylabel('offset (nm)'); A_.legend(fontsize=8)
        fig.suptitle(f'{tag}: lateral offsets of the channels fitted with the aberrations (model of SGD_jax.py), {len(results)} beads', fontsize=11)
        fig.tight_layout(); fig.savefig(out / 'offsets_vs_z.png', dpi=90); plt.close(fig)
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
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
