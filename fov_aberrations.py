# -*- coding: utf-8 -*-
"""
Field-dependent aberrations (FOV): Zernike coefficients of each PSF from its position in the reconstructed channel
images, with a model fitted on the bead z-stacks (fit_<i>.npz of SGD_aberrations_claude.py).

Model, for each plane, polarisation channel and Noll mode: Z = polynomial of X = (col - 80)/100, Y = (row - 106)/100
(pixels of the reconstructed channel images, the grid of the beads and of the SMLM data reconstructed with the same
calibration), of degree 0 (median of the beads), 1 or 2. The degree of each coefficient is the one that predicts the
beads left out best (leave one physical bead out: all the repeats of a bead in the shifted z-stacks go out together);
a higher degree is kept only if it lowers that error by more than DEGREE_GAIN m-lambda. Outside the area of the beads
the position is clamped to its bounding box (no extrapolation of a quadratic into the corners).

The SMLM pipelines (SGD_jax.py, SGD_jax_CNN.py, parameter fov_aberrations) compute the coefficients of each PSF from
its pre-localised position before the fit and keep them fixed during the fit.

Build a model:  python fov_aberrations.py <folder of the fit_<i>.npz> <output .npz> [--calibration calibration.npz]
"""
import re, glob, itertools, argparse
from pathlib import Path
import numpy as np

CENTER = (80., 106.)   # (col, row) of the centre of the polynomial, px
SCALE = 100.           # px per unit of X and Y
N_MIN = 3e4            # photons: fainter beads have unconstrained coefficients (up to 2000 m-lambda at chi2 ~1)
OUTLIER = 60.          # m-lambda from the median of the beads: left out of the fit of that coefficient
DEGREE_GAIN = 0.5      # m-lambda of leave-one-bead-out error a higher degree must gain to be kept
TERMS = 6              # 1, X, Y, X^2, XY, Y^2


def design(X, Y, degree):
    # columns of the polynomial up to degree, padded with zeros to TERMS columns
    cols = [np.ones_like(X), X, Y, X**2, X*Y, Y**2][:{0: 1, 1: 3, 2: 6}[degree]]
    return np.column_stack(cols + [np.zeros_like(X)]*(TERMS - len(cols)))


def physical_beads(stack, P, match_px=2.5, same_px=3., min_pairs=3):
    # the same bead found again in the z-stacks taken with a small shift of the stage: translation between every pair
    # of stacks (shift giving the most bead matches within match_px), stacks linked by >= min_pairs matches are one
    # field of view, least-squares offset of each stack in it, beads within same_px after the offsets = one bead
    stacks = sorted(set(stack))
    best = {}
    for a, b in itertools.combinations(stacks, 2):
        A, B = P[stack == a], P[stack == b]; bm = (0, None)
        for d in (B[None] - A[:, None]).reshape(-1, 2):
            n = int((np.linalg.norm(A[:, None] + d - B[None], axis=2).min(1) < match_px).sum())
            if n > bm[0]: bm = (n, d)
        best[(a, b)] = bm
    group = {s: s for s in stacks}  # connected stacks (union find)
    def root(s):
        while group[s] != s: s = group[s]
        return s
    for (a, b), (n, d) in best.items():
        if n >= min_pairs: group[root(a)] = root(b)
    bid = -np.ones(len(P), int); nb = 0
    for g in sorted(set(root(s) for s in stacks)):
        stk = [s for s in stacks if root(s) == g]; k = len(stk)
        rows, rhs, w = [], [], []
        for (a, b), (n, d) in best.items():
            if a in stk and b in stk and n >= min_pairs:
                r = np.zeros(k); r[stk.index(b)] = 1; r[stk.index(a)] -= 1; rows.append(r); rhs.append(d); w.append(n)
        off = np.zeros((k, 2))
        if rows:
            w = np.sqrt(np.array(w, float))[:, None]
            off = np.linalg.lstsq(np.vstack([np.array(rows)*w, np.eye(k)[:1]*100]), np.vstack([np.array(rhs)*w, [[0, 0]]]), rcond=None)[0]
        idx = np.where(np.isin(stack, stk))[0]
        Q = P[idx] - off[[stk.index(s) for s in stack[idx]]]
        for j, i in enumerate(idx):
            if bid[i] < 0:
                bid[idx[(np.linalg.norm(Q - Q[j], axis=1) < same_px) & (bid[idx] < 0)]] = nb; nb += 1
    return bid


def build(fit_folder, out_file, calibration=None, n_min=N_MIN):
    rows = []
    for f in glob.glob(str(Path(fit_folder) / 'fit_*.npz')):
        z = np.load(f, allow_pickle=True)
        m = re.search(r'Calib_Polar_\d{4}-\d{2}-\d{2}(?:_(\d+))?', str(z['bead_file']))
        rows.append(dict(stack=int(m.group(1) or 0) if m else 0, col=float(z['col']), row=float(z['row']),
                         zern=np.array(z['zern_mlambda'], float), noll=np.array(z['noll']), N=float(np.exp(np.median(z['p_logN'])))))
    n_all = len(rows); rows = [r for r in rows if r['N'] > n_min]
    noll = rows[0]['noll']
    stack = np.array([r['stack'] for r in rows]); P = np.array([[r['col'], r['row']] for r in rows])
    Z = np.array([r['zern'] for r in rows])  # (n, plane, channel, mode) m-lambda
    bid = physical_beads(stack, P)
    X, Y = (P[:, 0] - CENTER[0])/SCALE, (P[:, 1] - CENTER[1])/SCALE
    coef = np.zeros((3, 2, 15, TERMS)); degree = np.zeros((3, 2, 15), int); loo = np.full((3, 2, 15, 3), np.nan)
    for p in range(3):
        for c in range(2):
            for k, nl in enumerate(noll):
                z = Z[:, p, c, k]; ok = np.abs(z - np.median(z)) < OUTLIER
                for deg in range(3):  # leave one physical bead out
                    err = []
                    for b in np.unique(bid[ok]):
                        tr, te = ok & (bid != b), ok & (bid == b)
                        pred = np.median(z[tr]) if deg == 0 else design(X[te], Y[te], deg) @ np.linalg.lstsq(design(X[tr], Y[tr], deg), z[tr], rcond=None)[0]
                        err.append(z[te] - pred)
                    loo[p, c, nl - 1, deg] = np.sqrt(np.mean(np.concatenate(err)**2))
                d = 0
                for deg in (1, 2):
                    if loo[p, c, nl - 1, deg] < loo[p, c, nl - 1, d] - DEGREE_GAIN: d = deg
                degree[p, c, nl - 1] = d
                cf = np.zeros(TERMS)
                if d == 0: cf[0] = np.median(z[ok])
                else: cf = np.linalg.lstsq(design(X[ok], Y[ok], d), z[ok], rcond=None)[0]
                coef[p, c, nl - 1] = cf*2*np.pi/1000  # m-lambda -> rad
    meta = dict(coef_rad=coef, degree=degree, loo_mlambda=loo, center=np.array(CENTER), scale=SCALE,
                col_range=np.array([P[:, 0].min(), P[:, 0].max()]), row_range=np.array([P[:, 1].min(), P[:, 1].max()]),
                n_fits=len(rows), n_fits_all=n_all, n_beads=int(bid.max() + 1), noll=noll, fit_folder=str(fit_folder))
    if calibration is not None:
        from reconstruction_juliette import calibration_fingerprint
        meta.update(calibration_file=str(calibration), calibration_sha256=calibration_fingerprint(calibration))
    np.savez(out_file, **meta)
    return meta


def load(path):
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def for_reconstruction(value, reconstruction_path):
    # field model for SGD_jax.py / SGD_jax_CNN.py: value None (no field-dependent aberrations) or the path of a model
    # built by build(). Warning if it was measured with another calibration than the one that reconstructed the data
    # (same check as registration_for_reconstruction of reconstruction_juliette.py). Returns the model or None
    if value is None or str(value) in ('None', ''):
        return None
    model = load(str(value))
    data = Path(reconstruction_path).parent
    used = Path(reconstruction_path) / 'calibration_used.txt'
    if used.exists():
        data_sha = used.read_text().split('\n')[1].strip()
    elif (data / 'calibr' / 'calibration.npz').exists():
        from reconstruction_juliette import calibration_fingerprint
        data_sha = calibration_fingerprint(data / 'calibr' / 'calibration.npz')
    else:
        data_sha = None
    sha = str(model['calibration_sha256']) if 'calibration_sha256' in model else None
    if sha is None or sha != data_sha:
        print(f'WARNING fov_aberrations: {value} ' + ('has no calibration fingerprint' if sha is None else 'was built with another calibration')
              + ' than the one of the data: the positions of the PSF may not be those of the beads')
    print(f'Field-dependent aberrations from {value}\n' + describe(model))
    return model


def zernike_at(model, row_px, col_px):
    # Zernike coefficients (rad) of the PSF at (row_px, col_px) of the channel images: (n, 45) for the x channels and
    # (n, 45) for the y channels, index 15*plane + Noll - 1, as zern_x / zern_y of SGD_jax.py
    col = np.clip(np.asarray(col_px, float), *model['col_range'])
    row = np.clip(np.asarray(row_px, float), *model['row_range'])
    X, Y = (col - model['center'][0])/model['scale'], (row - model['center'][1])/model['scale']
    A = design(X, Y, 2)  # (n, 6); the unused terms of a coefficient of lower degree are 0
    z = np.einsum('nt,pcmt->npcm', A, model['coef_rad'])  # (n, plane, channel, mode)
    return z[:, :, 0].reshape(len(A), 45).astype(np.float32), z[:, :, 1].reshape(len(A), 45).astype(np.float32)


def describe(model):
    names = {5: 'astig 45', 6: 'astig 0', 7: 'coma y', 8: 'coma x', 9: 'trefoil y', 10: 'trefoil x', 11: 'spherical'}
    lines = [f"field model of {model['fit_folder']}: {model['n_fits']} fits ({model['n_beads']} physical beads), "
             f"col {model['col_range'][0]:.0f}-{model['col_range'][1]:.0f}, row {model['row_range'][0]:.0f}-{model['row_range'][1]:.0f} px",
             'degree (leave-one-bead-out rms, m-lambda: constant | linear | quadratic) and value at the centre (m-lambda)']
    for p in range(3):
        for c in range(2):
            lines.append(f'  plane {p} {"xy"[c]}: ' + '  '.join(
                f"Z{n} d{model['degree'][p, c, n-1]} {model['coef_rad'][p, c, n-1, 0]*1000/(2*np.pi):+5.1f}"
                for n in model['noll']))
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='builds the field model of the aberrations from the bead fits')
    ap.add_argument('fit_folder'); ap.add_argument('out_file'); ap.add_argument('--calibration', default=None)
    a = ap.parse_args()
    m = build(a.fit_folder, a.out_file, a.calibration)
    print(describe(m)); print('saved', a.out_file)
