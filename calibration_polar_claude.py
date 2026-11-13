#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration of J1 / J2 (Jones matrices of the paths behind the 2 beam splitters, planes 0 and 2 = J1, plane 1 = J2)
from the two existing measurements, in one model (2026-11-02, Claude; replaces Calibration_polar2.ipynb and
polar_analysis_6channels.py, see suivi 2026_11_02\\polarization_calibration):

A (Calibration_polar2.ipynb data): linear polarizer between the lamp and the objective rotated by 5 deg steps
  (positive from x to y, first image along y), full MFM image, 6 channels (plane 0, 1, 2 x Wollaston top = x,
  bottom = y, as in reconstruction_functions.py). Channels cut as subcanals() and divided by the lamp image of the
  session, as the reconstruction does: the model of the SGD sees the same normalised channels.
B (polar_analysis_6channels.py data): polarizer at X / Y / 45, quarter wave plate before the Wollaston rotated to
  9 angles, both Wollaston outputs (discs = images of a field diaphragm).

Model (A): channel c of image i (polarizer at theta_i), lamp-normalised: I_ci = k_i (u_c + v_c cos 2theta_i +
w_c sin 2theta_i); k_i = common factor of the image (lamp drift, ~3 % polarisation of the lamp light before the
polarizer). The lamp image itself is that partially polarised light: the lamp normalisation of the reconstruction
gives the planes a different x/y balance (plane 1: y +5-6 %), included in (u, v, w) as in the reconstructed data.

What the data fix:
- the zero of the rotating polarizer is unknown in every session (J rot(t) with the zero moved by -t fits the same):
  the common rotation of J1 and J2 is not measured and stays in rho_offset (beads); the relative one is measured.
- A (linear inputs, intensities) sees per channel u + v cos 2theta + w sin 2theta: a norm, a visibility V and an
  angle psi. That is all the model needs for a linearly polarised field (a dipole, below the critical angle), with
  the lamp normalisation of the reconstruction and any depolarisation included: J is built from it
  (J_from_response). The response to circular light (sign of Im of the rows) is measured by neither method.
- B (plate before the Wollaston) gives the polarisation state behind J for the X / Y / 45 inputs: DOP 0.89-0.99,
  input dependent, so no Jones matrix fits B exactly; used as a check of A (other session).

Usage: python calibration_polar_claude.py [out_dir]   (D: = analysis PC; ~2 min, mostly reading the images)
"""
import sys, os, glob, json, warnings, logging
import numpy as np
import tifffile
from scipy.optimize import least_squares
from scipy import ndimage
warnings.filterwarnings('ignore')
logging.getLogger('tifffile').setLevel(logging.ERROR)

DATA = '/mnt/d/Amaury/DATA/'
A_POLAR = DATA + '2026_04_30_calibration/polar/Calib_Polar_2026-05-04_{}/images/RAW_DATA/image_Pos0.ome.tif'
A_N = 37                                    # _0 .. _36: 90 .. 270 deg
A_LAMP = DATA + '2026_04_30_calibration/intensity/Calib_Polar_2026-04-30/images/RAW_DATA/image_Pos0.ome.tif'
A_CAL = DATA + '2026_04_30_calibration/calibration.npz'
B_DIR = DATA + '2026_08_24_calibr_polar_final_without_polarizer/'
B_INPUTS = {'X': 0., 'Y': 90., '45': 45.}  # nominal polarizer angles (deg)
B_PLATE = [0, 40, 80, 120, 160, 200, 240, 280, 320]  # scale of the plate (deg); it turns CLOCKWISE, fast axis along x at 0: angle = -scale
LAMBDA = 620                                 # nm, start value of the plate retardance (datasheet table); fitted

# pipeline J (SGD_jax.py, from polar_analysis_6channels.py, 2026-08-26), for the comparison
J1_PIPE = np.array([[0.89097912+0.j, -0.25009759-0.43946158j], [0.43033225-0.14481147j, 0.6335862+0.58557087j]])
J2_PIPE = np.array([[0.31676515+0.j, -0.93297585-0.07330244j], [-0.58381747+0.74754063j, -0.26412184+0.23328625j]])

def plate_retardance(lam):
    # quarter wave plate retardance (rad) from the datasheet table of polar_analysis_6channels.py
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polar_analysis_6channels.py')).read()
    ns = {}
    exec(src.split('# %% Calcul de phi')[1].split('#%% functions')[0], ns)
    return ns['phi_list'][ns['lbda_list'].index(lam)]*2*np.pi

# ---------------------------------------------------------------- images
def read(path):
    return tifffile.imread(path).astype(np.float64).reshape(-1, 512, 512)

def baseline(frames):
    # camera baseline of a stack (same as measure_baseline of reconstruction_functions.py): median of the darkest rows
    rows = np.median(np.median(frames, axis=0), axis=1)
    dark = rows < rows.min() + 0.2*(np.median(rows) - rows.min())
    return float(np.median(frames[:, dark]))

def channel_slices(cal, margin=10):
    # the 6 channels as cut by subcanals() (reconstruction_functions.py), without a margin at the edges
    X, Y, v = cal['X'], cal['Y'], cal['vecteurs']
    return [(slice(Y[0] + v[i][1] + margin, Y[1] + v[i][1] - margin), slice(X[2] + v[i][0] + margin, X[3] + v[i][0] - margin))
            for i in range(6)]

ORDER = [0, 3, 1, 4, 2, 5]   # channels of subcanals -> (plane 0 x, plane 0 y, plane 1 x, plane 1 y, plane 2 x, plane 2 y)

def load_A(tile=None):
    """
    method A: I (37, 6) = mean of each channel / mean of the lamp in the same channel, sem from the frames;
    with tile = (ny, nx): also the same per tile, I_t (37, 6, ny, nx), lamp-normalised per tile, and the lamp tiles
    """
    cal = np.load(A_CAL)
    sl = [channel_slices(cal)[i] for i in ORDER]
    lamp = read(A_LAMP); lamp = lamp.mean(0) - baseline(lamp)
    L = np.array([lamp[s].mean() for s in sl])
    I, E, It = [], [], []
    for k in range(A_N):
        fr = read(A_POLAR.format(k)); fr -= baseline(fr)
        m = np.array([[f[s].mean() for s in sl] for f in fr])        # (frames, 6)
        I.append(m.mean(0)/L); E.append(m.std(0, ddof=1)/np.sqrt(len(fr))/L)
        if tile is not None:
            mf = fr.mean(0); It.append([tiles(mf[s], tile) for s in sl])
    out = dict(theta=np.radians(90. + 5.*np.arange(A_N)), I=np.array(I), E=np.array(E), L=L)
    if tile is not None:
        Lt = np.array([tiles(lamp[s], tile) for s in sl])
        out.update(It=np.array(It)/Lt[None], Lt=Lt, Lt_rel=Lt/L[:, None, None], tile=tile, shape=lamp[sl[0]].shape)
    return out

def tiles(a, tile):
    ny, nx = tile; h, w = a.shape; a = a[:h//ny*ny, :w//nx*nx]
    return a.reshape(ny, h//ny, nx, w//nx).mean((1, 3))

def load_B(erode=6):
    """
    method B: I (3 inputs, 9 plate angles, 6 discs) above the background, sem from the frames. Discs found on the
    mean of all the images, cut to the channel bounds of the 04-30 calibration (the diaphragm image is clipped by
    the channel edges), background = median of the rows between the 2 rows of discs in the same columns
    """
    cal = np.load(A_CAL); sl = [channel_slices(cal, margin=3)[i] for i in ORDER]
    stacks = {n: [read(f'{B_DIR}{n}/{q}/images/RAW_DATA/image_Pos0.ome.tif') for q in B_PLATE] for n in B_INPUTS}
    mean = np.mean([s.mean(0) for v in stacks.values() for s in v], 0)
    masks, bgs = [], []
    for s in sl:
        sub = np.zeros_like(mean, bool); sub[s] = True
        cols = s[1]; between = np.zeros_like(sub); between[205:245, cols] = True
        b = np.median(mean[between]); peak = np.percentile(mean[sub], 99)
        m = sub & (mean > b + 0.6*(peak - b))
        lab, n = ndimage.label(m); big = np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1
        masks.append(ndimage.binary_erosion(lab == big, iterations=erode)); bgs.append(between)
    I = np.zeros((3, 9, 6)); E = np.zeros((3, 9, 6))
    for a, n in enumerate(B_INPUTS):
        for q, fr in enumerate(stacks[n]):
            for c, (m, bm) in enumerate(zip(masks, bgs)):
                v = fr[:, m].mean(1) - np.median(fr[:, bm], axis=1)
                I[a, q, c] = v.mean(); E[a, q, c] = v.std(ddof=1)/np.sqrt(len(v))
    return dict(I=I, E=E, npix=[int(m.sum()) for m in masks], masks=masks, plate=-np.radians(B_PLATE),
                inputs=np.radians(list(B_INPUTS.values())))

# ---------------------------------------------------------------- optics
def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def jones(p):
    # 7 parameters: [[a, b e^{i pb}], [c e^{i pc}, d e^{i pd}]] (global phase: J[0, 0] real)
    a, b, pb, c, pc, d, pd = p
    return np.array([[a, b*np.exp(1j*pb)], [c*np.exp(1j*pc), d*np.exp(1j*pd)]])

def jones_params(J):
    J = J*np.exp(-1j*np.angle(J[0, 0]))
    return np.array([abs(J[0, 0]), abs(J[0, 1]), np.angle(J[0, 1]), abs(J[1, 0]), np.angle(J[1, 0]), abs(J[1, 1]), np.angle(J[1, 1])])

def coherency(theta, P):
    # (n, 2, 2): partially polarised light, linear part along theta, trace 1
    e = np.stack([np.cos(theta), np.sin(theta)], -1)
    return (1 - P)/2*np.eye(2)[None] + P*e[:, :, None]*e[:, None, :]

def plate(w, delta):
    return rot(w) @ np.diag([1, np.exp(1j*delta)]) @ rot(-w)

def channel_intensity(M, G):
    # diag of M G M^H for a stack of G: (n, 2)
    return np.real(np.einsum('ij,njk,ik->ni', M, G, M.conj()))

def linear_response(J):
    # per channel (row r of J): norm |r|^2, visibility V and angle psi of the response to linear light,
    # I(theta) = norm (1 + V cos 2(theta - psi))/2, and the circular part Im(r_x* r_y) (seen by elliptical light only)
    out = []
    for r in J:
        a, b, c = abs(r[0])**2, abs(r[1])**2, r[0].conj()*r[1]
        out.append((a + b, np.hypot(a - b, 2*c.real)/(a + b), 0.5*np.arctan2(2*c.real, a - b), 2*c.imag/(a + b)))
    return np.array(out)

# ---------------------------------------------------------------- method A alone: linear response of each channel
def fit_A(A, P=1.):
    """
    I_ci = k_i (u_c + v_c cos 2 theta_i + w_c sin 2 theta_i): linear response of each channel (3 numbers, all that
    linear light can measure), k_i common to the 6 channels of image i (k_0 = 1). Alternating least squares
    """
    th, I, E = A['theta'], A['I'], A['E']
    B_ = np.c_[np.ones_like(th), np.cos(2*th), np.sin(2*th)]
    k = np.ones(len(th))
    for _ in range(200):
        coef = np.array([np.linalg.lstsq(B_*k[:, None]/E[:, c:c+1], I[:, c]/E[:, c], rcond=None)[0] for c in range(6)]).T  # (3, 6)
        pred = B_ @ coef
        k_new = np.sum(pred*I/E**2, 1)/np.sum(pred**2/E**2, 1); k_new /= k_new[0]
        if np.max(abs(k_new - k)) < 1e-10: break
        k = k_new
    model = k[:, None]*(B_ @ coef)
    u, v, w = coef
    resp = np.c_[2*u, np.hypot(v, w)/u, 0.5*np.arctan2(w, v)]    # norm, V, psi (rad)
    return dict(k=k, coef=coef, resp=resp, model=model, res=(I - model)/E, chi2=np.mean(((I - model)/E)**2))

# ---------------------------------------------------------------- method B: polarisation state behind J
def row1(w, d):
    # first row (S1 out) of the Mueller matrix of the plate (fast axis w, retardance d), Stokes S1, S2, S3 in: (n, 3)
    c, s = np.cos(2*w), np.sin(2*w)
    return np.stack([c*c + s*s*np.cos(d), c*s*(1 - np.cos(d)), -s*np.sin(d)], -1)

def stokes_B(B):
    """
    model-free: one normalised Stokes vector s (3, DOP = |s| <= 1 not imposed) per input and plane, measured through
    (I_x - I_y/g)/(I_x + I_y/g) = row1(plate) . s (independent of the lamp drift), with the plate offset w0, its
    retardance, the y gain g of each plane and an error of each of the 9 plate positions (the same for the 3 inputs:
    the hand-set positions drift by ~+-1 deg) fitted. A Jones matrix per plane does not fit (DOP < 1, input dependent)
    """
    I, E, om = B['I'], B['E'], B['plate']
    def unpack(x): return x[0], x[1], x[2:5], x[5:32].reshape(3, 3, 3), x[32:41] - x[32:41].mean()
    def obs(g):
        Ix, Iy = I[..., 0::2], I[..., 1::2]/g
        return (Ix - Iy)/(Ix + Iy), 2*np.sqrt(E[..., 0::2]**2 + (E[..., 1::2]/g)**2)/(Ix + Iy)
    def pred(x):
        w0, d, g, s, e = unpack(x)
        return np.stack([np.einsum('qk,pk->qp', row1(om + w0 + e, d), s[a]) for a in range(3)])
    def resid(x):
        o, oe = obs(x[2:5]); return ((pred(x) - o)/oe).ravel()
    best = None
    for w0 in np.radians(np.arange(-45, 45, 9)):
        r = least_squares(resid, np.r_[w0, np.pi/2, 1., 1., 1., np.zeros(27), np.zeros(9)],
                          bounds=(np.r_[-np.pi, 0.5, .5, .5, .5, -np.ones(27)*1.5, -np.ones(9)*.3], np.r_[np.pi, 2.6, 2, 2, 2, np.ones(27)*1.5, np.ones(9)*.3]))
        if best is None or r.cost < best.cost: best = r
    w0, d, g, s, e = unpack(best.x)
    # linear response of each channel (analyser x / y without the plate), in the scale of the B polarizer
    S0 = (I[..., 0::2] + I[..., 1::2]/g).mean(1)       # (input, plane)
    resp = np.zeros((6, 3))
    for p in range(3):
        for ch, sign in ((0, 1), (1, -1)):
            Ic = S0[:, p]*(1 + sign*s[:, p, 0])/2       # inputs X, Y, 45
            u, v, w = (Ic[0] + Ic[1])/2, (Ic[0] - Ic[1])/2, Ic[2] - (Ic[0] + Ic[1])/2
            resp[2*p + ch] = (2*u, np.hypot(v, w)/u, 0.5*np.arctan2(w, v))
    return dict(w0=w0, delta=d, g=g, s=s, err=e, chi2=2*best.cost/len(best.fun), resp=resp)

# ---------------------------------------------------------------- J from the A response
def J_from_response(coef, hand):
    """
    Jones rows that give exactly the measured linear response of each channel (u + v cos 2theta + w sin 2theta,
    lamp-normalised): |r_1|^2 = u + v, |r_2|^2 = u - v, Re(r_1* r_2) = w, Im(r_1* r_2) = +-sqrt(u^2 - v^2 - w^2).
    The sign (response to circular light) is not measured (linear inputs only): hand = (sign of the x row, sign of
    the y row), the same for the 3 planes. Each plane scaled to a mean row norm of 1: only its x / y balance is kept,
    the ~2 % between the planes (polar images vs lamp image) is not polarisation. Returns J_dichroic (3, 2, 2) in the
    frame of the A polarizer scale and in the plane order of the MODEL (plane 0 of the model = plane 2 of the
    reconstruction, see SGD_jax.py)
    """
    J = np.zeros((3, 2, 2), complex)
    for p in range(3):
        for ch in range(2):
            u, v, w = coef[:, 2*p + ch]
            im = np.sqrt(max(u*u - v*v - w*w, 0.))*hand[ch]
            r1 = np.sqrt(u + v); J[p, ch] = (r1, (w + 1j*im)/r1)
        J[p] /= np.sqrt(np.mean(np.sum(abs(J[p])**2, 1)))
    return J[::-1]

def expression(J):
    # J_dichroic as a python expression for SGD_batch.py
    row = lambda r: '[' + ', '.join(f'{z.real:.6f}{z.imag:+.6f}j' for z in r) + ']'
    return 'np.array([' + ', '.join('[' + ', '.join(row(r) for r in M) + ']' for M in J) + '])'

# ---------------------------------------------------------------- field dependence (A, tiles)
BEADS_SLB = {'central': (8800, -15400), 'top': (10100, -4300)}    # nm, (x, y) of the fit = (col, -row) x 120 nm

def field_maps(A, fa):
    """
    linear response of each channel per tile (lamp-normalised per tile), with the image factors k of the global fit.
    Tiles are at the same field position in the 6 channels (channels cut with the shifts of subcanals). Returns the
    angle psi and visibility V per channel and tile, the common rotation (mean of the 6 psi minus their median) and
    the relative rotation J2 - J1 (plane 1 minus planes 0 and 2)
    """
    th, k, It = A['theta'], fa['k'], A['It']
    Bm = np.c_[np.ones_like(th), np.cos(2*th), np.sin(2*th)]*k[:, None]
    coef = np.einsum('ij,jcyx->icyx', np.linalg.pinv(Bm), It)
    res = It - np.einsum('ji,icyx->jcyx', Bm, coef)
    u, v, w = coef
    V, psi = np.hypot(v, w)/u, 0.5*np.degrees(np.arctan2(w, v))
    sig = np.degrees(res.std(0)/(u*V*np.sqrt(len(th)/2)))
    dev = (psi - np.median(psi, (1, 2), keepdims=True) + 90) % 180 - 90
    rel = (dev[2] + dev[3])/2 - (dev[0] + dev[1] + dev[4] + dev[5])/4
    return dict(coef=coef, psi=psi, V=V, sig=sig, common=dev.mean(0), rel=rel - np.median(rel), Lrel=A['Lt_rel'])

def at_position(m, x, y, shape, margin=10):
    # value of a tile map at the fit position (x, y) nm (bilinear between tile centres)
    ny, nx = m.shape; th, tw = (shape[0] - 2*margin)/ny, (shape[1] - 2*margin)/nx
    row, col = -y/120, x/120
    return float(ndimage.map_coordinates(m, [[(row - margin)/th - 0.5], [(col - margin)/tw - 0.5]], order=1, mode='nearest')[0])

def plot_field(fm, shape, path, margin=10):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ext = [margin, shape[1] - margin, shape[0] - margin, margin]
    panels = [('common rotation (deg)', fm['common'], 'RdBu_r', True), ('relative rotation J2 - J1 (deg)', fm['rel'], 'RdBu_r', True),
              ('visibility J1 (plane 2 of the model = reconstruction plane 0, x row)', fm['V'][0], 'viridis', False),
              ('visibility J2 (plane 1, x row)', fm['V'][2], 'viridis', False)]
    fig, ax = plt.subplots(1, 4, figsize=(18, 5.5), constrained_layout=True)
    for a, (title, m, cmap, sym) in zip(ax, panels):
        lim = np.max(abs(m)) if sym else None
        im = a.imshow(m, extent=ext, cmap=cmap, vmin=-lim if sym else None, vmax=lim if sym else None)
        for n, (x, y) in BEADS_SLB.items():
            a.plot(x/120, -y/120, 'o', mfc='none', mec='k', ms=10); a.annotate(n, (x/120 + 6, -y/120), fontsize=9)
        a.set_title(title, fontsize=10); a.set_xlabel('column (px, x = col x 120 nm)'); a.set_ylabel('row (px, y = -row x 120 nm)')
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle('Field dependence of the channel response, method A (2026-05-04), tiles of 19.5 x 17.5 px; circles: SLB beads')
    fig.savefig(path, dpi=110); plt.close(fig)

# ---------------------------------------------------------------- report
def describe(J, name):
    # retardance and diattenuation of J (polar decomposition J = U H), linear response of the rows
    W, s, Vh = np.linalg.svd(J); U = W @ Vh
    D = (s[0]**2 - s[1]**2)/(s[0]**2 + s[1]**2)
    lr = linear_response(J/np.sqrt(np.sum(abs(J)**2)/2))
    ev = np.linalg.eigvals(U); ret = abs(np.angle(ev[0]/ev[1]))
    return (f'{name}: diattenuation {D:.3f} (s1^2/s2^2 = {s[0]**2/s[1]**2:.3f}), retardance {np.degrees(ret):.1f} deg; '
            f'rows x / y: norm {lr[0, 0]:.3f} / {lr[1, 0]:.3f}, V {lr[0, 1]:.3f} / {lr[1, 1]:.3f}, '
            f'psi {np.degrees(lr[0, 2]):+.1f} / {np.degrees(lr[1, 2]):+.1f} deg, circular {lr[0, 3]:+.3f} / {lr[1, 3]:+.3f}')

def main(out):
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, 'calibration_polar_claude.txt'), 'w')
    def p(*a):
        s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()
    A = load_A(tile=(10, 8)); B = load_B()
    np.savez(os.path.join(out, 'data_AB.npz'), **{f'A_{k}': v for k, v in A.items() if k != 'shape'},
             **{f'B_{k}': v for k, v in B.items() if k != 'masks'})
    p(f'A: {A["I"].shape[0]} polarizer angles, sem/I median {np.median(A["E"]/A["I"]):.4f}; lamp channel means {np.round(A["L"], 1)}')
    p(f'B: discs {B["npix"]} px, sem/I median {np.median(B["E"]/B["I"]):.4f}')
    # 1. A alone
    fa = fit_A(A)
    p('\n1. METHOD A ALONE: linear response per channel (I = norm (1 + V cos 2(theta - psi))/2, theta of the polarizer scale)')
    p(f'   chi2/n {fa["chi2"]:.2f}; k (image factor) range {fa["k"].min():.4f}-{fa["k"].max():.4f}, 2theta modulation '
      f'{np.hypot(*np.linalg.lstsq(np.c_[np.ones(A_N), np.cos(2*A["theta"]), np.sin(2*A["theta"])], fa["k"], rcond=None)[0][1:])/fa["k"].mean():.4f}')
    for c, nm in enumerate(['p0 x', 'p0 y', 'p1 x', 'p1 y', 'p2 x', 'p2 y']):
        n_, V, psi = fa['resp'][c]; p(f'   {nm}: norm {n_:.4f}  V {V:.4f}  psi {np.degrees(psi):+7.2f} deg')
    r = fa['resp']
    for pl in range(3):
        p(f'   plane {pl}: y/x norm {r[2*pl+1, 0]/r[2*pl, 0]:.4f}, psi_y - psi_x {np.degrees(r[2*pl+1, 2] - r[2*pl, 2]) % 180:.2f} deg')
    p(f'   relative angle plane 1 - plane 0 (psi_x) {np.degrees(r[2, 2] - r[0, 2]):+.2f}, plane 2 - plane 0 {np.degrees(r[4, 2] - r[0, 2]):+.2f} deg')
    # 2. B: polarisation state behind J, channel response, comparison with A
    sb = stokes_B(B)
    p(f'\n2. METHOD B: model-free Stokes vectors behind J, chi2/n {sb["chi2"]:.1f}; plate offset {np.degrees(sb["w0"]):+.2f} deg, '
      f'retardance {np.degrees(sb["delta"]):.1f} deg (table at {LAMBDA} nm {np.degrees(plate_retardance(LAMBDA)):.1f}), '
      f'y gains {np.round(sb["g"], 4)}, plate position errors {np.round(np.degrees(sb["err"]), 2)} deg')
    for a_, nm in enumerate(B_INPUTS):
        p('   ' + ' | '.join(f'{nm:2s} plane {q}: DOP {np.linalg.norm(sb["s"][a_, q]):.3f} alpha {0.5*np.degrees(np.arctan2(sb["s"][a_, q, 1], sb["s"][a_, q, 0])):+7.2f} '
                          f'eps {0.5*np.degrees(np.arcsin(np.clip(sb["s"][a_, q, 2]/np.linalg.norm(sb["s"][a_, q]), -1, 1))):+6.2f}' for q in range(3)))
    rb, ra = sb['resp'], fa['resp']
    p('   channel response (V, psi) B vs A:  ' + ' | '.join(f'{nm} V {rb[c, 1]:.3f}/{ra[c, 1]:.3f}' for c, nm in enumerate(['p0x', 'p0y', 'p1x', 'p1y', 'p2x', 'p2y'])))
    p(f'   relative angle of the x rows, plane 1 - plane 2: B {np.degrees(rb[2, 2] - rb[4, 2]):+.2f}, A {np.degrees(ra[2, 2] - ra[4, 2]):+.2f} deg; '
      f'plane 2 - plane 0: B {np.degrees(rb[4, 2] - rb[0, 2]):+.2f} (plane 0 disc of {B["npix"][0]} px), A {np.degrees(ra[4, 2] - ra[0, 2]):+.2f}')
    # 3. pipeline J against A
    U_ = lambda J: (lambda W, s_, Vh: W @ Vh)(*np.linalg.svd(J))
    for nm, J in (('J1 pipeline (unitary)', U_(J1_PIPE)), ('J2 pipeline (unitary)', U_(J2_PIPE))):
        lr = linear_response(J)
        p(f'\n3. {nm}: rows x / y V {lr[0, 1]:.3f} / {lr[1, 1]:.3f}, psi {np.degrees(lr[0, 2]):+.2f} / {np.degrees(lr[1, 2]):+.2f} deg')
    lr1, lr2 = linear_response(U_(J1_PIPE)), linear_response(U_(J2_PIPE))
    p(f'   relative angle of the x rows J2 - J1: pipeline {np.degrees(lr2[0, 2] - lr1[0, 2]):+.2f} deg, A {np.degrees(ra[2, 2] - ra[0, 2]):+.2f} deg')
    # 4. J from A
    out_J = {}
    for hand, nm in (((-1, 1), 'hand_pipeline'), ((1, -1), 'hand_flipped')):
        J = J_from_response(fa['coef'], hand); out_J[nm] = J
        p(f'\n4. J_dichroic from A, circular signs of the rows x / y = {hand} ({nm}):')
        p('   ' + expression(J))
        for q in range(3): p('   ' + describe(J[q], f'model plane {q} (reconstruction plane {2 - q})'))
    np.savez(os.path.join(out, 'J_A.npz'), **out_J, coef=fa['coef'], k=fa['k'], theta=A['theta'])
    json.dump({'A_resp': fa['resp'].tolist(), 'A_k': fa['k'].tolist(), 'B_resp': sb['resp'].tolist(),
               'B': {k: np.asarray(v).tolist() for k, v in sb.items() if k != 'resp'},
               'J_expr': {k: expression(v) for k, v in out_J.items()}}, open(os.path.join(out, 'fits.json'), 'w'), indent=1)
    # 5. field dependence
    fm = field_maps(A, fa)
    np.savez(os.path.join(out, 'field_A.npz'), **fm)
    plot_field(fm, A['shape'], os.path.join(out, 'field_maps.png'))
    p(f'\n5. FIELD DEPENDENCE (A, tiles {A["tile"]}): sigma of psi per tile {np.round(np.median(fm["sig"], (1, 2)), 3)} deg')
    for nm, m in (('common rotation', fm['common']), ('relative rotation J2 - J1', fm['rel']), ('V J1 x (recon. plane 0)', fm['V'][0]),
                  ('V J2 x', fm['V'][2]), ('y/x plane 1 as reconstructed', fm['coef'][0, 3]/fm['coef'][0, 2]*fm['Lrel'][3]/fm['Lrel'][2])):
        p(f'   {nm}: p5-p95 {np.percentile(m, 5):+.3f} .. {np.percentile(m, 95):+.3f}, std {m.std():.3f}')
    for nm, (x, y) in BEADS_SLB.items():
        p(f'   SLB bead {nm}: common {at_position(fm["common"], x, y, A["shape"]):+.2f} deg, relative {at_position(fm["rel"], x, y, A["shape"]):+.2f} deg, '
          f'V J1 {at_position(fm["V"][0], x, y, A["shape"]):.3f}, V J2 {at_position(fm["V"][2], x, y, A["shape"]):.3f}')
    log.close()
    return A, B, fa, sb

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/mnt/d/Amaury/simu_precision_claude/2026-11-02_polar_calib')
