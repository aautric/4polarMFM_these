# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 2026

@author: Amaury Autric
amaury.autric@polytechnique.org
data analysis
"""

# %% Libraries
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import copy
from tqdm import tqdm
from tkinter import Tk, filedialog
from matplotlib.colors import hsv_to_rgb
import pandas as pd
import matplotlib as mpl
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter1d
import bisect
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import seaborn as sns
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/'))
# %% select file

#data = pd.read_csv("\\\\NAS_LOCCO\\Amaury\\DATA\\4_polar_MFM_these\\test_jax.csv", delimiter=';')
#data = pd.read_csv("/mnt/z/DATA/4_polar_MFM_these/test_jax.csv", delimiter=';')
look_up_folder = '/mnt/d/Amaury/DATA'
path = filedialog.askopenfilename(
    initialdir=look_up_folder,
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

# Open file dialog
data = pd.read_csv(path, sep=';', comment='#')
if "eta" in data.columns: # case to process SGD data
    frame = data['frame'].to_numpy().astype(int)
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    z = data['z'].to_numpy()
    rho = data['rho'].to_numpy()
    eta = data['eta'].to_numpy()
    delta = data['delta'].to_numpy()
    N_photons = data['N_photon'].to_numpy()
    score = data['score'].to_numpy()
    # deviance per pixel (goodness of fit, ~1 for a good fit), only in the runs where it was computed
    deviance = data['deviance'].to_numpy() if 'deviance' in data.columns else None
    #x_start = data['x_start'].to_numpy()
    #y_start = data['y_start'].to_numpy()
    #z_start = data['z_start'].to_numpy()
    #rho_start = data['rho_start'].to_numpy()
    #delta_start = data['delta_start'].to_numpy()
else: # case ratiometric only
    data = pd.read_csv(path, delimiter=',')
    frame = data['frame'].to_numpy()
    x = data['x [nm]'].to_numpy()
    y = data['y [nm]'].to_numpy()
    z = data['z [nm]'].to_numpy()
    rho = (data['rho'].to_numpy()+60)%180
    delta = data['delta'].to_numpy()
    N_photons = data['intensity [a.u.]'].to_numpy()
    score = data['sigmax [nm]'].to_numpy()
    deviance = None

# aberrations fitted in SGD2, copied by post_process_SGD.py on each localisation of a batch:
# one row per batch is kept, with the frames the batch covers
zern_columns = [c for c in data.columns if c[:2] in ('x_', 'y_') and '_Z' in c]
if 'batch' in data.columns and len(zern_columns) > 0:
    zern_data = data.groupby('batch').agg(first_frame=('frame', 'min'), last_frame=('frame', 'max'),
                                          **{c: (c, 'first') for c in zern_columns})
    zern_data = zern_data.dropna().reset_index() # the batches of a run without aberration fit
    # field-dependent aberrations (fov_aberrations of SGD_jax_CNN.py / SGD_jax.py): one set per PSF, not per batch
    zern_per_psf = bool((data.groupby('batch')[zern_columns].nunique() > 1).any().any())
    print('Aberrations of '+str(len(zern_data))+' batches loaded' + (', one set per PSF (field-dependent)' if zern_per_psf else ''))
else:
    zern_data = None

# goodness of fit used to filter: deviance per pixel (~1 for a good fit, same meaning whatever the loss and the
# background), or the score (final loss) for the files without deviance (older runs, ratiometric data).
# fit_quality is the same array as deviance / score, so it follows the merging of link_localizations
if deviance is not None:
    fit_quality, fit_quality_label = deviance, 'Deviance per pixel'
else:
    fit_quality, fit_quality_label = score, 'Final loss per PSF (no deviance in this file)'

# %% basic plots to see the thresholding
%matplotlib inline
plt.rcParams['figure.figsize'] = [12,3]
fig, ax = plt.subplots(1,3)
# range from the 1st to the 99th percentile: a few badly fitted PSF would otherwise put all the others in one bin
quality_range = np.nanpercentile(fit_quality, [1, 99])
hist = ax[0].hist(fit_quality, bins=50, range=quality_range)
ax[0].set_title(f'median {np.nanmedian(fit_quality):.3g}, {np.sum((fit_quality < quality_range[0]) | (fit_quality > quality_range[1]))} PSF out of the range', fontsize=10)
ax[0].set_xlabel(fit_quality_label)
ax[0].set_ylabel('Occurences')

hist = ax[1].hist(N_photons, bins=50)
ax[1].set_xlim((0, 20000))
ax[1].set_xlabel('Photon number per PSF')
ax[1].set_ylabel('Occurences')

hist = ax[2].hist(z, bins=80)
ax[2].set_xlabel('z')
ax[2].set_ylabel('Occurences')

# %% zernike aberrations across the SGD
if zern_data is None:
    print('No aberrations in '+path+' (none in the _linked csv: open fit_<run>.csv or fit_<run>_drift.csv), '
          'or run post_process_SGD.py on a run with aberrations')
elif zern_per_psf:
    # field-dependent: the coefficients of each PSF vs its position, one column per channel, one row per mode
    to_mlambda = 1000/(2*np.pi) # rad to milli-waves
    column = lambda pol, p, m: pol+'_p'+str(p)+'_Z'+str(m)
    modes = [m for m in range(1, 16) if any((data[column(pol, p, m)] != 0).any() for pol in 'xy' for p in range(3))]
    shown = np.random.default_rng(0).choice(len(data), min(len(data), 20000), replace=False)
    fig, ax = plt.subplots(len(modes), 6, figsize=(16, 2.2*len(modes)), sharex=True, sharey=True, squeeze=False)
    for k, m in enumerate(modes):
        for p in range(3):
            for c, pol in enumerate(['x', 'y']):
                values = data[column(pol, p, m)].to_numpy()[shown]*to_mlambda
                med = np.median(values); span = max(np.abs(values - med).max(), 0.5)  # colours centred on the median
                sc = ax[k, 2*p+c].scatter(x[shown]/1000, y[shown]/1000, c=values, s=1, cmap='RdBu_r', vmin=med - span, vmax=med + span)
                fig.colorbar(sc, ax=ax[k, 2*p+c])
                ax[k, 2*p+c].set_title(f'Z{m} plane {p} {pol}: {med:+.1f} m$\\lambda$', fontsize=8)
    for a in ax[-1]: a.set_xlabel('x (um)')
    for a in ax[:, 0]: a.set_ylabel('y (um)')
    fig.suptitle('Field-dependent aberrations of each PSF (fixed in the fit), m$\\lambda$ (title: median)')
    plt.show()
else:
    to_mlambda = 1000/(2*np.pi) # rad to milli-waves
    column = lambda pol, p, m: pol+'_p'+str(p)+'_Z'+str(m) # plane p, polarisation pol, Noll index m
    # only the fitted modes, the switched off ones (piston, tilt, defocus) stay at 0
    modes = [m for m in range(1, 16) if any((zern_data[column(pol, p, m)] != 0).any() for pol in 'xy' for p in range(3))]
    batch_frame = (zern_data['first_frame'] + zern_data['last_frame'])/2

    plt.rcParams['figure.figsize'] = [14, 10]
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    for p in range(3):
        for c, pol in enumerate(['x', 'y']):
            for m in modes:
                ax[p,c].plot(batch_frame, zern_data[column(pol, p, m)]*to_mlambda, label='Z'+str(m))
            ax[p,c].set(title='plane '+str(p)+' - '+pol+' polarisation', ylabel='coefficient (m$\\lambda$)')
    ax[2,0].set_xlabel('frame (middle of the batch)')
    ax[2,1].set_xlabel('frame (middle of the batch)')
    ax[0,1].legend(title='Noll index', fontsize=8, loc='upper left', bbox_to_anchor=(1.01,1))
    fig.suptitle('Aberrations fitted in SGD 2, one point per batch')
    plt.show()

    # distribution over the batches
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    for p in range(3):
        for c, pol in enumerate(['x', 'y']):
            values = np.array([zern_data[column(pol, p, m)] for m in modes])*to_mlambda # modes x batches
            # the density of a constant coefficient (frozen with a speed of 0) cannot be computed, it is shown as a point
            varying = np.std(values, axis=1) > 0
            positions = np.arange(len(modes))
            if varying.any():
                parts = ax[p,c].violinplot(list(values[varying]), positions=positions[varying], showmeans=True, showmedians=True, showextrema=False)
                parts['cmeans'].set_color('k')
                parts['cmedians'].set_color('r')
            ax[p,c].scatter(positions[~varying], values[~varying, 0], color='k', s=15)
            ax[p,c].axhline(0, color='k', lw=0.8)
            ax[p,c].set(title='plane '+str(p)+' - '+pol+' polarisation', ylabel='coefficient (m$\\lambda$)')
            ax[p,c].set_xticks(positions)
            ax[p,c].set_xticklabels(['Z'+str(m) for m in modes])
    fig.suptitle('Distribution of the aberrations over the '+str(len(zern_data))+' batches (black: mean, red: median)')
    plt.show()

# %% TO BE MODIFIED
# threshold on fit_quality: deviance per pixel (~1 for a good fit), or the final loss for files without deviance
quality_thresh = 2.0 if deviance is not None else 1500
mask1 = (fit_quality<quality_thresh) & (delta<150) & (delta>50) & (N_photons>50) & (N_photons<10000) & (z<2000) & (z>0) #& (eta>30) & (eta<150)
#mask1 = (delta<150) & (delta>60)
#%% mask selection
%matplotlib widget
plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax = fig.add_subplot()
vals = frame[mask1]
sc = ax.scatter(x[mask1] , y[mask1] , c=vals , cmap='coolwarm', s=1)
ax.set_title("select a point cloud")
ax.axis('equal')
cb = plt.colorbar(sc)
cb.ax.invert_yaxis()
points = np.column_stack((x, y))
mask = np.zeros(len(x), dtype=bool)

def onselect(verts):
    global mask
    path = Path(verts)
    mask = path.contains_points(points) & mask1
    print(mask)

lasso = LassoSelector(ax, onselect)
plt.show()


#%% drift correction by cross-correlation (no fiducial), localisations of mask
# redundant cross-correlation (RCC): the localisations of mask are cut in blocks of segment_frames frames, each block
# rendered as an image (pixel rcc_pixel nm, counts clipped at 1 per pixel so that a few clusters do not dominate,
# gaussian blur of rcc_blur px); the shift between two blocks is the peak of their cross-correlation (sub-pixel:
# parabola through the 3 points around the maximum), and the drift of each block is the least squares solution of
# d_j - d_i = shift_ij (d = 0 for the first block). In two steps, because far apart blocks have different molecules and
# the peak can jump to another filament:
# 1. pairs of blocks at most n_near apart, peak searched within max_drift nm -> first drift
# 2. all the pairs, peak searched within refine_window nm around the shift predicted by the first drift; pairs with a
#    residual above rcc_reject nm removed and solved again.
# The drift is interpolated linearly between the centres of the blocks. Uncertainty: the same on two random halves of
# the localisations, rms of their difference / 2 (each half has half the localisations: error of the whole ~ diff / 2).
# More localisations per block (longer blocks, larger mask) give a more precise drift: on actin1 (whole field, 2000
# frames, ~16000 localisations per block) ~9 nm, the two halves diverging slowly over the acquisition (up to ~40 nm)
# z drift (rcc_z): the image cross-correlation gives x and y only. For z, the same blocks: after the xy correction, the
# localisations of two blocks closer than z_match_xy in xy sample the same structure (whatever blinks in each block), the
# z shift between the blocks is the median of their z differences (second pass within z_window of the first median, to
# leave out structures crossing at another height), and the drift of each block is the least squares solution of
# d_j - d_i = shift_ij weighted by the number of matches, as for xy. Same uncertainty from the two halves.
# xcorr, ycorr (zcorr): drift for each frame number (index = frame), for the cell "drif correction : applying the
# correction" (skip the cell "drift correction computation with no fiducial")
from scipy.fft import rfft2, irfft2
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
segment_frames = 2000 # frames per block: more frames, more localisations per image (precise shift), less time resolution
rcc_pixel = 20.       # nm
rcc_blur = 1.         # px, gaussian blur of the images (~ localisation precision)
max_drift = 1000.     # nm, maximum shift searched between two blocks at most n_near apart
n_near = 3            # step 1: pairs of blocks at most n_near apart
refine_window = 100.  # nm, step 2: half width of the search around the predicted shift
rcc_reject = 30.      # nm, pairs with a residual above this value are left out
rcc_check = True      # uncertainty from two random halves of the localisations (3 times longer)
rcc_z = True          # z drift as well (correct_z), see z_drift
z_match_xy = 50.      # nm, localisations of two blocks closer than this in xy are matched (same structure)
z_window = 250.       # nm, z differences further than this from the first median are left out
z_min_pairs = 50      # matches needed for a pair of blocks to be used

def rcc_drift(xs_, ys_, fs_, f_first, f_last):
    # drift (nm) of the blocks of frames [f_first, f_last], at their centres; localisations xs_, ys_ of frames fs_
    edges_f = np.arange(f_first, f_last + segment_frames, segment_frames)
    n_seg = len(edges_f) - 1
    pad = max_drift + 4*rcc_pixel # the image is padded so that a shift of max_drift does not wrap around
    ex = np.arange(xs_.min() - pad, xs_.max() + pad + rcc_pixel, rcc_pixel)
    ey = np.arange(ys_.min() - pad, ys_.max() + pad + rcc_pixel, rcc_pixel)
    shape_img = (len(ex) - 1, len(ey) - 1)
    ffts, centres, counts_seg = [], [], []
    for k in range(n_seg):
        s_ = (fs_ >= edges_f[k]) & (fs_ < edges_f[k+1])
        img = gaussian_filter(np.minimum(np.histogram2d(xs_[s_], ys_[s_], bins=(ex, ey))[0], 1.), rcc_blur).astype(np.float32)
        ffts.append(rfft2(img - img.mean()))
        centres.append(np.mean(fs_[s_]) if s_.any() else (edges_f[k] + edges_f[k+1])/2)
        counts_seg.append(s_.sum())
    centres, counts_seg = np.array(centres), np.array(counts_seg)

    def peak_shift(fa, fb, centre_nm=(0., 0.), half_nm=max_drift):
        # shift (nm) of block b relative to block a, b(r) = a(r - shift), searched within half_nm of centre_nm
        c = np.fft.fftshift(irfft2(np.conj(fa)*fb, s=shape_img))
        m_px = int(half_nm/rcc_pixel)
        ox, oy = int(round(centre_nm[0]/rcc_pixel)), int(round(centre_nm[1]/rcc_pixel))
        cx, cy = shape_img[0]//2 + ox, shape_img[1]//2 + oy
        win = c[cx-m_px:cx+m_px+1, cy-m_px:cy+m_px+1]
        i, j = np.unravel_index(np.argmax(win), win.shape)
        def sub(a, b, c_): # vertex of the parabola through 3 points
            den = a - 2*b + c_
            return 0. if den == 0 else 0.5*(a - c_)/den
        di = sub(win[i-1, j], win[i, j], win[i+1, j]) if 0 < i < win.shape[0]-1 else 0.
        dj = sub(win[i, j-1], win[i, j], win[i, j+1]) if 0 < j < win.shape[1]-1 else 0.
        return (ox + i + di - m_px)*rcc_pixel, (oy + j + dj - m_px)*rcc_pixel

    def solve(pairs, shifts):
        # least squares with d = 0 for the first block, solved again without the pairs of residual > rcc_reject
        keep = np.ones(len(pairs), dtype=bool)
        for _ in range(2):
            A = np.zeros((keep.sum(), n_seg))
            A[np.arange(keep.sum()), pairs[keep, 1]] = 1; A[np.arange(keep.sum()), pairs[keep, 0]] = -1
            d = np.vstack(([0., 0.], np.linalg.lstsq(A[:, 1:], shifts[keep], rcond=None)[0]))
            resid = np.hypot(*(d[pairs[:, 1]] - d[pairs[:, 0]] - shifts).T)
            keep = resid < rcc_reject
        return d, keep, resid

    valid = counts_seg > 0
    pairs = np.array([(a, b) for a in range(n_seg) for b in range(a+1, min(a+n_near+1, n_seg)) if valid[a] and valid[b]])
    d_seg, _, _ = solve(pairs, np.array([peak_shift(ffts[a], ffts[b]) for a, b in pairs]))
    pairs = np.array([(a, b) for a in range(n_seg) for b in range(a+1, n_seg) if valid[a] and valid[b]])
    shifts = np.array([peak_shift(ffts[a], ffts[b], d_seg[b] - d_seg[a], refine_window) for a, b in pairs])
    d_seg, keep, resid = solve(pairs, shifts)
    return d_seg, centres, counts_seg, keep, resid

def z_drift(xs_, ys_, zs_, fs_, f_first, f_last):
    # z drift (nm) of the blocks of frames [f_first, f_last] (the blocks of rcc_drift), at their centres; xs_, ys_
    # already corrected for the xy drift. Returns the drift, the centres, the matches of each pair used, the residuals
    edges_f = np.arange(f_first, f_last + segment_frames, segment_frames)
    n_seg = len(edges_f) - 1
    seg = np.searchsorted(edges_f, fs_, side='right') - 1
    trees, zseg, centres = [], [], []
    for k in range(n_seg):
        s_ = seg == k
        trees.append(cKDTree(np.column_stack((xs_[s_], ys_[s_]))) if s_.any() else None)
        zseg.append(zs_[s_])
        centres.append(np.mean(fs_[s_]) if s_.any() else (edges_f[k] + edges_f[k+1])/2)
    pairs, shifts, n_match = [], [], []
    for a in range(n_seg):
        for b in range(a+1, n_seg):
            if trees[a] is None or trees[b] is None:
                continue
            m = trees[a].sparse_distance_matrix(trees[b], z_match_xy, output_type='ndarray')
            if len(m) < z_min_pairs:
                continue
            dz = zseg[b][m['j']] - zseg[a][m['i']]
            dz = dz[np.abs(dz - np.median(dz)) < z_window]
            pairs.append((a, b)); shifts.append(np.median(dz)); n_match.append(len(dz))
    pairs, shifts, n_match = np.array(pairs), np.array(shifts), np.array(n_match)
    w = np.sqrt(n_match)
    A = np.zeros((len(pairs), n_seg))
    A[np.arange(len(pairs)), pairs[:, 1]] = 1; A[np.arange(len(pairs)), pairs[:, 0]] = -1
    d = np.r_[0., np.linalg.lstsq(A[:, 1:]*w[:, None], shifts*w, rcond=None)[0]]
    return d, np.array(centres), n_match, d[pairs[:, 1]] - d[pairs[:, 0]] - shifts

sel_rcc = mask if np.any(mask) else mask1
xs_, ys_, fs_ = x[sel_rcc], y[sel_rcc], frame[sel_rcc]
f_first, f_last = int(fs_.min()), int(fs_.max())
frames_all = np.arange(int(np.max(frame)) + 1)
def drift_curves(d_seg, centres):
    # drift for each frame, 0 on the first frame of the selection
    cx_ = np.interp(frames_all, centres, d_seg[:, 0]); cy_ = np.interp(frames_all, centres, d_seg[:, 1])
    return cx_ - cx_[f_first], cy_ - cy_[f_first]
def z_curve(xs_, ys_, zs_, fs_):
    # z drift for each frame, 0 on the first frame of the selection, from localisations corrected for the xy drift
    dz_seg, zc_, n_match, zres = z_drift(xs_, ys_, zs_, fs_, f_first, f_last)
    c_ = np.interp(frames_all, zc_, dz_seg)
    return c_ - c_[f_first], dz_seg, zc_, n_match, zres

d_seg, centres, counts_seg, keep, resid = rcc_drift(xs_, ys_, fs_, f_first, f_last)
xcorr, ycorr = drift_curves(d_seg, centres)
correct_z = rcc_z
print(f'{len(centres)} blocks of {segment_frames} frames ({np.median(counts_seg):.0f} localisations per block, median), '
      f'{keep.sum()}/{len(keep)} pairs kept, residual of the kept pairs {np.median(resid[keep]):.1f} nm (median); '
      f'drift range x {np.ptp(xcorr[f_first:f_last+1]):.0f} nm, y {np.ptp(ycorr[f_first:f_last+1]):.0f} nm')
if np.median(counts_seg) < 8000:
    print('WARNING: less than 8000 localisations per block, the drift may be noisy: longer blocks (segment_frames) or a larger mask')
if rcc_z:
    fi_ = fs_.astype(int)
    zcorr, dz_seg, zcentres, n_match, zres = z_curve(xs_ - xcorr[fi_], ys_ - ycorr[fi_], z[sel_rcc], fs_)
    print(f'z: {len(n_match)} pairs of blocks, {np.median(n_match):.0f} matches per pair (median), residual '
          f'{np.median(np.abs(zres)):.1f} nm (median); drift range z {np.ptp(zcorr[f_first:f_last+1]):.0f} nm')

fig, ax = plt.subplots(1, 3 if rcc_z else 2, figsize=(18 if rcc_z else 12, 4))
in_sel = slice(f_first, f_last + 1)
if rcc_check:
    half_ = np.random.default_rng(0).random(len(xs_)) < 0.5
    curves_h = [drift_curves(*rcc_drift(xs_[h], ys_[h], fs_[h], f_first, f_last)[:2]) for h in (half_, ~half_)]
    err_x = np.std(curves_h[0][0][in_sel] - curves_h[1][0][in_sel])/2
    err_y = np.std(curves_h[0][1][in_sel] - curves_h[1][1][in_sel])/2
    msg = f'uncertainty (two halves of the localisations): x {err_x:.1f} nm, y {err_y:.1f} nm'
    if rcc_z:
        curves_h = [c_ + (z_curve(xs_[h] - c_[0][fi_[h]], ys_[h] - c_[1][fi_[h]], z[sel_rcc][h], fs_[h])[0],)
                    for c_, h in zip(curves_h, (half_, ~half_))]
        err_z = np.std(curves_h[0][2][in_sel] - curves_h[1][2][in_sel])/2
        msg += f', z {err_z:.1f} nm'
    print(msg)
panels = [(ax[0], xcorr, 'x', d_seg[:, 0], centres), (ax[1], ycorr, 'y', d_seg[:, 1], centres)]
if rcc_z:
    panels.append((ax[2], zcorr, 'z', dz_seg, zcentres))
for k, (a_, curve, lab, pts, cen) in enumerate(panels):
    if rcc_check:
        for h, c_ in enumerate(curves_h):
            a_.plot(frames_all[in_sel], c_[k][in_sel], lw=0.8, alpha=0.6, label=f'half {"AB"[h]}')
    a_.plot(frames_all[in_sel], curve[in_sel], 'k-', lw=1.5, label='all')
    a_.plot(cen, pts - np.interp(f_first, cen, pts), 'ko', ms=4)
    a_.set_xlabel('frame'); a_.set_ylabel(f'drift {lab} (nm)'); a_.grid(alpha=0.4); a_.legend(fontsize=9)
plt.tight_layout()
plt.show()
#%% drift correction computation with no fiducial
def mean_x_per_20_frames(x, frame, window=20):

    bins = (frame // window).astype(int)
    n_bins = bins.max() + 1

    sum_x = np.bincount(bins, weights=x, minlength=n_bins)
    count = np.bincount(bins, minlength=n_bins)

    mean_x = np.zeros(n_bins)

    nonzero = count > 0
    mean_x[nonzero] = sum_x[nonzero] / count[nonzero]

    mean_x[~nonzero] = np.nan  # only real empty bins
    mean_x = np.where(np.isnan(mean_x), np.nanmean(mean_x), mean_x)
    return mean_x

mean_x = mean_x_per_20_frames(x[mask], frame[mask], 100)
mean_y = mean_x_per_20_frames(y[mask], frame[mask], 100)

for i in range(len(mean_x)):
    if np.isnan(mean_x[i]):
        mean_x[i] = (mean_x[i-1]+mean_x[i+1])/2
        mean_y[i] = (mean_y[i-1]+mean_y[i+1])/2
        

plt.rcParams['figure.figsize'] = [8,4]
fig, ax = plt.subplots()
xcorr = gaussian_filter1d(mean_x, 7)

xcorr = xcorr-xcorr[0]
ax.plot(xcorr)
ax.set_xlabel('bin of 100 frames')
ax.set_ylabel('drift x (nm)')
plt.show()
fig, ax = plt.subplots()
ycorr = gaussian_filter1d(mean_y, 11)
ycorr = ycorr-ycorr[0]
ax.plot(ycorr)
ax.set_xlabel('bin of 100 frames')
ax.set_ylabel('drift y (nm)')
plt.show()
correct_z = False
#%% Drift correction with fiducial

plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax = fig.add_subplot()
sc = ax.scatter(x , y, c=N_photons , cmap='coolwarm', s=0.1, vmax=10000)
ax.set_title("Select the zone of the fiducial")
ax.axis('equal')
cb = plt.colorbar(sc)
cb.ax.invert_yaxis()
points = np.column_stack((x, y))
mask = np.zeros(len(x), dtype=bool)

def onselect(verts):
    global mask_fiducial
    path = Path(verts)
    mask_fiducial = path.contains_points(points) 

lasso = LassoSelector(ax, onselect)
#%% fiducial suite
_, unique_indices = np.unique(frame[mask_fiducial], return_index=True)
mask_fiducial_clean = np.where(mask_fiducial)[0][unique_indices]
frames_fiducial = frame[mask_fiducial_clean]
indices_fiducial = mask_fiducial_clean

frames_array = np.array(sorted(frames_fiducial))
diffs = np.diff(frames_array)

if (diffs != 1).any():
    missing_or_dup = frames_array[:-1][diffs != 1]
    print(f'Non-consecutive frames at: {missing_or_dup}')
    
    # first remove duplicates
    _, unique_indices2 = np.unique(frames_fiducial, return_index=True)
    indices_fiducial = indices_fiducial[unique_indices2]
    frames_fiducial = frames_fiducial[unique_indices2]
    
    # then fill missing frames by repeating previous
    full_range = np.arange(frames_fiducial.min(), frames_fiducial.max() + 1)
    new_indices = []
    new_frames = []
    for f in full_range:
        if f in frames_fiducial:
            idx = np.where(frames_fiducial == f)[0][0]
            new_indices.append(indices_fiducial[idx])
            new_frames.append(f)
        else:
            # repeat previous index
            print(f'Missing frame {f}, repeating previous')
            new_indices.append(new_indices[-1])
            new_frames.append(f)
    
    frames_fiducial = np.array(new_frames)
    indices_fiducial = np.array(new_indices)
    frames_array = frames_fiducial
    
    diffs = np.diff(frames_array)
    if (diffs != 1).any():
        raise ValueError('Could not fix frames')
    print('Frames fixed successfully')
filter_kernel = 101
xcorr = gaussian_filter1d(x[indices_fiducial]-x[indices_fiducial[0]],filter_kernel)
ycorr = gaussian_filter1d(y[indices_fiducial]-y[indices_fiducial[0]],filter_kernel)
zcorr = gaussian_filter1d(z[indices_fiducial]-z[indices_fiducial[0]],filter_kernel)
fig = plt.figure()
plt.plot(xcorr)
plt.plot(ycorr)
plt.plot(zcorr)
plt.show()
correct_z = True
#%% drif correction : applying the correction
frame_bins = np.linspace(0, np.max(frame), len(xcorr)).astype(int)

for ind, frame_ in tqdm(enumerate(frame)):
    idx = bisect.bisect_left(frame_bins, frame_)
    #print(frame_, idx, frame_bins[idx-1])
    x[ind]-=xcorr[idx-1]
    y[ind]-=ycorr[idx-1]
    if correct_z:
        z[ind]-=zcorr[idx-1]
    #print(ind, frame_, x[ind], y[ind], xcorr[idx-1]*0.45, ycorr[idx-1]*0.6)
    
#%% correction doublons
# 1. duplicates in the same frame (the same PSF fitted twice): pairs closer than same_frame_xy in xy and link_z in z are
#    merged into one localisation (mean, circular for rho and eta). Not used for the precision: same image
# 2. the same molecule on several frames: tracking frame by frame, each localisation is attached to the closest track
#    whose last point is closer than link_xy in xy and link_z in z, at most max_gap missing frames before (one to one,
#    closest pairs first). Each track of 2 points or more is merged into its last point; its std give the precision
# thresholds from the distances between localisations of actin1 (10-15) and actin2 (10-10), after the filters on the
# deviance, delta, N and z:
# - same frame: the duplicates are within 10 nm for most of them (actin1), up to ~50 nm for actin2; two distinct
#   emitters closer than ~200 nm are not detected separately, so nothing real is lost below 50 nm
# - consecutive frames: distance between the two positions of the same molecule, std sqrt(2)*(precision per axis):
#   half of the maximum at ~12 nm (actin1) and ~23 nm (actin2), above the background of the other molecules up to
#   ~60 nm (actin1) and ~100 nm (actin2). With 60 nm, wrong links (other molecules) are < 0.3 % of the links
# - z: |dz| between consecutive frames of the same molecule, 95th percentile 52 nm (actin1) and 77 nm (actin2),
#   99th 90 and 125 nm, ~3*sqrt(2)*(precision in z); also used for the duplicates of the same frame
# conservative values below (few wrong links and merges, the tail of the less precise molecules is cut: precision
# slightly underestimated); 50 / 60 / 100 nm keep this tail (actin1: lateral 10.0 nm instead of 8.4 with 30 / 60)
# distances: in the plane sqrt(dx^2 + dy^2) (not x and y separately), and |dz| separately
from scipy.spatial import cKDTree
link_xy = 40.        # nm, the same molecule on consecutive frames
link_z = 70.         # nm, consecutive frames and duplicates of the same frame; ~130 nm for SLB (fewer photons)
max_gap = 1          # number of missing frames allowed inside a track (0: consecutive frames only)
same_frame_xy = 20.  # nm, duplicates in the same frame (the same PSF fitted twice)

def circular_mean_180(angles):
    # mean of angles defined modulo 180 deg (rho, eta): 178 and 2 give 0, not 90
    return (np.degrees(np.angle(np.mean(np.exp(2j*np.radians(angles)))))/2) % 180

def circular_std_180(angles):
    # std (n-1) of angles defined modulo 180 deg, deviations taken around the circular mean in [-90, 90[
    deviations = (angles - circular_mean_180(angles) + 90) % 180 - 90
    return np.sqrt(np.sum(deviations**2)/(len(angles)-1))

def merge_into(keep, others, arrays, angles):
    # arrays[keep] = mean over keep and others (circular for the angles), the others set to nan
    group = np.concatenate(([keep], others))
    for name, a in arrays.items():
        a[keep] = circular_mean_180(a[group]) if name in angles else np.mean(a[group])
        a[others] = np.nan

def link_localizations(x, y, z, rho, eta, delta, N_photons, score, frame, deviance=None,
                       link_xy=link_xy, link_z=link_z, max_gap=max_gap, same_frame_xy=same_frame_xy):
    frame = frame.astype(float)
    arrays = {'x': x, 'y': y, 'z': z, 'rho': rho, 'eta': eta, 'delta': delta, 'N_photons': N_photons, 'score': score,
              'frame': frame}
    if deviance is not None: # modified in place as the other arrays
        arrays['deviance'] = deviance
    angles = ('rho', 'eta')
    valid = np.where(np.isfinite(x) & np.isfinite(frame))[0]
    valid = valid[np.argsort(frame[valid], kind='stable')]
    frames, starts = np.unique(frame[valid], return_index=True)
    groups = np.split(valid, starts[1:]) # indices of the localisations of each frame

    # 1. duplicates in the same frame
    n_duplicates = 0
    for g in groups:
        if len(g) < 2:
            continue
        pairs = [(a, b) for a, b in cKDTree(np.column_stack((x[g], y[g]))).query_pairs(same_frame_xy)
                 if abs(z[g[a]] - z[g[b]]) < link_z]
        if not pairs:
            continue
        parent = list(range(len(g))) # union find: a duplicate of a duplicate is the same emitter
        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b in pairs:
            parent[root(a)] = root(b)
        for r in set(root(a) for a in range(len(g))):
            members = g[[a for a in range(len(g)) if root(a) == r]]
            if len(members) > 1:
                merge_into(members[0], members[1:], arrays, angles)
                n_duplicates += len(members) - 1
    groups = [g[np.isfinite(x[g])] for g in groups]

    # 2. tracks over the frames
    tracks = []      # indices of each track
    active = []      # index in tracks of the tracks that can still be extended
    for f, g in zip(frames, groups):
        active = [t for t in active if f - frame[tracks[t][-1]] <= max_gap + 1]
        attached = np.zeros(len(g), dtype=bool)
        if active and len(g):
            last = np.array([tracks[t][-1] for t in active])
            dist, nearest = cKDTree(np.column_stack((x[last], y[last]))).query(
                np.column_stack((x[g], y[g])), k=min(5, len(last)), distance_upper_bound=link_xy)
            dist, nearest = dist.reshape(len(g), -1), nearest.reshape(len(g), -1)
            candidates = sorted((dist[a, c], a, nearest[a, c]) for a in range(len(g)) for c in range(dist.shape[1])
                                if np.isfinite(dist[a, c]) and abs(z[g[a]] - z[last[nearest[a, c]]]) < link_z)
            used = set()
            for d, a, c in candidates: # closest pairs first, one to one
                if attached[a] or c in used:
                    continue
                tracks[active[c]].append(g[a]); attached[a] = True; used.add(c)
        for a in np.where(~attached)[0]:
            tracks.append([g[a]]); active.append(len(tracks) - 1)

    # 3. each track of 2 points or more merged into its last point, with its std for the precision
    stdx, stdy, stdz, stdrho, stdeta, stddelta, nlink = [], [], [], [], [], [], []
    meanx, meany, meanz, meanrho, meaneta, meandelta, meanN, meanscore = [], [], [], [], [], [], [], []
    for t in tracks:
        if len(t) < 2:
            continue
        idx = np.array(t)
        nlink.append(len(idx))
        stdx.append(np.std(x[idx], ddof=1)); stdy.append(np.std(y[idx], ddof=1)); stdz.append(np.std(z[idx], ddof=1))
        stdrho.append(circular_std_180(rho[idx])); stdeta.append(circular_std_180(eta[idx]))
        stddelta.append(np.std(delta[idx], ddof=1))
        meanx.append(np.mean(x[idx])); meany.append(np.mean(y[idx])); meanz.append(np.mean(z[idx]))
        meanrho.append(circular_mean_180(rho[idx])); meaneta.append(circular_mean_180(eta[idx]))
        meandelta.append(np.mean(delta[idx])); meanN.append(np.mean(N_photons[idx])); meanscore.append(np.mean(score[idx]))
        merge_into(idx[-1], idx[:-1], arrays, angles)

    print(f'{n_duplicates} duplicates in the same frame merged; {len(nlink)} molecules linked over several frames '
          f'(link_xy {link_xy} nm, link_z {link_z} nm, max_gap {max_gap}); removed {int(np.sum(np.isnan(x)))} over {len(x)}')
    return (x, y, z, rho, eta, delta, N_photons, score, frame,
            np.array(stdx), np.array(stdy), np.array(stdz), np.array(stdrho), np.array(stdeta), np.array(stddelta),
            np.array(meanx), np.array(meany), np.array(meanz), np.array(meanrho), np.array(meaneta), np.array(meandelta), np.array(meanN), np.array(meanscore), np.array(nlink))

x, y, z, rho, eta, delta, N_photons, score, frame, stdx, stdy, stdz, stdrho, stdeta, stddelta, meanx, meany, meanz, meanrho, meaneta, meandelta, meanN, meanscore, nlink = link_localizations(x, y, z, rho, eta, delta, N_photons, score, frame, deviance=deviance)
#%% std x, y, lateral, z, rho, eta, delta
# precision: std of the localisations linked over consecutive frames (same molecule). The mean and median of the
# per-molecule std underestimate the precision (molecules seen on 2-3 frames only); the pooled std
# sqrt(sum((n-1) s^2) / sum(n-1)) uses all the frames of all the molecules and is unbiased (sensitive to wrong links).
# lateral precision: per axis, sqrt((std x^2 + std y^2)/2) for each molecule; the radial one (distance in xy to the mean
# position) is sqrt(2) times larger
if len(stdx) == 0 or len(nlink) != len(stdx):
    raise ValueError(str(len(stdx))+' linked molecules ('+str(len(nlink))+' in nlink): load the csv again and run '
                     'the "correction doublons" cell once (a second linking on already linked data finds almost nothing)')
stdlat = np.sqrt((stdx**2 + stdy**2)/2)
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
summary = []
for a, values, label, unit in [(ax[0,0], stdx, 'std x', 'nm'), (ax[0,1], stdy, 'std y', 'nm'),
                               (ax[0,2], stdlat, 'std lateral', 'nm'), (ax[0,3], stdz, 'std z', 'nm'),
                               (ax[1,0], stdrho, 'std $\\rho$', 'deg'), (ax[1,1], stdeta, 'std $\\eta$', 'deg'),
                               (ax[1,2], stddelta, 'std $\\delta$', 'deg')]:
    mean_v, median_v = np.nanmean(values), np.nanmedian(values)
    ok = ~np.isnan(values)
    pooled_v = np.sqrt(np.sum((nlink[ok]-1)*values[ok]**2) / np.sum(nlink[ok]-1))
    a.hist(values, bins=100)
    a.axvline(mean_v, color='r', label='mean')
    a.axvline(median_v, color='k', linestyle='--', label='median')
    a.axvline(pooled_v, color='g', label='pooled')
    a.set_title(f'pooled {pooled_v:.1f} {unit} (mean {mean_v:.1f}, median {median_v:.1f})', fontsize=11)
    a.set_xlabel(label+' ('+unit+')')
    a.set_ylabel('Occurences')
    a.legend(fontsize=9)
    summary.append((label.replace('$\\', '').replace('$', ''), unit, pooled_v, mean_v, median_v))
print(f'{len(stdx)} molecules seen on more than one frame (median {np.median(nlink):.0f} frames)')
print(f"{'':16s} {'pooled':>8s} {'mean':>8s} {'median':>8s}")
for name, unit, pooled_v, mean_v, median_v in summary:
    print(f'{name+" ("+unit+")":16s} {pooled_v:8.2f} {mean_v:8.2f} {median_v:8.2f}')
ax[1,3].axis('off')
ax[1,3].text(0, 0.95, f'{len(stdx)} molecules on > 1 frame\n\n{"":16s}{"pooled":>7s}{"mean":>7s}{"median":>7s}\n'
             + '\n'.join(f'{name+" ("+unit+")":16s}{pooled_v:7.1f}{mean_v:7.1f}{median_v:7.1f}' for name, unit, pooled_v, mean_v, median_v in summary),
             va='top', family='monospace', fontsize=11)
plt.tight_layout()
plt.show()
#%% select new filter
mask1 = (fit_quality<quality_thresh) & (z<1500) & (z>0) & (delta<150) &  (delta>50) & (N_photons>50) & (N_photons<10000)
#%% select mask
%matplotlib qt
plt.rcParams['figure.figsize'] = [10,10]
fig = plt.figure()
ax = fig.add_subplot()
vals = z[mask1]

sc = ax.scatter(x[mask1] , y[mask1] , c=vals , cmap='plasma', s=0.1)
ax.set_title("select a point cloud")
ax.axis('equal')
cb = plt.colorbar(sc)
cb.ax.invert_yaxis()
points = np.column_stack((x, y))
mask = np.zeros(len(x), dtype=bool)

def onselect(verts):
    global mask
    path = Path(verts)
    mask = path.contains_points(points) & mask1
    print(mask)

lasso = LassoSelector(ax, onselect)
plt.show()


#%% SIZE OF THE SCATTER POINTS
s = 0.1
#%% plot of the localisations, white on black, no colour coding
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('black')
ax.scatter(x[mask]/1000, y[mask]/1000, color='white', s=s, rasterized=True)
ax.set_aspect('equal')
ax.set_xlabel('x ($\\mu$m)')
ax.set_ylabel('y ($\\mu$m)')
ax.set_title(f'{np.sum(mask)} localisations')
plt.show()
#%% plot rho
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
ax0.set_facecolor('black')
#ax1.set_facecolor('black')
sc = ax0.scatter(x[mask]/1000, y[mask]/1000,
                 c=rho[mask] / 180.0,
                 cmap='hsv', s=s, rasterized=True)
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')

# half circle polar histogram with hsv colormap
rho_rad = np.deg2rad(rho[mask])
rho_mirrored = np.concatenate([rho_rad, rho_rad + np.pi])
bins = np.linspace(0, 2*np.pi, 73)  # 72 bins = 5° each
counts, edges = np.histogram(rho_mirrored, bins=bins)
width = edges[1] - edges[0]
centers = (edges[:-1] + edges[1:]) / 2

# color each bar by its angle using hsv colormap
colors = plt.cm.hsv((centers % np.pi) / np.pi)
ax1.bar(centers, counts, width=width, bottom=0, alpha=0.9, color=colors)

# half circle: show only 0 to pi (upper half), counter clockwise, starting right
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_zero_location('E')   # 0° on the right
ax1.set_theta_direction(1)         # counter clockwise

# clean up ticks
ax1.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax1.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
ax1.set_ylabel('Count', labelpad=30)
ax1.set_title(r'$\rho$')

# colour scale of rho as the outer border of the histogram (replaces the small circular colorbar)
r_max = counts.max()*1.05
ring = np.linspace(0, np.pi, 512)
ax1.pcolormesh(*np.meshgrid(ring, [r_max, r_max*1.08]), np.tile(np.rad2deg(ring), (2, 1)),
               cmap='hsv', vmin=0, vmax=180, shading='auto')
ax1.set_ylim(0, r_max*1.08)
ax1.set_rticks(np.round(np.linspace(0, counts.max(), 4)[1:], -int(np.floor(np.log10(counts.max()))) + 1)) # 3 count ticks, inside the border
ax1.spines['polar'].set_visible(False)

plt.tight_layout()
plt.show()


#%% plot z
%matplotlib qt
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()
ax.set_facecolor('black')
sc = ax.scatter(x[mask]/1000, y[mask]/1000,
                 c=z[mask] / 1000 ,
                 cmap='plasma', s=s, rasterized=True)#, vmin=0.4, vmax=1.)
cbar = plt.colorbar(sc)
cbar.set_label("$z$ ($\\mu$m)")
plt.axis('equal')
#plt.xlim((10000, 22500))
#plt.ylim((2200, 14000))
ax.set_xlabel('x ($\\mu$m)')
ax.set_ylabel('y ($\\mu$m)')
plt.show()

#%% histogram of z (mask)
# each bar coloured by the rho of its localisations (axial mean, hsv as in the plot rho cell)
z_bin = 20. # nm
ok_z = np.isfinite(z[mask]) & np.isfinite(rho[mask])
z_m, rho_zm = z[mask][ok_z], rho[mask][ok_z]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(figsize=(7, 4))
counts_z, edges_z, bars_z = ax.hist(z_m/1000, bins=np.arange(z_m.min(), z_m.max() + z_bin, z_bin)/1000)
bin_z = np.clip(np.digitize(z_m/1000, edges_z) - 1, 0, len(bars_z) - 1)
for k, b_ in enumerate(bars_z):
    r_ = np.deg2rad(rho_zm[bin_z == k])
    if len(r_):
        b_.set_facecolor(plt.cm.hsv((np.rad2deg(np.arctan2(np.sum(np.sin(2*r_)), np.sum(np.cos(2*r_))))/2 % 180)/180))
sm_rho = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(0, 180))
cbar = plt.colorbar(sm_rho, ax=ax, ticks=[0, 45, 90, 135, 180])
cbar.set_label('$\\rho$ (°), mean of the bin')
ax.set_xlabel('$z$ ($\\mu$m)')
ax.set_ylabel('Occurences')
ax.set_title(f'{len(z_m)} localisations, std {np.std(z_m):.0f} nm', fontsize=13)
plt.tight_layout()
plt.show()

#%% Plot delta
plt.rcParams["figure.figsize"] = [12,7]
plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(1, 2)
# ---------------- Scatter plot ----------------
ax[0].set_facecolor("black")
sc = ax[0].scatter(
    x[mask] / 1000,
    y[mask] / 1000,
    c=delta[mask],
    cmap="viridis",
    s=s,
    vmin=50,
    vmax=150, rasterized=True)
cbar = plt.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.04)
cbar.set_label(r"$\delta$ (°)")
cbar.set_ticks([50, 75, 100, 125, 150])
ax[0].set_aspect("equal")
ax[0].set_xlabel(r"x ($\mu$m)")
ax[0].set_ylabel(r"y ($\mu$m)")
ax[0].set_title(r"$\delta$ map")
# ---------------- Histogram ----------------
ax[1].hist(
    delta[mask],
    bins=40,
    range=(50, 150),
    color="tab:blue",
    edgecolor="black")
ax[1].set_xlim(50, 150)
ax[1].set_xlabel(r"$\delta$ (°)")
ax[1].set_ylabel("Count")
ax[1].set_title(r"$\delta$ distribution")
plt.tight_layout()
plt.show()
#%% plot deviance and photon number
# two maps of the points of mask, coloured by the deviance per pixel of each PSF (goodness of fit: ~1 when the
# model describes the data, whatever the photon number and the background) and by its photon number.
# Without deviance in the csv (runs before it was added), the score is shown instead, but it mostly follows
# the background. Colour range from the 1st to the 99th percentile so that a few outliers do not hide the rest
plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, 2)
if deviance is not None and len(deviance) == len(x):
    map_quality = (deviance[mask], 'deviance per pixel')
else:
    if deviance is not None: # deviance left from another csv loaded before in the session
        print('deviance has '+str(len(deviance))+' values for '+str(len(x))+' points: it comes from another file, run the cell that loads the csv again')
    map_quality = (score[mask], 'score')
for a, values, cmap_name, label in [(ax[0], map_quality[0], 'viridis', map_quality[1]),
                                    (ax[1], N_photons[mask], 'YlOrRd', 'N photons')]:
    a.set_facecolor('black')
    vmin, vmax = np.nanpercentile(values, [1, 99])
    sc = a.scatter(x[mask]/1000, y[mask]/1000, c=values, cmap=cmap_name, vmin=vmin, vmax=vmax, s=s, rasterized=True)
    fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04, label=label)
    a.set_aspect('equal')
    a.set_xlabel('x ($\\mu$m)')
    a.set_ylabel('y ($\\mu$m)')
    a.set_title(label+' map')
plt.tight_layout()
plt.show()
#%%colormap
hls_colors = sns.color_palette("hls", 256)
cmap = mcolors.ListedColormap(hls_colors)
#%% plot eta
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1,2)
ax[0].set_facecolor('black')
sc = ax[0].scatter(x[mask]/1000, y[mask]/1000,
                 c=eta[mask] / 180.0,
                 cmap=cmap, s=s, vmin=30/180, vmax=150/180, rasterized=True)
ax[0].set_aspect('equal')
ax[0].set_xlabel('x ($\\mu$m)')
ax[0].set_ylabel('y ($\\mu$m)')

### second plot #####

eta_vals = eta[mask]

# original + mirrored (theta -> theta + pi)
eta_rad = np.deg2rad(eta_vals)
eta_mirrored = np.concatenate([eta_rad, eta_rad + np.pi])

# bins over full circle
bins = np.linspace(0, 2*np.pi, 201)
counts, edges = np.histogram(eta_mirrored, bins=bins)

centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]

# keep symmetry in coloring (wrap back to [0, pi])
centers_folded = centers % np.pi

# consistent color mapping (same as scatter)
#colors = cmap(norm(np.rad2deg(centers_folded)))

norm = plt.Normalize(30, 150)

# rebuild polar axis
ax[1].remove()
ax1 = fig.add_subplot(1, 2, 2, projection='polar')

# White angular tick labels
for label in ax1.get_xticklabels():
    label.set_color('black')
hls_colors_bar = cmap(norm(np.rad2deg(centers_folded))) # same colours as the scatter (30-150, clipped)
# White radial tick labels
for label in ax1.get_yticklabels():
    label.set_color('black')
ax1.bar(
    centers,
    counts,
    width=width,
    bottom=0,
    color=hls_colors_bar,
    edgecolor='none',
    alpha=0.95
)
# reference A sin^2(eta), normalised to the same number of counts as the histogram (sum over the bins)
iso_scale = counts.sum() / np.sum(np.sin(centers)**2)
iso_theta = np.linspace(0, 2*np.pi, 721)
ax1.plot(iso_theta, iso_scale*np.sin(iso_theta)**2, color='k', lw=1.5, label=r'A sin$^2\eta$ (same counts)')
ax1.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.12), fontsize=10, frameon=False)
# orientation: 0° top, clockwise
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_facecolor('white')
ax1.spines['polar'].set_color('black')
ax1.tick_params(axis='x', colors='black')
ax1.tick_params(axis='y', colors='black')
ax1.yaxis.label.set_color('black')

for label in ax1.get_xticklabels():
    label.set_color('black')

for label in ax1.get_yticklabels():
    label.set_color('black')
# show only 0–360 but symmetric meaning is enforced
ax1.set_thetagrids(
    np.arange(0, 360, 60),
    labels=[f"{d % 180}°" for d in np.arange(0, 360, 60)] # eta and eta + 180 are the same: labels modulo 180
)

ax1.set_ylabel("Count", labelpad=25)

# colour scale of eta as the outer border of the histogram, as in the plot rho cell (eta and eta + 180 have the same
# colour, below 30 and above 150 the colours of 30 and 150 as in the scatter)
r_max = max(counts.max(), iso_scale)*1.05
ring = np.linspace(0, 2*np.pi, 1024)
ax1.pcolormesh(*np.meshgrid(ring, [r_max, r_max*1.08]), np.tile(np.rad2deg(ring) % 180, (2, 1)),
               cmap=cmap, norm=norm, shading='auto')
ax1.set_ylim(0, r_max*1.08)
ax1.set_rticks(np.round(np.linspace(0, counts.max(), 4)[1:], -int(np.floor(np.log10(counts.max()))) + 1)) # 3 count ticks, inside the border
ax1.spines['polar'].set_visible(False)
ax1.set_title(r'$\eta$')
plt.show()

#%% plot eta 2
# create hls colormap from seaborn
plt.rcParams['figure.figsize'] = [5, 3]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()
ax.set_facecolor('black')
sc = ax.scatter(
    y[mask] / 1000,
    z[mask] / 1000,
    c=eta[mask],
    cmap=cmap,
    s=s,
    vmin=0,
    vmax=180
)
plt.axis('equal')
plt.xlim(-17, -14)
cax = fig.add_axes(
    [0.25, 0.60, 0.16, 0.16],
    projection='polar'
)

theta = np.linspace(0, 2*np.pi, 1024)
r = np.array([0.8, 1.0])
Theta, R = np.meshgrid(theta, r)
theta_deg = np.rad2deg(theta)
C = np.tile(theta_deg % 180, (2, 1))
cax.pcolormesh(
    Theta, R, C,
    cmap=cmap,
    vmin=0,
    vmax=180,
    shading='auto',
    edgecolors='none'
)
cax.set_theta_zero_location('N')
cax.set_theta_direction(-1)
ticks = np.arange(0, 360, 60)
cax.set_thetagrids(ticks, labels=[f'{t%180}°' for t in ticks])
cax.set_facecolor('black')
cax.tick_params(axis='x', colors='white', pad=5)
cax.set_rticks([])
cax.grid(False)
cax.spines['polar'].set_visible(False)
cax.text(0, 0, r'$\eta$', ha='center', va='center', fontsize=14, color='white')

ax.set_xlabel('y ($\\mu$m)')
ax.set_ylabel('z ($\\mu$m)')
plt.tight_layout()
plt.show()
#%% plot eta-z
new_zoom=mask
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots()
color_data = 16.2+y[new_zoom]/1000
scatter = ax.scatter(eta[new_zoom], z[new_zoom], c = color_data, cmap='spring', s=s, rasterized=True)
cb = plt.colorbar(scatter)
etaa=np.linspace(0, 180, 500)
ax.plot(etaa, 600*(1+np.abs(np.cos(etaa*np.pi/180))))
cb.set_label('y ($\\mu$m)')
#cb.set_ticks([0.0, 1.0])
#cb.set_ticklabels(['0', '90'])
ax.set_ylabel('z (nm)')
ax.set_xlabel('$\\eta$ $(^{\\circ})$')
#plt.ylim((400,1800))
plt.grid()
plt.show()
#%% planarity of the orientations (mask)
# orientation tensor T = <v v^T> of the dipole directions v = (sin eta cos rho, sin eta sin rho, cos eta) of the
# localisations of mask (functions of bio_data_analysis/analyse_SMOLM_noyau_simu_functions.py, as for the nuclei).
# Eigenvalues l1 >= l2 >= l3: P = 1 - l3/l1 (0 isotropic, 1 all the dipoles in one plane), P2 = (l1 + l2)/(l1 + l2 + l3)
# (2/3 isotropic, 1 planar); the normal of the plane is the eigenvector of l3, given as (eta, rho) of the fit.
# Local planarity: the same with the neighbours of each localisation within planarity_scale nm in 3D (at least
# min_neighbors). P of a few vectors is > 0 even without any order (finite sample), so it is compared with the same
# computation after shuffling the (rho, eta) pairs among the localisations of mask (same distribution, same numbers of
# neighbours, no spatial order), and among the localisations of blocks of block_xy x block_xy x block_z nm (keeps an
# orientation that changes slowly over the field, e.g. a bias with x, y or z). Without local order, 5 % of the points
# are above the 95th percentile of the shuffled P.
# Run it after the correction doublons cell: the same molecule seen on several frames would look like local order.
sys.path.append('/mnt/c/Users/Amaury/Documents/GitHub/bio_data_analysis')
from analyse_SMOLM_noyau_simu_functions import orientation_planarity, local_planarity, cart2sph
planarity_scales = [200., 400.]  # nm, radius of the neighbourhood
min_neighbors = 10
block_xy, block_z = 1000., 250.  # nm, blocks of the second shuffle
P_name = 'P'  # local measure shown: 'P2' = 1 - l3 (2/3 isotropic, 1 planar) or 'P' = 1 - l3/l1 (0 isotropic, 1 planar)
P_index, P_floor = {'P': (0, 0.), 'P2': (1, 2/3)}[P_name] # output of local_planarity, value without order (isotropic)
# colours of the maps: turbo (blue -> red, not the hsv of rho), stretched so that it changes colour quickly around
# the median of the local P of each map (green): P_low_percentile -> median blue -> green, median -> 1 green -> red
P_low_percentile = 2

sel_p = mask & np.isfinite(x) & np.isfinite(rho) & np.isfinite(eta)
xp, yp, zp, rhop, etap = x[sel_p], y[sel_p], z[sel_p], rho[sel_p] % 180, eta[sel_p]
P_g, P2_g, normal_g, ev_g, _ = orientation_planarity(rhop, etap, verbose=False)
normal_eta, normal_rho = np.rad2deg(cart2sph(*normal_g))
print(f'{sel_p.sum()} localisations; global planarity P {P_g:.3f}, P2 {P2_g:.3f}, eigenvalues {np.round(ev_g, 3)}; '
      f'normal eta {normal_eta:.0f}, rho {normal_rho:.0f} deg')

rng = np.random.default_rng(0)
def shuffle_in_groups(groups):
    # (rho, eta) pairs permuted among the localisations of each group
    order = np.argsort(groups, kind='stable')
    perm = np.empty(len(groups), dtype=int)
    for g in np.split(order, np.flatnonzero(np.diff(groups[order])) + 1):
        perm[g] = rng.permutation(g)
    return rhop[perm], etap[perm]
block_id = np.unique(np.column_stack((xp//block_xy, yp//block_xy, zp//block_z)), axis=0, return_inverse=True)[1].ravel()

P_local = {} # scale: (P_name of the data, shuffled in mask, shuffled in blocks)
for scale in planarity_scales:
    P_local[scale] = []
    for rho_, eta_ in [(rhop, etap), shuffle_in_groups(np.zeros(len(xp), dtype=int)), shuffle_in_groups(block_id)]:
        P_local[scale].append(local_planarity(xp, yp, zp, eta_, rho_, scale=scale, min_neighbors=min_neighbors)[P_index])
    Pd, Ps, Pb = P_local[scale]
    ok = np.isfinite(Pd)
    print(f'local {P_name}, {scale:.0f} nm: {ok.sum()} points with >= {min_neighbors} neighbours, median {np.median(Pd[ok]):.3f}; '
          f'shuffled in mask {np.median(Ps[ok]):.3f} (above its 95th percentile {np.mean(Pd[ok] > np.percentile(Ps[ok], 95)):.3f}), '
          f'shuffled in blocks {np.median(Pb[ok]):.3f} (above its 95th percentile {np.mean(Pd[ok] > np.percentile(Pb[ok], 95)):.3f})')

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(len(planarity_scales), 2, figsize=(12, 7), squeeze=False)
for row, scale in enumerate(planarity_scales):
    Pd, Ps, Pb = P_local[scale]
    ok = np.isfinite(Pd)
    a = ax[row, 0]
    a.set_facecolor('black')
    P_centre, P_min = np.median(Pd[ok]), min(np.percentile(Pd[ok], P_low_percentile), np.median(Pd[ok]) - 0.01)
    # turbo resampled so that its middle (green) falls on the median: linear colour bar from P_min to 1
    P_axis = np.linspace(P_min, 1., 256)
    P_cmap = mcolors.ListedColormap(plt.get_cmap('turbo')(np.interp(P_axis, [P_min, P_centre, 1.], [0., 0.5, 1.])))
    sc = a.scatter(xp[ok]/1000, yp[ok]/1000, c=Pd[ok], cmap=P_cmap, vmin=P_min, vmax=1., s=s, rasterized=True)
    a.set_aspect('equal')
    cb = plt.colorbar(sc, ax=a, fraction=0.046, label=f'local {P_name}')
    cb.set_ticks(np.round([P_min, P_centre, 1.], 2))
    a.set(xlabel='x ($\\mu$m)', ylabel='y ($\\mu$m)', title=f'local planarity {P_name}, {scale:.0f} nm')
    a = ax[row, 1]
    bins_p = np.linspace(P_floor, 1, 51)
    for values, label, ls in [(Pd, 'data', '-'), (Ps, 'shuffled in mask', ':'), (Pb, 'shuffled in blocks', '--')]:
        a.hist(values[ok], bins=bins_p, density=True, histtype='step', ls=ls, lw=2, label=label)
    a.set(xlabel=f'local {P_name}', ylabel='density', title=f'{ok.sum()} points with >= {min_neighbors} neighbours')
    a.axvline(P_centre, color='gray', lw=0.8, label=f'median {P_centre:.2f} (green on the map)')
    a.legend(fontsize=8, loc='best')
fig.suptitle(f'{sel_p.sum()} localisations, global P {P_g:.3f}, P2 {P2_g:.3f}, normal $\\eta$ {normal_eta:.0f}°, $\\rho$ {normal_rho:.0f}°')
plt.tight_layout()
plt.show()
#%% bead fit
import torch
new_zoom = mask
def loss__(x,y,radius,centerx, centery):
    return torch.sum(((x-centerx)**2 + (y-centery)**2-radius**2)**2)
def find_params(xxxx, yyyy):
    params = torch.tensor([1000.,8800.,-15400.], requires_grad=True)#  ,16000, 7800
    optimizer = torch.optim.Adam([params], lr=50)
    loss0 = []
    for i in tqdm(range(500)):
        optimizer.zero_grad()  # Reset gradients
        loss = loss__(torch.tensor(xxxx), torch.tensor(yyyy), params[0], params[1], params[2])
        loss0.append(loss.detach().numpy())
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
    fig, ax = plt.subplots()
    ax.plot(loss0)
    plt.show()
    return params.detach().numpy()
    
def fit_circle(xx, yy):
    ''' closed-form least squares circle fit (Kasa): minimizes the same loss as loss__, exactly, without iterations.
    returns [radius, centerx, centery] as find_params '''
    A = np.c_[2*xx, 2*yy, np.ones_like(xx)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, xx**2 + yy**2, rcond=None)
    return np.array([np.sqrt(c + cx**2 + cy**2), cx, cy])

#params = find_params(x[new_zoom], y[new_zoom])
r_frac = 0.7  # keep only the localizations near the equator of the bead: r > r_frac*R
params = fit_circle(x[mask], y[mask])  # 1st fit on the whole bead (projection = filled disk) -> center
r_all = np.sqrt((x-params[1])**2 + (y-params[2])**2)
R_bead = np.percentile(r_all[mask], 95)  # apparent radius of the bead
new_zoom = mask & (r_all > r_frac*R_bead)
params = fit_circle(x[new_zoom], y[new_zoom])  # 2nd fit on the equatorial ring only (points really on a circle)
r_all = np.sqrt((x-params[1])**2 + (y-params[2])**2)
new_zoom = mask & (r_all > r_frac*R_bead)
print(f'center = ({params[1]:.1f}, {params[2]:.1f}) nm, R = {params[0]:.1f} nm, '
      f'{new_zoom.sum()}/{mask.sum()} localizations kept (r > {r_frac}*{R_bead:.0f} nm)')
rho_th = np.arctan2(y[new_zoom]-params[2], x[new_zoom]-params[1])*(180/np.pi)%180
th = np.linspace(0,2*np.pi,100)
line = np.arctan2(params[1]+(200+params[0])*np.sin(th)-params[1], params[2]+(200+params[0])*np.cos(th)-params[2])*(180/np.pi)%180.
huesL = line / 180
hsv_colorsL = np.stack((huesL, np.ones_like(huesL), np.ones_like(huesL)), axis=1)
rgb_colorsL = hsv_to_rgb(hsv_colorsL)
#%% plot of the bead fit
plt.rcParams['figure.figsize'] = [6, 6]
fig, ax = plt.subplots()
hues = rho[new_zoom] / 180
hsv_colors = np.stack((hues, np.ones_like(hues), np.ones_like(hues)), axis=1)
rgb_colors = hsv_to_rgb(hsv_colors)
ax.scatter(x[new_zoom], y[new_zoom], c=rgb_colors, s=10)
plt.axis('equal')
ax.scatter(params[1], params[2], marker='x', s=50, rasterized=True)
ax.scatter(params[1]+(200+params[0])*np.cos(th), params[2]+(200+params[0])*np.sin(th), c=rgb_colorsL)
ax.axhline(params[2])
ax.axvline(params[1])
plt.show()

plt.rcParams['figure.figsize'] = [6, 6]
fig, ax = plt.subplots()
hues = rho_th / 180
hsv_colors = np.stack((hues, np.ones_like(hues), np.ones_like(hues)), axis=1)
rgb_colors = hsv_to_rgb(hsv_colors)
ax.scatter(x[new_zoom], y[new_zoom], c=rgb_colors, s=10)
plt.axis('equal')
ax.scatter(params[1], params[2], marker='x', s=50, rasterized=True)
ax.scatter(params[1]+(200+params[0])*np.cos(th), params[2]+(200+params[0])*np.sin(th), c=rgb_colorsL)
ax.axhline(params[2])
ax.axvline(params[1])
plt.show()

#%% rho bias 
plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [2, 1]},
    constrained_layout=True
)
delta_rho = (rho[new_zoom] - rho_th + 90) % 180 - 90 
# TOP: scatter
ax[0].scatter(
    rho_th,
    delta_rho,
    s=4,
    alpha=1,
    c=rgb_colors,
    rasterized=True
)
ax[0].set_ylabel(r'$\Delta\rho$ ($^\circ$)')
ax[0].set_ylim(-60, 60)
ax[0].set_xlim(0, 180)
ax[0].grid()
# BOTTOM: IQR boxes
rr = np.linspace(0, 180, 37)  # bins every 5°
centers = rr[:-1] + 2.5
data_boxes = []
centers_valid = []
for i in range(36):
    values = delta_rho[
        (rho_th > rr[i]) & (rho_th < rr[i+1])
    ]
    values = values[np.abs(values)<40]  # outliers
    if len(values) > 0:
        data_boxes.append(values)
        centers_valid.append(centers[i])
ax[1].boxplot(
    data_boxes,
    positions=centers_valid,

    widths=4,
    showfliers=False,
    patch_artist=True,
    # IQR box
    boxprops=dict(
        facecolor='white',
        edgecolor='black',
        alpha=0.5,
        linewidth=1.2
    ),
    # Median
    medianprops=dict(
        color='black',
        linewidth=2
    ),
    # No whiskers/caps
    whiskerprops=dict(color='none'),
    capprops=dict(color='none')
)
ax[1].axhline(0., c='r')
ax[1].set_xlabel(r'$\rho_{\mathrm{theory}}$ ($^\circ$)')
ax[1].set_ylabel(r'$\Delta\rho$ ($^\circ$)')
ax[1].set_ylim(-10, 10)
ax[1].set_xlim(0, 180)
ax[1].set_xticks(np.linspace(0, 180, 7), ['0', '30', '60', '90', '120', '150', '180'])
ax[1].grid()
plt.show()

#%% harmonics of the rho bias and of delta vs rho_theory (bead): which calibration error
# Delta rho(rho) and delta(rho) fitted with c0 + a2 cos 2rho + b2 sin 2rho + a4 cos 4rho + b4 sin 4rho (least squares,
# |Delta rho| > 40 deg left out as in the boxes, uncertainties by bootstrap over the localisations). Signatures:
#   constant term of Delta rho ............ common rotation (rho_offset)
#   180 deg period (2 rho) of Delta rho ... x/y balance (normalisation, y gain): sin 2rho behind J1 (planes 0, 2),
#                                            cos 2rho behind J2 (plane 1); in delta the other one (cos / sin)
#   90 deg period (4 rho) ................. relative rotation of the J1 and J2 paths (rotation, rotation2) and/or a
#                                            different contrast in J1 and J2 (depolarisation of plane 1)
def harmonics(theta_deg, values, n_boot=300, seed=0):
    t = np.deg2rad(theta_deg)
    A = np.column_stack([np.ones_like(t), np.cos(2*t), np.sin(2*t), np.cos(4*t), np.sin(4*t)])
    coef = np.linalg.lstsq(A, values, rcond=None)[0]
    rng_h = np.random.default_rng(seed)
    boot = np.array([np.linalg.lstsq(A[i], values[i], rcond=None)[0]
                     for i in (rng_h.integers(0, len(t), len(t)) for _ in range(n_boot))])
    return coef, boot.std(axis=0), A

dr = (rho[new_zoom] - rho_th + 90) % 180 - 90
keep_h = np.abs(dr) < 40
harm = {'Delta rho': harmonics(rho_th[keep_h], dr[keep_h]),
        'delta': harmonics(rho_th, delta[new_zoom])}
names_h = ['constant', 'cos 2rho', 'sin 2rho', 'cos 4rho', 'sin 4rho']
for name, (coef, err, _) in harm.items():
    print(f'{name} vs rho_theory ({len(_)} localisations):')
    for n_, c_, e_ in zip(names_h, coef, err):
        print(f'   {n_:9s} {c_:+6.2f} +/- {e_:4.2f} deg' + ('   *' if abs(c_) > 2*e_ and n_ != 'constant' else ''))
    for period, (ia, ib) in (('180 deg period (2 rho)', (1, 2)), ('90 deg period (4 rho)', (3, 4))):
        amp = np.hypot(coef[ia], coef[ib]); n_harm = 2 if ia == 1 else 4
        phase = np.rad2deg(np.arctan2(coef[ib], coef[ia]))/n_harm % (360/n_harm) # a cos(n rho) + b sin(n rho) maximum
        print(f'   {period}: amplitude {amp:.2f} deg, maximum at rho = {phase:.0f} deg (+ k*{360//n_harm})')
print('* : coefficient larger than twice its uncertainty')

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
th_fine = np.linspace(0, 180, 361)
A_fine = np.column_stack([np.ones_like(th_fine), np.cos(np.deg2rad(2*th_fine)), np.sin(np.deg2rad(2*th_fine)),
                          np.cos(np.deg2rad(4*th_fine)), np.sin(np.deg2rad(4*th_fine))])
for a_, (name, theta_h, vals_h) in zip(ax, (('Delta rho', rho_th[keep_h], dr[keep_h]), ('delta', rho_th, delta[new_zoom]))):
    coef = harm[name][0]
    centres_h = np.arange(2.5, 180, 5)
    med_h = [np.median(vals_h[np.abs(theta_h - c_) < 2.5]) if np.any(np.abs(theta_h - c_) < 2.5) else np.nan for c_ in centres_h]
    a_.plot(centres_h, med_h, 'ko', ms=4, label='median per 5 deg')
    a_.plot(th_fine, A_fine @ coef, 'r-', lw=2, label='fit (all terms)')
    a_.plot(th_fine, coef[0] + A_fine[:, 1:3] @ coef[1:3], 'C0--', label='constant + 180 deg period')
    a_.plot(th_fine, coef[0] + A_fine[:, 3:5] @ coef[3:5], 'C2--', label='constant + 90 deg period')
    a_.set_ylabel(('$\\Delta\\rho$' if name == 'Delta rho' else '$\\delta$') + ' ($^\\circ$)'); a_.grid(alpha=0.4)
ax[0].axhline(0, color='k', lw=0.8); ax[0].legend(fontsize=9, ncol=2)
ax[1].set_xlabel(r'$\rho_{\mathrm{theory}}$ ($^\circ$)'); ax[1].set_xlim(0, 180)
ax[1].set_xticks(np.linspace(0, 180, 7))
plt.show()

#%% check of the bead center: delta rho vs the full azimuth phi (0-360)
# a center error d gives delta_rho ~ (d/r) sin(phi - phi0): a 360 periodic sinusoid, which cancels in the
# rho bias plot (phi and phi+180 have the same rho_th) but is visible here. Flat = good center
phi_th = np.arctan2(y[new_zoom]-params[2], x[new_zoom]-params[1])*(180/np.pi)%360
delta_rho = (rho[new_zoom] - rho_th + 90) % 180 - 90
ok = np.abs(delta_rho) < 40  # outliers
# least squares fit delta_rho = a cos(phi) + b sin(phi) + c
A = np.c_[np.cos(np.deg2rad(phi_th[ok])), np.sin(np.deg2rad(phi_th[ok])), np.ones(ok.sum())]
(a, b, c), *_ = np.linalg.lstsq(A, delta_rho[ok], rcond=None)
amp = np.hypot(a, b)
print(f'360-periodic component: {amp:.2f} deg -> fitted center off by ~ {np.deg2rad(amp)*params[0]:.1f} nm '
      f'towards phi = {(np.degrees(np.arctan2(b, a))+90)%360:.0f} deg, mean delta rho = {c:.2f} deg')
plt.rcParams['figure.figsize'] = [8, 4]
fig, ax = plt.subplots()
ax.scatter(phi_th, delta_rho, s=4, c=rgb_colors, rasterized=True)
pp = np.linspace(0, 360, 361)
ax.plot(pp, a*np.cos(np.deg2rad(pp)) + b*np.sin(np.deg2rad(pp)) + c, c='k', lw=2)
bins_phi = np.linspace(0, 360, 37)
med = [np.median(delta_rho[ok & (phi_th >= bins_phi[i]) & (phi_th < bins_phi[i+1])])
       if np.any(ok & (phi_th >= bins_phi[i]) & (phi_th < bins_phi[i+1])) else np.nan for i in range(36)]
ax.plot(bins_phi[:-1] + 5, med, 'o', c='k', ms=4)
ax.axhline(0., c='r')
ax.set_xlabel(r'$\phi$ (azimuth on the bead, $^\circ$)')
ax.set_ylabel(r'$\Delta\rho$ ($^\circ$)')
ax.set_xlim(0, 360)
ax.set_ylim(-60, 60)
ax.set_xticks(np.linspace(0, 360, 7))
ax.grid()
plt.show()

 #%% delta bias
plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [2, 1]},
    constrained_layout=True
)
# Data
delta_vals = delta[new_zoom]
# KDE density
xy = np.vstack([rho_th, delta_vals])
kde = gaussian_kde(xy)
density = kde(xy)
density_norm = (
    (density - density.min())
    / (density.max() - density.min())
)
rgb_colors_array = np.array(rgb_colors)
rgba_colors = np.column_stack([
    rgb_colors_array,
    density_norm
])
# TOP: scatter
ax[0].scatter(
    rho_th,
    delta_vals,
    s=10,
    c=rgb_colors,
    linewidths=0,
    rasterized=True,
    alpha=1
)
# Reference lines
ax[0].axhline(
    50,
    color='gray',
    linestyle='--'
)
ax[0].axhline(
    150,
    color='gray',
    linestyle='--'
)
ax[0].set_ylim(30, 160)
ax[0].set_xlim(0, 180)
ax[0].set_ylabel(r'$\delta$ ($^\circ$)')
ax[0].grid()
# BOTTOM: IQR boxes
rr = np.linspace(0, 180, 37)  # bins every 5°
centers = rr[:-1] + 2.5
data_boxes = []
centers_valid = []
for i in range(36):
    values = delta_vals[
        (rho_th > rr[i]) &
        (rho_th < rr[i + 1])
    ]

    if len(values) > 0:
        data_boxes.append(values)
        centers_valid.append(centers[i])
ax[1].boxplot(
    data_boxes,
    positions=centers_valid,
    widths=4,
    showfliers=False,
    patch_artist=True,
    # IQR box
    boxprops=dict(
        facecolor='white',
        edgecolor='black',
        alpha=0.7,
        linewidth=1.2
    ),
    # Median
    medianprops=dict(
        color='black',
        linewidth=2
    ),
    # Remove whiskers and caps
    whiskerprops=dict(color='none'),
    capprops=dict(color='none')
)
ax[1].set_ylim(50, 150)
#ax[1].set_xlim(0, 180)
ax[1].set_xlabel(r'$\rho_{\mathrm{theory}}$ ($^\circ$)')
ax[1].set_ylabel(r'$\delta$ ($^\circ$)')
ax[1].set_xticks(np.linspace(0, 180, 7), ['0', '30', '60', '90', '120', '150', '180'])
ax[1].grid()
plt.show()
# %%function for the pca

from scipy import stats
def pca_xy(x, y, z):
    """
    PCA on the xy plane of a 3D point cloud.
    Returns coordinates in the PCA frame.
    """
    # stack xy coordinates
    points_xy = np.column_stack([x, y])  # shape (N, 2)
    
    # center the data
    mean_xy = np.mean(points_xy, axis=0)
    points_centered = points_xy - mean_xy
    
    # compute covariance matrix and PCA
    cov = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # project onto PCA axes
    w = points_centered @ eigenvectors  # shape (N, 2): w[:,0] is PC1, w[:,1] is PC2
    
    return w, eigenvectors, eigenvalues, mean_xy

def theil_sen_slope(xx, yy, n_max=3000, seed=0):
    # robust slope (median of the slopes between pairs of points) with its 95 % confidence interval.
    # The number of pairs grows as n^2: above n_max points, computed on a fixed random sample of n_max points
    if len(xx) > n_max:
        keep = np.random.default_rng(seed).choice(len(xx), n_max, replace=False)
        xx, yy = xx[keep], yy[keep]
    slope, intercept, low, high = stats.theilslopes(yy, xx)
    return slope, intercept, low, high

# usage
w, eigenvectors, eigenvalues, mean_xy = pca_xy(x[mask], y[mask], z[mask])

# w[:,0] — coordinate along first principal axis (most variance)
# w[:,1] — coordinate along second principal axis
print(f"Principal axis direction: {eigenvectors[:,0]}")
print(f"Explained variance: {eigenvalues/eigenvalues.sum()*100}")

# %% lasso selection on z along the PCA axis (refines mask, then the PCA is recomputed)
# z vs the position along the first PCA axis of the points of mask: circle the points to keep (e.g. one filament,
# without the clusters or the points of another structure). mask is reduced to them and w, eigenvectors are
# recomputed, so that the orientation cells below use the selection. mask_before_pca_lasso keeps the previous mask
%matplotlib widget
mask_before_pca_lasso = mask.copy()
plt.rcParams['figure.figsize'] = [7, 4]
fig, ax = plt.subplots()
ax.scatter(w[:,0]/1000, z[mask]/1000, s=1, c=frame[mask], cmap='coolwarm')
ax.set_xlabel('PCA axis ($\\mu$m)')
ax.set_ylabel('z ($\\mu$m)')
ax.set_title('circle the points to keep')
pca_points = np.column_stack((w[:,0]/1000, z[mask]/1000))
mask_indices = np.where(mask)[0] # index in the full arrays of each point of the plot

def onselect_pca(verts):
    global mask, w, eigenvectors, eigenvalues, mean_xy
    inside = Path(verts).contains_points(pca_points)
    mask = np.zeros(len(x), dtype=bool)
    mask[mask_indices[inside]] = True
    w, eigenvectors, eigenvalues, mean_xy = pca_xy(x[mask], y[mask], z[mask])
    print(f'{inside.sum()} points kept out of {len(inside)}, PCA recomputed')

lasso_pca = LassoSelector(ax, onselect_pca)
plt.show()

# %% orientation of filaments
from scipy import stats

plt.rcParams['figure.figsize'] = [5, 9]
plt.rcParams.update({'font.size': 13})

fig = plt.figure()
ax0 = fig.add_subplot(3, 1, 1)
ax_xy = fig.add_subplot(3, 1, 2)
ax1 = fig.add_subplot(3, 1, 3, projection='polar')

# the direction of a PCA axis is arbitrary, it is chosen with an in-plane angle in [0, 180[ as rho,
# so that the tilt of the PCA has the same sign convention as the tilt from eta
if eigenvectors[1, 0] < 0:
    eigenvectors[:, 0] = -eigenvectors[:, 0]
    w[:, 0] = -w[:, 0]

# --- linear regression ---
# robust line (Theil-Sen) of z along the PCA axis, with its 95 % confidence interval
slope, intercept, slope_low, slope_high = theil_sen_slope(w[:,0]/1000, z[mask]/1000)
slope_deg = np.rad2deg(np.arctan(slope))
print(f"Slope (Theil-Sen): {slope_deg:.2f}° [95 % CI {np.rad2deg(np.arctan(slope_low)):.2f}, {np.rad2deg(np.arctan(slope_high)):.2f}]")

x_fit = np.linspace(w[:,0].min()/1000, w[:,0].max()/1000, 500)
y_fit = slope * x_fit + intercept

# tilt of the dipoles from eta, with the same convention as the polar plot: tilt = 90 - eta
tilt_median = 90 - np.median(eta[mask])
tilt_mean = 90 - np.mean(eta[mask])
# the lines of the tilts from eta go through the centre of the data, as the regression line
w_mean, z_mean = np.mean(w[:,0])/1000, np.mean(z[mask])/1000
lines = [(slope_deg, 'red', '-', 'PCA'),
         (tilt_median, 'k', '-', '$\\eta$ median'),
         (tilt_mean, 'k', '--', '$\\eta$ mean')]

# --- scatter plot w vs z ---
# points coloured by eta, with the colormap and the range of the plot eta cell
eta_norm = plt.Normalize(30, 150)
sc0 = ax0.scatter(w[:,0]/1000, z[mask]/1000, s=0.1, c=eta[mask], cmap=cmap, norm=eta_norm, alpha=0.5, rasterized=True)
fig.colorbar(sc0, ax=ax0, label='$\\eta$ (°)').solids.set_alpha(1) # opaque colorbar despite the transparent points
for tilt, color, style, name in lines:
    ax0.plot(x_fit, z_mean + np.tan(np.deg2rad(tilt))*(x_fit - w_mean), color=color, linestyle=style, linewidth=2,
             label=f'{name}: {tilt:.2f}°')
ax0.set_xlabel('PCA axis ($\\mu$m)')
ax0.set_ylabel('z ($\\mu$m)')
ax0.legend()
ax0.grid(True, alpha=0.4)

# --- xy scatter with PCA axis direction ---
sc_xy = ax_xy.scatter(x[mask]/1000, y[mask]/1000, s=0.1, c=eta[mask], cmap=cmap, norm=eta_norm, alpha=0.3, rasterized=True)
fig.colorbar(sc_xy, ax=ax_xy, label='$\\eta$ (°)').solids.set_alpha(1)

# PCA axis direction (eigenvectors[:,0] is the first PC)
cx = np.mean(x[mask]/1000)
cy = np.mean(y[mask]/1000)
scale_arrow = (x[mask].max() - x[mask].min()) / 2000 * 0.4  # 40% of the data range

# first PC direction
pc1 = eigenvectors[:, 0]  # from your PCA
ax_xy.annotate('', xy=(cx + pc1[0]*scale_arrow, cy + pc1[1]*scale_arrow),
                xytext=(cx - pc1[0]*scale_arrow, cy - pc1[1]*scale_arrow),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax_xy.set_aspect('equal')
ax_xy.set_xlabel('x ($\\mu$m)')
ax_xy.set_ylabel('y ($\\mu$m)')
ax_xy.grid(True, alpha=0.4)
ax_xy.set_title('PCA axis in xy plane')

# --- polar histogram ---
eta_vals = eta[mask]
eta_rad = np.deg2rad(eta_vals)
eta_mirrored = np.concatenate([eta_rad, eta_rad + np.pi])

bins = np.linspace(0, 2*np.pi, 201)
counts, edges = np.histogram(eta_mirrored, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]

norm = plt.Normalize(30, 150)
centers_folded = centers % np.pi
colors = cmap(norm(np.rad2deg(centers_folded)))
ax1.bar(centers, counts, width=width, bottom=0, color=colors, edgecolor='none', alpha=0.95)

# same three lines as in the first plot, at eta = 90 - tilt, mirrored as the histogram
for tilt, color, style, name in lines:
    eta_line = np.deg2rad(90 - tilt)
    ax1.axvline(eta_line, color=color, linestyle=style, linewidth=2, label=f'{name}: {tilt:.2f}°')
    ax1.axvline(eta_line + np.pi, color=color, linestyle=style, linewidth=2)

ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_facecolor('white')
ax1.spines['polar'].set_color('black')
ax1.tick_params(axis='x', colors='black')
ax1.tick_params(axis='y', colors='black')
ax1.yaxis.label.set_color('black')

for label in ax1.get_xticklabels():
    label.set_color('black')
for label in ax1.get_yticklabels():
    label.set_color('black')

ax1.set_thetagrids(np.arange(0, 360, 60), labels=[f"{d}°" for d in np.arange(0, 360, 60)])
ax1.set_ylabel("Count", labelpad=25)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=9)

# %% orientation of filaments, compact: z along the PCA axis (equal scales) and polar histogram of eta
# same computations as the cell above (to run after it or after the pca cell). The legend of the slopes is placed
# outside the plots, above them, by the constrained layout: it keeps its room when the figure is resized
from scipy import stats
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(6, 8), layout='constrained')
gs = GridSpec(2, 1, height_ratios=[0.6, 1.3], figure=fig)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], projection='polar')

# direction of the PCA axis with an in-plane angle in [0, 180[ as rho (same sign convention as the tilt from eta)
if eigenvectors[1, 0] < 0:
    eigenvectors[:, 0] = -eigenvectors[:, 0]
    w[:, 0] = -w[:, 0]

slope, intercept, slope_low, slope_high = theil_sen_slope(w[:,0]/1000, z[mask]/1000) # robust line, 95 % CI
slope_deg = np.rad2deg(np.arctan(slope))
x_fit = np.linspace(w[:,0].min()/1000, w[:,0].max()/1000, 500)
tilt_median = 90 - np.median(eta[mask])
tilt_mean = 90 - np.mean(eta[mask])
w_mean, z_mean = np.mean(w[:,0])/1000, np.mean(z[mask])/1000
lines = [(slope_deg, 'red', '-', 'PCA'),
         (tilt_median, 'k', '-', '$\\eta$ median'),
         (tilt_mean, 'k', '--', '$\\eta$ mean')]

# --- z along the PCA axis, equal scales ---
eta_norm = plt.Normalize(30, 150) # colours of the histogram
ax0.scatter(w[:,0]/1000, z[mask]/1000, s=0.1, color='steelblue', alpha=0.5, rasterized=True)
handles = []
for tilt, color, style, name in lines:
    handles += ax0.plot(x_fit, z_mean + np.tan(np.deg2rad(tilt))*(x_fit - w_mean), color=color, linestyle=style,
                        linewidth=2, label=f'{name}: {tilt:.2f}°')
ax0.set_aspect('equal')
ax0.set_xlabel('PCA axis ($\\mu$m)')
ax0.set_ylabel('z ($\\mu$m)')
ax0.grid(True, alpha=0.4)

# --- legend of the slopes, outside the plots, on top ---
legend = fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), fontsize=11)
legend_width = legend.get_window_extent(fig.canvas.get_renderer()).width # width with the entries on one row
def fit_legend(event=None):
    # the entries on one row if they fit in the width of the figure, else one per row (called at each resize).
    # The legend is rebuilt: changing its number of columns afterwards does not move the entries
    global legend
    ncol = len(handles) if legend_width < 0.98*fig.bbox.width else 1
    if ncol != legend._ncols:
        legend.remove()
        legend = fig.legend(handles=handles, loc='outside upper center', ncol=ncol, fontsize=11)
fig.canvas.mpl_connect('resize_event', fit_legend)
fit_legend()

# --- polar histogram of eta, same lines, no legend ---
eta_rad = np.deg2rad(eta[mask])
counts, edges = np.histogram(np.concatenate([eta_rad, eta_rad + np.pi]), bins=np.linspace(0, 2*np.pi, 201))
centers = (edges[:-1] + edges[1:]) / 2
ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=cmap(eta_norm(np.rad2deg(centers % np.pi))),
        edgecolor='none', alpha=0.95)
for tilt, color, style, name in lines:
    eta_line = np.deg2rad(90 - tilt)
    ax1.axvline(eta_line, color=color, linestyle=style, linewidth=2)
    ax1.axvline(eta_line + np.pi, color=color, linestyle=style, linewidth=2)
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_thetagrids(np.arange(0, 360, 60), labels=[f"{d}°" for d in np.arange(0, 360, 60)])
ax1.set_ylabel("Count", labelpad=25)

plt.show()
# %% orientation of filaments in the plane: rho of the PCA axis vs the fitted rho (precision and consistency)
# same as the compact eta cell, for rho: the in-plane angle of the first PCA axis (from x, counterclockwise, in
# [0, 180[ as rho) is compared with the rho of the localisations of mask. rho is an axis (rho and rho+180 are the
# same), so its mean is the circular mean of 2 rho / 2 and the differences are wrapped in [-90, 90[
from matplotlib.gridspec import GridSpec

def axial_mean(angles_deg):
    # mean of an axis in [0, 180[ and its circular std (deg), from the mean resultant of 2*angle
    r = np.mean(np.exp(2j*np.deg2rad(angles_deg)))
    return np.rad2deg(np.angle(r))/2 % 180, np.rad2deg(np.sqrt(-2*np.log(np.abs(r))))/2

def wrap90(d):
    return (d + 90) % 180 - 90

rho_pca = np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) % 180
# 95 % confidence interval of the PCA axis: bootstrap over the points of mask
rng = np.random.default_rng(0)
xy_mask = np.column_stack((x[mask], y[mask]))
boot = []
for _ in range(500):
    pick = xy_mask[rng.integers(0, len(xy_mask), len(xy_mask))]
    vals, vecs = np.linalg.eigh(np.cov((pick - pick.mean(0)).T))
    boot.append(wrap90(np.rad2deg(np.arctan2(vecs[1, -1], vecs[0, -1])) - rho_pca))
rho_pca_low, rho_pca_high = rho_pca + np.percentile(boot, [2.5, 97.5])

rho_mean, rho_circ_std = axial_mean(rho[mask])
d_rho = wrap90(rho[mask] - rho_pca) # rho of each localisation relative to the PCA axis
rho_median = (rho_pca + np.median(d_rho)) % 180
print(f'rho of the PCA axis: {rho_pca:.2f}° [95 % CI {rho_pca_low % 180:.2f}, {rho_pca_high % 180:.2f}], {np.sum(mask)} points')
print(f'rho fitted: median {rho_median:.2f}°, mean {rho_mean:.2f}°')
print(f'consistency, rho - rho_PCA: median {np.median(d_rho):.2f}°, mean {np.rad2deg(np.angle(np.mean(np.exp(2j*np.deg2rad(d_rho)))))/2:.2f}°')
print(f'precision, spread of rho around the PCA axis: robust std {1.4826*np.median(np.abs(d_rho - np.median(d_rho))):.2f}°, '
      f'circular std {rho_circ_std:.2f}°, std {np.std(d_rho, ddof=1):.2f}°')

lines_rho = [(rho_pca, 'red', '-', 'PCA'),
             (rho_median, 'k', '-', '$\\rho$ median'),
             (rho_mean, 'k', '--', '$\\rho$ mean')]

plt.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(6, 8), layout='constrained')
gs = GridSpec(2, 1, height_ratios=[1, 1.3], figure=fig)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], projection='polar')

# --- xy of the points with the three directions through their centre, equal scales ---
ax0.scatter(x[mask]/1000, y[mask]/1000, s=0.1, color='steelblue', alpha=0.5, rasterized=True)
cx, cy = np.mean(x[mask])/1000, np.mean(y[mask])/1000
half = 0.5*np.max(w[:, 0].max() - w[:, 0].min())/1000 # half length of the points along the PCA axis
handles_rho = []
for angle, color, style, name in lines_rho:
    u = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    handles_rho += ax0.plot([cx - half*u[0], cx + half*u[0]], [cy - half*u[1], cy + half*u[1]], color=color,
                            linestyle=style, linewidth=2, label=f'{name}: {angle:.2f}°')
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')
ax0.grid(True, alpha=0.4)

# --- legend of the directions, outside the plots, on top (one row if it fits, as in the eta cell) ---
legend_rho = fig.legend(handles=handles_rho, loc='outside upper center', ncol=len(handles_rho), fontsize=11)
legend_rho_width = legend_rho.get_window_extent(fig.canvas.get_renderer()).width
def fit_legend_rho(event=None):
    global legend_rho
    ncol = len(handles_rho) if legend_rho_width < 0.98*fig.bbox.width else 1
    if ncol != legend_rho._ncols:
        legend_rho.remove()
        legend_rho = fig.legend(handles=handles_rho, loc='outside upper center', ncol=ncol, fontsize=11)
fig.canvas.mpl_connect('resize_event', fit_legend_rho)
fit_legend_rho()

# --- polar histogram of rho (mirrored on 360 as rho and rho+180 are the same), same lines ---
rho_rad = np.deg2rad(rho[mask])
counts, edges = np.histogram(np.concatenate([rho_rad, rho_rad + np.pi]), bins=np.linspace(0, 2*np.pi, 201))
centers = (edges[:-1] + edges[1:]) / 2
ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=plt.cm.hsv((centers % np.pi) / np.pi),
        edgecolor='none', alpha=0.95)
for angle, color, style, name in lines_rho:
    ax1.axvline(np.deg2rad(angle), color=color, linestyle=style, linewidth=2)
    ax1.axvline(np.deg2rad(angle) + np.pi, color=color, linestyle=style, linewidth=2)
ax1.set_theta_zero_location('E') # as the rho cells: 0 along x, counterclockwise
ax1.set_theta_direction(1)
ax1.set_thetagrids(np.arange(0, 360, 60), labels=[f"{d}°" for d in np.arange(0, 360, 60)])
ax1.set_ylabel("Count", labelpad=25)

plt.show()
#%% rho of the filament vs a threshold on the deviance, interactive (inside mask)
# polar histogram of rho of the localisations of mask with a deviance below the threshold of the slider, with the rho
# of the PCA axis (fixed, cell above) and the median / mean rho of the selection; right: precision (robust std of
# rho - rho_PCA), consistency (median of rho - rho_PCA) and % of localisations kept vs the threshold.
# mask_dev keeps the current selection for the next cells
%matplotlib qt
dev_in = fit_quality[mask]
d_rho_all = wrap90(rho[mask] - rho_pca)
thr_values = np.linspace(np.nanpercentile(dev_in, 1), np.nanpercentile(dev_in, 99.5), 60)
def rho_stats(sel):
    d = d_rho_all[sel]
    return np.median(d), 1.4826*np.median(np.abs(d - np.median(d))), sel.mean()*100
curves = np.array([rho_stats(dev_in < t) if np.sum(dev_in < t) > 10 else (np.nan,)*3 for t in thr_values])

fig = plt.figure(figsize=(12, 6))
ax_p = fig.add_subplot(1, 2, 1, projection='polar')
ax_c = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(bottom=0.2, wspace=0.35)
ax_slider = fig.add_axes([0.2, 0.06, 0.6, 0.04])

# polar histogram (fixed bins, heights updated by the slider)
edges_p = np.linspace(0, 2*np.pi, 201)
centers_p = (edges_p[:-1] + edges_p[1:])/2
bars_p = ax_p.bar(centers_p, np.zeros(len(centers_p)), width=edges_p[1] - edges_p[0], bottom=0,
                  color=plt.cm.hsv((centers_p % np.pi)/np.pi), edgecolor='none', alpha=0.95)
for a_ in (np.deg2rad(rho_pca), np.deg2rad(rho_pca) + np.pi):
    ax_p.axvline(a_, color='red', lw=2, label='PCA' if a_ < np.pi else None)
line_med = [ax_p.axvline(0, color='k', lw=2, label='$\\rho$ median'), ax_p.axvline(0, color='k', lw=2)]
line_mean = [ax_p.axvline(0, color='k', lw=2, ls='--', label='$\\rho$ mean'), ax_p.axvline(0, color='k', lw=2, ls='--')]
ax_p.set_theta_zero_location('E'); ax_p.set_theta_direction(1)
ax_p.set_thetagrids(np.arange(0, 360, 60), labels=[f"{d}°" for d in np.arange(0, 360, 60)])
ax_p.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.15), fontsize=9)

# curves vs the threshold
ax_c.plot(thr_values, curves[:, 1], 'C0-', label='precision: robust std of $\\rho - \\rho_{PCA}$ (°)')
ax_c.plot(thr_values, curves[:, 0], 'C1-', label='consistency: median of $\\rho - \\rho_{PCA}$ (°)')
ax_c2 = ax_c.twinx()
ax_c2.plot(thr_values, curves[:, 2], 'C2--', label='% kept')
ax_c2.set_ylabel('% of the localisations of mask kept', color='C2')
ax_c.axhline(0, color='k', lw=0.6)
v_thr = ax_c.axvline(thr_values[-1], color='k', ls=':')
ax_c.set_xlabel('maximum deviance'); ax_c.set_ylabel('deg'); ax_c.grid(alpha=0.4)
ax_c.legend(loc='upper left', fontsize=8); ax_c2.legend(loc='lower right', fontsize=8)

dev_slider = Slider(ax_slider, 'max deviance', thr_values[0], thr_values[-1], valinit=thr_values[-1])

def update_dev(thr):
    global mask_dev
    sel = dev_in < thr
    mask_dev = np.zeros(len(x), dtype=bool); mask_dev[np.where(mask)[0][sel]] = True
    r_rad = np.deg2rad(rho[mask][sel])
    counts_p, _ = np.histogram(np.concatenate([r_rad, r_rad + np.pi]), bins=edges_p)
    for b_, c_ in zip(bars_p, counts_p):
        b_.set_height(c_)
    ax_p.set_ylim(0, max(counts_p.max(), 1)*1.05)
    if sel.sum() > 1:
        med, prec, kept = rho_stats(sel)
        r_median = (rho_pca + med) % 180
        r_mean = axial_mean(rho[mask][sel])[0]
        for l_, a_ in zip(line_med, (r_median, r_median + 180)):
            l_.set_xdata([np.deg2rad(a_)]*2)
        for l_, a_ in zip(line_mean, (r_mean, r_mean + 180)):
            l_.set_xdata([np.deg2rad(a_)]*2)
        ax_p.set_title(f'deviance < {thr:.2f}: {sel.sum()} localisations ({kept:.0f} %)\n'
                       f'$\\rho_{{PCA}}$ {rho_pca:.1f}°, median {r_median:.1f}°, mean {r_mean:.1f}°\n'
                       f'consistency {med:+.2f}°, precision {prec:.2f}°', fontsize=10)
    v_thr.set_xdata([thr, thr])
    fig.canvas.draw_idle()

dev_slider.on_changed(update_dev)
update_dev(dev_slider.val)
plt.show()
#%% rho of the filament in a window of eta, interactive (inside mask)
# polar histogram of rho of the localisations of mask with eta in a window of eta_bin degrees centred on the value
# of the slider, with the rho of the PCA axis (fixed, cell above) and the median / mean rho of the selection; right:
# precision (robust std of rho - rho_PCA), consistency (median of rho - rho_PCA) and % of localisations kept vs the centre of the window.
# mask_eta_bin keeps the current selection for the next cells (mask_dev of the cell above is not changed)
%matplotlib qt
eta_bin = 10 # width of the window of eta in degrees
eta_in = eta[mask]
d_rho_all_e = wrap90(rho[mask] - rho_pca)
thr_values_e = np.arange(eta_bin/2, 180 - eta_bin/2 + 1e-9, 2.5) # centres of the window
in_eta_bin = lambda c: np.abs(eta_in - c) <= eta_bin/2
def rho_stats_e(sel):
    d = d_rho_all_e[sel]
    return np.median(d), 1.4826*np.median(np.abs(d - np.median(d))), sel.mean()*100
curves_e = np.array([rho_stats_e(in_eta_bin(t)) if np.sum(in_eta_bin(t)) > 10 else (np.nan,)*3 for t in thr_values_e])

fig_e = plt.figure(figsize=(12, 6))
ax_p_e = fig_e.add_subplot(1, 2, 1, projection='polar')
ax_c_e = fig_e.add_subplot(1, 2, 2)
fig_e.subplots_adjust(bottom=0.2, wspace=0.35)
ax_slider_e = fig_e.add_axes([0.2, 0.06, 0.6, 0.04])

# polar histogram (fixed bins, heights updated by the slider)
edges_p_e = np.linspace(0, 2*np.pi, 201)
centers_p_e = (edges_p_e[:-1] + edges_p_e[1:])/2
bars_p_e = ax_p_e.bar(centers_p_e, np.zeros(len(centers_p_e)), width=edges_p_e[1] - edges_p_e[0], bottom=0,
                  color=plt.cm.hsv((centers_p_e % np.pi)/np.pi), edgecolor='none', alpha=0.95)
for a_ in (np.deg2rad(rho_pca), np.deg2rad(rho_pca) + np.pi):
    ax_p_e.axvline(a_, color='red', lw=2, label='PCA' if a_ < np.pi else None)
line_med_e = [ax_p_e.axvline(0, color='k', lw=2, label='$\\rho$ median'), ax_p_e.axvline(0, color='k', lw=2)]
line_mean_e = [ax_p_e.axvline(0, color='k', lw=2, ls='--', label='$\\rho$ mean'), ax_p_e.axvline(0, color='k', lw=2, ls='--')]
ax_p_e.set_theta_zero_location('E'); ax_p_e.set_theta_direction(1)
ax_p_e.set_thetagrids(np.arange(0, 360, 60), labels=[f"{d}°" for d in np.arange(0, 360, 60)])
ax_p_e.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.15), fontsize=9)

# curves vs the threshold
ax_c_e.plot(thr_values_e, curves_e[:, 1], 'C0-', label='precision: robust std of $\\rho - \\rho_{PCA}$ (°)')
ax_c_e.plot(thr_values_e, curves_e[:, 0], 'C1-', label='consistency: median of $\\rho - \\rho_{PCA}$ (°)')
ax_c2_e = ax_c_e.twinx()
ax_c2_e.plot(thr_values_e, curves_e[:, 2], 'C2--', label='% kept')
ax_c2_e.set_ylabel('% of the localisations of mask kept', color='C2')
ax_c_e.axhline(0, color='k', lw=0.6)
v_thr_e = ax_c_e.axvline(90, color='k', ls=':')
ax_c_e.set_xlabel(f'centre of the {eta_bin:g}° window of $\\eta$ (°)'); ax_c_e.set_ylabel('deg'); ax_c_e.grid(alpha=0.4)
ax_c_e.legend(loc='upper left', fontsize=8); ax_c2_e.legend(loc='lower right', fontsize=8)

eta_slider = Slider(ax_slider_e, 'eta centre', thr_values_e[0], thr_values_e[-1], valinit=90)

def update_eta(thr):
    global mask_eta_bin
    sel = in_eta_bin(thr)
    mask_eta_bin = np.zeros(len(x), dtype=bool); mask_eta_bin[np.where(mask)[0][sel]] = True
    r_rad = np.deg2rad(rho[mask][sel])
    counts_p, _ = np.histogram(np.concatenate([r_rad, r_rad + np.pi]), bins=edges_p_e)
    for b_, c_ in zip(bars_p_e, counts_p):
        b_.set_height(c_)
    ax_p_e.set_ylim(0, max(counts_p.max(), 1)*1.05)
    if sel.sum() > 1:
        med, prec, kept = rho_stats_e(sel)
        r_median = (rho_pca + med) % 180
        r_mean = axial_mean(rho[mask][sel])[0]
        for l_, a_ in zip(line_med_e, (r_median, r_median + 180)):
            l_.set_xdata([np.deg2rad(a_)]*2)
        for l_, a_ in zip(line_mean_e, (r_mean, r_mean + 180)):
            l_.set_xdata([np.deg2rad(a_)]*2)
        ax_p_e.set_title(f'eta in [{thr - eta_bin/2:.1f}, {thr + eta_bin/2:.1f}]°: {sel.sum()} localisations ({kept:.0f} %)\n'
                       f'$\\rho_{{PCA}}$ {rho_pca:.1f}°, median {r_median:.1f}°, mean {r_mean:.1f}°\n'
                       f'consistency {med:+.2f}°, precision {prec:.2f}°', fontsize=10)
    v_thr_e.set_xdata([thr, thr])
    fig_e.canvas.draw_idle()

eta_slider.on_changed(update_eta)
update_eta(eta_slider.val)
plt.show()
#%% examples of PSF (data, fit, residual) of the selection, in windows of rho
# localisations of mask_dev (or mask) in each window of rho_windows, n_examples drawn at random per window, figures of
# density_experimental.make_examples in <csv folder>/mask_examples/<window>/. The row index of the csv still points
# to a real fit after the drift correction and the linking (a linked molecule is kept on one of its rows): the
# frame, x, y of this row are read again from the csv and found in the npz of the run. The rho of the title is the
# one of this frame, the rho of the selection (mean over the linked frames) is printed next to it
import importlib
from pathlib import Path as Path_lib # Path is the one of matplotlib (lasso)
sys.path.insert(0, '/mnt/c/Users/Amaury/Documents/GitHub/4polarMFM_these')
import density_experimental as DE
importlib.reload(DE)
rho_windows = {'rho_180': (180, 5), 'rho_150': (150, 5)} # name: (centre, half width) in degrees
alt_rho = {'rho_180': {'rho': 150, 'eta': 90}} # name: extra row, the same fit with rho (and eta) replaced by these values
n_examples = 6
selection = mask_dev if 'mask_dev' in globals() else mask
if not isinstance(path, str) or not path: # askopenfilename gives () when the dialog is cancelled
    raise ValueError('path is not a csv file (file dialog cancelled?): set path = "<the fit csv loaded at the start>" '
                     'and run this cell again')
raw_csv = pd.read_csv(path, sep=';', comment='#') # x, y, frame of the session are drift corrected and linked
rng = np.random.default_rng(0)
groups = {}
for name, (centre, half) in rho_windows.items():
    rows = np.where(selection & (np.abs(wrap90(rho - centre)) <= half))[0]
    picked = np.sort(rng.choice(rows, min(n_examples, len(rows)), replace=False))
    groups[name] = [(k, int(raw_csv['frame'][r]), raw_csv['x'][r], raw_csv['y'][r]) for k, r in enumerate(picked)]
    print(f'{name}: {len(rows)} localisations of the selection, examples:')
    for k, r in enumerate(picked):
        print(f'   {k:2d}: row {r}, frame {int(raw_csv["frame"][r])}, rho of the selection {rho[r]:.1f} '
              f'(this frame {raw_csv["rho"][r]:.1f}), eta {eta[r]:.0f}, delta {delta[r]:.0f}, deviance {fit_quality[r]:.2f}')
out_examples = Path_lib(path).parent / 'mask_examples'
out_examples.mkdir(exist_ok=True)
with plt.ioff(): # figures saved and closed, no window
    DE.make_examples(Path_lib(path).parent, Path_lib(path).name, groups, out_examples, alt_rho)
print('figures in', out_examples)
#%% ############ interactive plots ######################
plt.tight_layout()
plt.show()
print(90-np.mean(eta[mask]))
print(90-np.median(eta[mask]))
#%% plot rho, interactive window of 10 degrees in rho
# the slider sets the centre of the window, mask_rho keeps the points of mask inside it for the next cells
%matplotlib qt
rho_window = 10 # width of the window in degrees
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
fig.subplots_adjust(bottom=0.2)
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.04])

def in_rho_window(rho_values, centre):
    # distance on the 180 degrees circle, so that a window around 0 also takes the points around 180
    return np.abs((rho_values - centre + 90) % 180 - 90) <= rho_window/2

ax0.set_facecolor('black')
# only the points in the window are shown, the limits are the ones of the whole mask to keep the view fixed
ax0.set_xlim(np.min(x[mask])/1000, np.max(x[mask])/1000)
ax0.set_ylim(np.min(y[mask])/1000, np.max(y[mask])/1000)
sc_window = ax0.scatter([], [], c=[], cmap='hsv', vmin=0, vmax=180, s=10*s, rasterized=True)
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')

# half circle polar histogram, as in the cell before
rho_rad = np.deg2rad(rho[mask])
counts, edges = np.histogram(np.concatenate([rho_rad, rho_rad + np.pi]), bins=np.linspace(0, 2*np.pi, 73))
centers = (edges[:-1] + edges[1:]) / 2
bars = ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=plt.cm.hsv((centers % np.pi) / np.pi))
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_zero_location('E')
ax1.set_theta_direction(1)
ax1.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax1.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
ax1.set_ylabel('Count', labelpad=30)

rho_slider = Slider(ax_slider, r'$\rho$ centre (°)', 0, 180, valinit=120, valstep=1)

def update_rho_window(centre):
    global mask_rho
    mask_rho = mask & in_rho_window(rho, centre)
    sc_window.set_offsets(np.column_stack((x[mask_rho]/1000, y[mask_rho]/1000)))
    sc_window.set_array(rho[mask_rho])
    # the bars outside the window are faded
    for bar, c in zip(bars, np.rad2deg(centers) % 180):
        bar.set_alpha(0.9 if in_rho_window(c, centre) else 0.15)
    ax0.set_title(f'$\\rho$ in [{centre-rho_window/2:.0f}°, {centre+rho_window/2:.0f}°]: {np.sum(mask_rho)} points')
    fig.canvas.draw_idle()

rho_slider.on_changed(update_rho_window)
update_rho_window(rho_slider.val)
plt.show()

#%% plot z, interactive window in z
# the slider sets the centre of the window, mask_z keeps the points of mask inside it for the next cells
%matplotlib qt
z_window = 100 # width of the window in nm
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 15})
fig, (ax0, ax1) = plt.subplots(1, 2)
fig.subplots_adjust(bottom=0.2)
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.04])
z_min, z_max = np.min(z[mask]), np.max(z[mask])

ax0.set_facecolor('black')
# only the points in the window are shown, the limits are the ones of the whole mask to keep the view fixed
ax0.set_xlim(np.min(x[mask])/1000, np.max(x[mask])/1000)
ax0.set_ylim(np.min(y[mask])/1000, np.max(y[mask])/1000)
sc_window = ax0.scatter([], [], c=[], cmap='plasma', vmin=z_min, vmax=z_max, s=10*s, rasterized=True)
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')

z_bin = 100 # nm
counts, edges = np.histogram(z[mask], bins=np.arange(np.floor(z_min/z_bin)*z_bin, np.ceil(z_max/z_bin)*z_bin + z_bin, z_bin))
centers = (edges[:-1] + edges[1:]) / 2
# same colormap as the plot z cell: plasma over the z range of the mask
bars = ax1.bar(centers, counts, width=z_bin, color=plt.cm.plasma(np.clip((centers - z_min) / (z_max - z_min), 0, 1)), edgecolor='black')
ax1.set_xlabel('z (nm)')
ax1.set_ylabel('Count')

z_slider = Slider(ax_slider, 'z centre (nm)', z_min, z_max, valinit=(z_min + z_max)/2)

def update_z_window(centre):
    global mask_z
    mask_z = mask & (np.abs(z - centre) <= z_window/2)
    sc_window.set_offsets(np.column_stack((x[mask_z]/1000, y[mask_z]/1000)))
    sc_window.set_array(z[mask_z])
    # the bars that do not overlap the window are faded
    for bar, c in zip(bars, centers):
        bar.set_alpha(0.9 if abs(c - centre) < (z_window + z_bin)/2 else 0.15)
    ax0.set_title(f'z in [{centre-z_window/2:.0f}, {centre+z_window/2:.0f}] nm: {np.sum(mask_z)} points')
    fig.canvas.draw_idle()

z_slider.on_changed(update_z_window)
update_z_window(z_slider.val)
plt.show()

#%% plot eta, interactive window of 10 degrees in eta
# the slider sets the centre of the window, mask_eta keeps the points of mask inside it for the next cells
%matplotlib qt
eta_window = 10 # width of the window in degrees
eta_cmap = mcolors.ListedColormap(sns.color_palette("hls", 256)) # same colormap as the plot eta cell
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
fig.subplots_adjust(bottom=0.2)
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.04])

def in_eta_window(eta_values, centre):
    # eta is defined modulo 180 degrees, a window around 0 also takes the points around 180
    return np.abs((eta_values - centre + 90) % 180 - 90) <= eta_window/2

ax0.set_facecolor('black')
# only the points in the window are shown, the limits are the ones of the whole mask to keep the view fixed
ax0.set_xlim(np.min(x[mask])/1000, np.max(x[mask])/1000)
ax0.set_ylim(np.min(y[mask])/1000, np.max(y[mask])/1000)
sc_window = ax0.scatter([], [], c=[], cmap=eta_cmap, vmin=30/180, vmax=150/180, s=10*s, rasterized=True) # as in the plot eta cell
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')

# half circle polar histogram of eta
eta_rad = np.deg2rad(eta[mask])
counts, edges = np.histogram(eta_rad, bins=np.linspace(0, np.pi, 37))
centers = (edges[:-1] + edges[1:]) / 2
bars = ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=eta_cmap(centers / np.pi), edgecolor='none') # as in the plot eta cell
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_zero_location('N') # 0 on top
ax1.set_theta_direction(-1) # clockwise, 90 on the right
ax1.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax1.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
ax1.set_ylabel('Count', labelpad=30)

eta_slider = Slider(ax_slider, r'$\eta$ centre (°)', 0, 180, valinit=90, valstep=1)

def update_eta_window(centre):
    global mask_eta
    mask_eta = mask & in_eta_window(eta, centre)
    sc_window.set_offsets(np.column_stack((x[mask_eta]/1000, y[mask_eta]/1000)))
    sc_window.set_array(eta[mask_eta] / 180)
    # the bars outside the window are faded
    for bar, c in zip(bars, np.rad2deg(centers)):
        bar.set_alpha(0.9 if in_eta_window(c, centre) else 0.15)
    ax0.set_title(f'$\\eta$ in [{centre-eta_window/2:.0f}°, {centre+eta_window/2:.0f}°]: {np.sum(mask_eta)} points')
    fig.canvas.draw_idle()

eta_slider.on_changed(update_eta_window)
update_eta_window(eta_slider.val)
plt.show()

#%% plot z of the points in an interactive window of 10 degrees in eta
# the slider sets the centre of the eta window, the image and the last histogram show z of the points inside it,
# mask_eta_z keeps the points of mask inside it for the next cells
%matplotlib qt
eta_window = 10 # width of the window in degrees
eta_cmap = mcolors.ListedColormap(sns.color_palette("hls", 256)) # same colormap as the plot eta cell
plt.rcParams['figure.figsize'] = [18, 6]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2, projection='polar')
ax2 = fig.add_subplot(1, 3, 3)
fig.subplots_adjust(bottom=0.2, wspace=0.5) # room for the colorbar of z
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.04])

def in_eta_window(eta_values, centre):
    # eta is defined modulo 180 degrees, a window around 0 also takes the points around 180
    return np.abs((eta_values - centre + 90) % 180 - 90) <= eta_window/2

ax0.set_facecolor('black')
# only the points in the window are shown, the limits are the ones of the whole mask to keep the view fixed
ax0.set_xlim(np.min(x[mask])/1000, np.max(x[mask])/1000)
ax0.set_ylim(np.min(y[mask])/1000, np.max(y[mask])/1000)
# z in um with plasma over the z range of the mask, as in the plot z cell
sc_window = ax0.scatter([], [], c=[], cmap='plasma', vmin=np.min(z[mask])/1000, vmax=np.max(z[mask])/1000, s=10*s, rasterized=True)
cbar = plt.colorbar(sc_window, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_label("$z$ ($\\mu$m)")
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')

# half circle polar histogram of eta
eta_rad = np.deg2rad(eta[mask])
counts, edges = np.histogram(eta_rad, bins=np.linspace(0, np.pi, 37))
centers = (edges[:-1] + edges[1:]) / 2
bars = ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=eta_cmap(centers / np.pi), edgecolor='none') # as in the plot eta cell
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_zero_location('N') # 0 on top
ax1.set_theta_direction(-1) # clockwise, 90 on the right
ax1.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax1.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
ax1.set_ylabel('Count', labelpad=30)

# histogram of z of the points in the window, 100 nm bins over the z range of the mask
z_bin = 100 # nm
z_min, z_max = np.min(z[mask]), np.max(z[mask])
z_edges = np.arange(np.floor(z_min/z_bin)*z_bin, np.ceil(z_max/z_bin)*z_bin + z_bin, z_bin)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
z_bars = ax2.bar(z_centers, np.zeros(len(z_centers)), width=z_bin, edgecolor='black',
                 color=plt.cm.plasma(np.clip((z_centers - z_min) / (z_max - z_min), 0, 1)))
ax2.set_xlabel('z (nm)')
ax2.set_ylabel('Count')

eta_z_slider = Slider(ax_slider, r'$\eta$ centre (°)', 0, 180, valinit=90, valstep=1)

def update_eta_z_window(centre):
    global mask_eta_z
    mask_eta_z = mask & in_eta_window(eta, centre)
    sc_window.set_offsets(np.column_stack((x[mask_eta_z]/1000, y[mask_eta_z]/1000)))
    sc_window.set_array(z[mask_eta_z] / 1000)
    z_counts, _ = np.histogram(z[mask_eta_z], bins=z_edges)
    for bar, count in zip(z_bars, z_counts):
        bar.set_height(count)
    ax2.set_ylim(0, max(1, z_counts.max())*1.1)
    ax2.set_title(f'median z: {np.median(z[mask_eta_z]):.0f} nm' if np.any(mask_eta_z) else 'no point')
    # the bars outside the window are faded
    for bar, c in zip(bars, np.rad2deg(centers)):
        bar.set_alpha(0.9 if in_eta_window(c, centre) else 0.15)
    ax0.set_title(f'$\\eta$ in [{centre-eta_window/2:.0f}°, {centre+eta_window/2:.0f}°]: {np.sum(mask_eta_z)} points')
    fig.canvas.draw_idle()

eta_z_slider.on_changed(update_eta_z_window)
update_eta_z_window(eta_z_slider.val)
plt.show()

#%% plot yz of the points in an interactive window of 10 degrees in eta
# same as the cell before with a yz view instead of xy: the slider sets the centre of the eta window,
# the yz scatter and the last histogram show z of the points inside it, mask_eta_yz keeps them for the next cells
%matplotlib qt
eta_window = 10 # width of the window in degrees
eta_cmap = mcolors.ListedColormap(sns.color_palette("hls", 256)) # same colormap as the plot eta cell
plt.rcParams['figure.figsize'] = [18, 6]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2, projection='polar')
ax2 = fig.add_subplot(1, 3, 3)
fig.subplots_adjust(bottom=0.2, wspace=0.5) # room for the colorbar of z
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.04])

def in_eta_window(eta_values, centre):
    # eta is defined modulo 180 degrees, a window around 0 also takes the points around 180
    return np.abs((eta_values - centre + 90) % 180 - 90) <= eta_window/2

ax0.set_facecolor('black')
# only the points in the window are shown, the limits are the ones of the whole mask to keep the view fixed
ax0.set_xlim(np.min(y[mask])/1000, np.max(y[mask])/1000)
ax0.set_ylim(np.min(z[mask])/1000, np.max(z[mask])/1000)
# z in um with plasma over the z range of the mask, as in the plot z cell
sc_window = ax0.scatter([], [], c=[], cmap='plasma', vmin=np.min(z[mask])/1000, vmax=np.max(z[mask])/1000, s=10*s, rasterized=True)
cbar = plt.colorbar(sc_window, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_label("$z$ ($\\mu$m)")
# no equal aspect: the z range is much smaller than the y one
ax0.set_xlabel('y ($\\mu$m)')
ax0.set_ylabel('z ($\\mu$m)')

# half circle polar histogram of eta
eta_rad = np.deg2rad(eta[mask])
counts, edges = np.histogram(eta_rad, bins=np.linspace(0, np.pi, 37))
centers = (edges[:-1] + edges[1:]) / 2
bars = ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=eta_cmap(centers / np.pi), edgecolor='none') # as in the plot eta cell
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_zero_location('N') # 0 on top
ax1.set_theta_direction(-1) # clockwise, 90 on the right
ax1.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax1.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
ax1.set_ylabel('Count', labelpad=30)

# histogram of z of the points in the window, 100 nm bins over the z range of the mask
z_bin = 100 # nm
z_min, z_max = np.min(z[mask]), np.max(z[mask])
z_edges = np.arange(np.floor(z_min/z_bin)*z_bin, np.ceil(z_max/z_bin)*z_bin + z_bin, z_bin)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
z_bars = ax2.bar(z_centers, np.zeros(len(z_centers)), width=z_bin, edgecolor='black',
                 color=plt.cm.plasma(np.clip((z_centers - z_min) / (z_max - z_min), 0, 1)))
ax2.set_xlabel('z (nm)')
ax2.set_ylabel('Count')

eta_yz_slider = Slider(ax_slider, r'$\eta$ centre (°)', 0, 180, valinit=90, valstep=1)

def update_eta_yz_window(centre):
    global mask_eta_yz
    mask_eta_yz = mask & in_eta_window(eta, centre)
    sc_window.set_offsets(np.column_stack((y[mask_eta_yz]/1000, z[mask_eta_yz]/1000)))
    sc_window.set_array(z[mask_eta_yz] / 1000)
    z_counts, _ = np.histogram(z[mask_eta_yz], bins=z_edges)
    for bar, count in zip(z_bars, z_counts):
        bar.set_height(count)
    ax2.set_ylim(0, max(1, z_counts.max())*1.1)
    ax2.set_title(f'median z: {np.median(z[mask_eta_yz]):.0f} nm' if np.any(mask_eta_yz) else 'no point')
    # the bars outside the window are faded
    for bar, c in zip(bars, np.rad2deg(centers)):
        bar.set_alpha(0.9 if in_eta_window(c, centre) else 0.15)
    ax0.set_title(f'$\\eta$ in [{centre-eta_window/2:.0f}°, {centre+eta_window/2:.0f}°]: {np.sum(mask_eta_yz)} points')
    fig.canvas.draw_idle()

eta_yz_slider.on_changed(update_eta_yz_window)
update_eta_yz_window(eta_yz_slider.val)
plt.show()


# %%
