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
    print('Aberrations of '+str(len(zern_data))+' batches loaded')
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
    print('No aberrations in '+path+', run post_process_SGD.py on a run with the aberration fit')
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
quality_thresh = 1.6 if deviance is not None else 1500
mask1 = (fit_quality<quality_thresh) & (delta<150) & (delta>50) & (N_photons>50) & (N_photons<10000) & (z<1800) & (z>0) #& (eta>30) & (eta<150)
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
# link_z: about 3*sqrt(2)*(precision in z); 60 nm for actin, ~130 nm for SLB (z less precise at low photon numbers)
from scipy.spatial import cKDTree
link_xy = 30.        # nm
link_z = 100.         # nm
max_gap = 1          # number of missing frames allowed inside a track (0: consecutive frames only)
same_frame_xy = 30.  # nm, duplicates in the same frame

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
#%% std x, z, rho, eta, delta
# precision: std of the localisations linked over consecutive frames (same molecule). The mean and median of the
# per-molecule std underestimate the precision (molecules seen on 2-3 frames only); the pooled std
# sqrt(sum((n-1) s^2) / sum(n-1)) uses all the frames of all the molecules and is unbiased (sensitive to wrong links)
if len(stdx) == 0 or len(nlink) != len(stdx):
    raise ValueError(str(len(stdx))+' linked molecules ('+str(len(nlink))+' in nlink): load the csv again and run '
                     'the "correction doublons" cell once (a second linking on already linked data finds almost nothing)')
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
summary = []
for a, values, label, unit in [(ax[0,0], stdx, 'std x', 'nm'), (ax[0,1], stdz, 'std z', 'nm'),
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
ax[0,2].axis('off')
ax[0,2].text(0, 0.95, f'{len(stdx)} molecules on > 1 frame\n\n{"":16s}{"pooled":>7s}{"mean":>7s}{"median":>7s}\n'
             + '\n'.join(f'{name+" ("+unit+")":16s}{pooled_v:7.1f}{mean_v:7.1f}{median_v:7.1f}' for name, unit, pooled_v, mean_v, median_v in summary),
             va='top', family='monospace', fontsize=11)
plt.tight_layout()
plt.show()
#%% select new filter
mask1 = (fit_quality<quality_thresh) & (z<800) & (z>300) & (delta<150) &  (delta>50) & (N_photons>50) & (N_photons<10000)
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
s = 0.05
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
plt.rcParams['figure.figsize'] = [8, 5]
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
plt.rcParams['figure.figsize'] = [7, 5]
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

#%% Plot delta
plt.rcParams["figure.figsize"] = [8,4]
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
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1,2)
ax[0].set_facecolor('black')
sc = ax[0].scatter(x[mask]/1000, y[mask]/1000,
                 c=eta[mask] / 180.0,
                 cmap=cmap, s=s, vmin=30/180, vmax=150/180, rasterized=True)
# radial colorbar inset
cax2 = fig.add_axes([0.35, 0.63, 0.16, 0.16], projection='polar')

theta = np.linspace(np.deg2rad(30), np.deg2rad(150), 512)
r = np.array([0.8, 1.0])

Theta, R = np.meshgrid(theta, r)
C = np.tile(np.linspace(30, 150, theta.size), (2, 1))
cax2.set_facecolor('black')
cax2.pcolormesh(
    Theta,
    R,
    C,
    cmap=cmap,
    vmin=30,
    vmax=150,
    shading='auto',
    edgecolors='none'
)

# orientation
cax2.set_theta_zero_location('N')   # 0° at top
cax2.set_theta_direction(-1)        # clockwise
cax2.set_facecolor('white')
cax2.spines['polar'].set_color('black')
cax2.tick_params(axis='x', colors='black')
cax2.tick_params(axis='y', colors='black')
cax2.yaxis.label.set_color('black')
# ticks
ticks = np.arange(30, 151, 30)
cax2.set_thetagrids(ticks, labels=[f'{t}°' for t in ticks])

# cosmetics
cax2.set_rticks([])
cax2.grid(False)
cax2.spines['polar'].set_visible(False)

# label
cax2.text(
    np.deg2rad(90),
    0.4,
    r'$\eta$ (°)',
    ha='center',
    va='center',
    fontsize=14, color='black'
)

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
hls_colors_bar = [cmap(c / np.pi) for c in centers_folded]
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
    labels=[f"{d}°" for d in np.arange(0, 360, 60)]
)

ax1.set_ylabel("Count", labelpad=25)
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
#%%
############ interactive plots ######################

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

#%% gif of the rho window: all points, then the rho_theory dots of the bead fit, then the sliding rho window
# needs the bead fit cell (params, th, line, rgb_colorsL) and the plot rho cell above (rho_window, in_rho_window)
from matplotlib.animation import FuncAnimation, PillowWriter
gif_path = os.path.join(os.getcwd(), 'rho_window.gif')
gif_fps = 10
faded_alpha = 0.35  # opacity of the histogram bars outside the window. Not lower: too pale colors are merged with white in the 256-color palette of the gif
n_fade, n_hold_all = 5, 15     # phase 1: all the points fade in, then stay
n_ring, n_hold_ring = 20, 10   # phase 2: the rho_theory dots appear one after the other around the bead, then stay
n_sweep, n_hold_end = 90, 10   # phase 3: the rho window slides from 0 to 180 degrees, then stays at the end

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
fig.subplots_adjust(bottom=0.15)

x_ring = (params[1]+(200+params[0])*np.cos(th))/1000
y_ring = (params[2]+(200+params[0])*np.sin(th))/1000
ax0.set_facecolor('black')
# limits fixed for the whole gif, large enough for the points and the rho_theory dots
ax0.set_xlim(min(np.min(x[mask])/1000, np.min(x_ring)) - 0.05, max(np.max(x[mask])/1000, np.max(x_ring)) + 0.05)
ax0.set_ylim(min(np.min(y[mask])/1000, np.min(y_ring)) - 0.05, max(np.max(y[mask])/1000, np.max(y_ring)) + 0.05)
ax0.set_aspect('equal')
ax0.set_xlabel('x ($\\mu$m)')
ax0.set_ylabel('y ($\\mu$m)')
sc_all = ax0.scatter(x[mask]/1000, y[mask]/1000, c=rho[mask], cmap='hsv', vmin=0, vmax=180, s=10*s, alpha=0., rasterized=True)
sc_window = ax0.scatter([], [], c=[], cmap='hsv', vmin=0, vmax=180, s=10*s, rasterized=True)
sc_ring = ax0.scatter([], [], s=20, zorder=3)
ring_rgba = np.column_stack((rgb_colorsL, np.ones(len(th))))

# half circle polar histogram, as in the plot rho cell
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

n1_, n2_ = n_fade + n_hold_all, n_fade + n_hold_all + n_ring + n_hold_ring
n_frames = n2_ + n_sweep + n_hold_end

def update_gif(i):
    if i < n1_:  # phase 1: all the points
        sc_all.set_alpha(min(1., (i+1)/n_fade))
        ax0.set_title(f'all localizations: {np.sum(mask)} points')
    elif i < n2_:  # phase 2: rho_theory dots
        k = min(len(th), int(len(th)*(i-n1_+1)/n_ring))
        sc_ring.set_offsets(np.column_stack((x_ring[:k], y_ring[:k])))
        sc_ring.set_color(ring_rgba[:k])
        ax0.set_title('$\\rho_{\\mathrm{theory}}$ (bead fit)')
    else:  # phase 3: sliding rho window, as in the plot rho cell
        sc_all.set_alpha(0.)
        centre = 180*min(1., (i-n2_)/(n_sweep-1))
        mask_rho = mask & in_rho_window(rho, centre)
        sc_window.set_offsets(np.column_stack((x[mask_rho]/1000, y[mask_rho]/1000)))
        sc_window.set_array(rho[mask_rho])
        # the rho_theory dots and the bars outside the window are faded
        ring_rgba[:, 3] = np.where(in_rho_window(line, centre), 1., 0.2)
        sc_ring.set_color(ring_rgba)
        for bar, c in zip(bars, np.rad2deg(centers) % 180):
            bar.set_alpha(0.9 if in_rho_window(c, centre) else faded_alpha)
        ax0.set_title(f'$\\rho$ in [{centre-rho_window/2:.0f}°, {centre+rho_window/2:.0f}°]: {np.sum(mask_rho)} points')

anim = FuncAnimation(fig, update_gif, frames=n_frames)
anim.save(gif_path, writer=PillowWriter(fps=gif_fps), dpi=100)  # loops forever by default
plt.close(fig)
print('gif saved in', gif_path)

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


#%% gif of the eta window in yz: all points, then the sliding eta window
# needs the cell above (eta_window, eta_cmap, in_eta_window)
from matplotlib.animation import FuncAnimation, PillowWriter
gif_path = os.path.join(os.getcwd(), 'eta_yz_window.gif')
gif_fps = 10
faded_alpha = 0.35  # opacity of the histogram bars outside the window. Not lower: too pale colors are merged with white in the 256-color palette of the gif
n_fade, n_hold_all = 5, 15     # phase 1: all the points fade in, then stay
n_sweep, n_hold_end = 90, 10   # phase 2: the eta window slides from 0 to 180 degrees, then stays at the end

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
fig.subplots_adjust(bottom=0.15, wspace=0.4) # room for the colorbar of z
z_min, z_max = np.min(z[mask]), np.max(z[mask])

ax0.set_facecolor('black')
# limits fixed for the whole gif, with a small margin
ax0.set_xlim(np.min(y[mask])/1000 - 0.05, np.max(y[mask])/1000 + 0.05)
ax0.set_ylim(z_min/1000 - 0.05, z_max/1000 + 0.05)
ax0.set_aspect('equal')
sc_all = ax0.scatter(y[mask]/1000, z[mask]/1000, c=z[mask]/1000, cmap='plasma', vmin=z_min/1000, vmax=z_max/1000, s=10*s, alpha=0., rasterized=True)
sc_window = ax0.scatter([], [], c=[], cmap='plasma', vmin=z_min/1000, vmax=z_max/1000, s=10*s, rasterized=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
cax = make_axes_locatable(ax0).append_axes('right', size='5%', pad=0.1)  # colorbar as high as the equal aspect plot
cbar = plt.colorbar(sc_window, cax=cax)
cbar.set_label("$z$ ($\\mu$m)")
ax0.set_xlabel('y ($\\mu$m)')
ax0.set_ylabel('z ($\\mu$m)')

# half circle polar histogram of eta, as in the cell above
eta_rad = np.deg2rad(eta[mask])
counts, edges = np.histogram(eta_rad, bins=np.linspace(0, np.pi, 37))
centers = (edges[:-1] + edges[1:]) / 2
bars = ax1.bar(centers, counts, width=edges[1]-edges[0], bottom=0, color=eta_cmap(centers / np.pi), edgecolor='none')
ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_zero_location('N') # 0 on top
ax1.set_theta_direction(-1) # clockwise, 90 on the right
ax1.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
ax1.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
ax1.set_ylabel('Count', labelpad=30)

n1_ = n_fade + n_hold_all
n_frames = n1_ + n_sweep + n_hold_end

def update_gif(i):
    if i < n1_:  # phase 1: all the points
        sc_all.set_alpha(min(1., (i+1)/n_fade))
        ax0.set_title(f'all localizations: {np.sum(mask)} points')
    else:  # phase 2: sliding eta window, as in the cell above
        sc_all.set_alpha(0.)
        centre = 180*min(1., (i-n1_)/(n_sweep-1))
        mask_eta_yz = mask & in_eta_window(eta, centre)
        sc_window.set_offsets(np.column_stack((y[mask_eta_yz]/1000, z[mask_eta_yz]/1000)))
        sc_window.set_array(z[mask_eta_yz] / 1000)
        # the bars outside the window are faded
        for bar, c in zip(bars, np.rad2deg(centers)):
            bar.set_alpha(0.9 if in_eta_window(c, centre) else faded_alpha)
        ax0.set_title(f'$\\eta$ in [{centre-eta_window/2:.0f}°, {centre+eta_window/2:.0f}°]: {np.sum(mask_eta_yz)} points')

anim = FuncAnimation(fig, update_gif, frames=n_frames)
anim.save(gif_path, writer=PillowWriter(fps=gif_fps), dpi=100)  # loops forever by default
plt.close(fig)
print('gif saved in', gif_path)

# %%
