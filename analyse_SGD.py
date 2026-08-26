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
    
# %% basic plots to see the thresholding

%matplotlib inline
plt.rcParams['figure.figsize'] = [12,3]
fig, ax = plt.subplots(1,3)
hist = ax[0].hist(score, bins=50)
ax[0].set_xlabel('Finale loss per PSF')
ax[0].set_ylabel('Occurences')

hist = ax[1].hist(N_photons, bins=50)
ax[1].set_xlim((0, 20000))
ax[1].set_xlabel('Photon number per PSF')
ax[1].set_ylabel('Occurences')

hist = ax[2].hist(z, bins=80)
ax[2].set_xlabel('z')
ax[2].set_ylabel('Occurences')

# %% TO BE MODIFIED
loss_thresh = -1*10**4
mask1 = (score<loss_thresh) & (delta<150) & (delta>50) & (N_photons>300) & (N_photons<10000) & (z<1500) & (z>0) #& (eta>30) & (eta<150)
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
#%%
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
def recursive_call(i, x, y, z, rho, eta, delta, frame, previous_index, not_counted):
    mask = ((frame==frame[i]+1)|(frame==frame[i]+1)|(frame==frame[i]+1)) & ((x-x[i])**2+(y-y[i])**2<30**2) & ((z-z[i])**2<60**2) #& ((rho-rho[i])<30)
    if (mask==True).any():
        not_counted[i] = False
        return recursive_call(np.where(mask)[0][0], x, y, z, rho, eta, delta, frame, 
                             previous_index=np.concatenate((previous_index, np.where(mask)[0])),
                             not_counted=not_counted)
    else:
        return previous_index.astype(int)

def link_localizations(x, y, z, rho, eta, delta, N_photons, score, frame):
    not_counted = np.ones(len(x), dtype=bool)
    frame = frame.astype(float) 
    stdx, stdy, stdz, stdrho, stdeta, stddelta = [], [], [], [], [], []
    meanx, meany, meanz, meanrho, meaneta, meandelta, meanN, meanscore = [], [], [], [], [], [], [], []
    
    for i in tqdm(range(len(x))):
        if not_counted[i]:
            indices = recursive_call(i, x, y, z, rho, eta, delta, frame,
                                     previous_index=np.array([i]),
                                     not_counted=not_counted)
            if len(indices) > 1:
                stdx.append(np.std(x[indices]))
                stdy.append(np.std(y[indices]))
                stdz.append(np.std(z[indices]))
                stdrho.append(np.std(rho[indices]))
                stdeta.append(np.std(eta[indices]))
                stddelta.append(np.std(delta[indices]))
                meanx.append(np.mean(x[indices]))
                meany.append(np.mean(y[indices]))
                meanz.append(np.mean(z[indices]))
                meanrho.append(np.mean(rho[indices]))
                meaneta.append(np.mean(eta[indices]))
                meandelta.append(np.mean(delta[indices]))
                meanN.append(np.mean(N_photons[indices]))
                meanscore.append(np.mean(score[indices]))
                x[indices[-1]] = np.mean(x[indices])
                x[indices[:-1]] = np.nan
                y[indices[-1]] = np.mean(y[indices])
                y[indices[:-1]] = np.nan
                z[indices[-1]] = np.mean(z[indices])
                z[indices[:-1]] = np.nan
                rho[indices[-1]] = np.mean(rho[indices])
                rho[indices[:-1]] = np.nan
                eta[indices[-1]] = np.mean(eta[indices])
                eta[indices[:-1]] = np.nan
                delta[indices[-1]] = np.mean(delta[indices])
                delta[indices[:-1]] = np.nan
                N_photons[indices[-1]] = np.mean(N_photons[indices])
                N_photons[indices[:-1]] = np.nan
                score[indices[-1]] = np.mean(score[indices])
                score[indices[:-1]] = np.nan
                frame[indices[-1]] = np.mean(frame[indices])
                frame[indices[:-1]] = np.nan

    print('removed ', len(np.where(np.isnan(x))[0]), ' over ', len(x))
    return (x, y, z, rho, eta, delta, N_photons, score, frame,
            np.array(stdx), np.array(stdy), np.array(stdz), np.array(stdrho), np.array(stdeta), np.array(stddelta),
            np.array(meanx), np.array(meany), np.array(meanz), np.array(meanrho), np.array(meaneta), np.array(meandelta), np.array(meanN), np.array(meanscore))#%% select zone to analyse

x, y, z, rho, eta, delta, N_photons, score, frame, stdx, stdy, stdz, stdrho, stdeta, stddelta, meanx, meany, meanz, meanrho, meaneta, meandelta, meanN, meanscore = link_localizations(x, y, z, rho, eta, delta, N_photons, score, frame)
#%% std x
fig, ax = plt.subplots()
hh = ax.hist(stdx, bins=100)
ax.set_xlabel('std x (nm)')
#%% std z
plt.show()
hh = plt.hist(stdz, bins=100)
plt.xlabel('std z (nm)')
#%%std rho
plt.show()
hh = plt.hist(stdrho, bins=100)
plt.xlabel('std $\\rho$ (degree)')
plt.show()
#%% std eta
hh = plt.hist(stdeta, bins=100)
plt.xlabel('std $\\eta$ (degree)')
plt.show()
#%% std delta
hh = plt.hist(stddelta, bins=100)
plt.xlabel('std $\\delta$ (degree)')
plt.show()
#%% select new filter
mask1 = (score<loss_thresh) & (z<1400) & (z>0) & (delta<150) &  (delta>50) & (N_photons>300) & (N_photons<10000)
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
s = 1.
#%% plot rho
plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
ax0.set_facecolor('black')
#ax1.set_facecolor('black')
sc = ax0.scatter(x[mask]/1000, y[mask]/1000,
                 c=rho[mask] / 180.0,
                 cmap='hsv', s=s, rasterized=True)
cax = fig.add_axes([0.5, 0.62, 0.15, 0.15], projection='polar')
theta = np.linspace(0, np.pi, 512)
r = np.array([0.8, 1.0])
Theta, R = np.meshgrid(theta, r)
C = np.tile(np.linspace(0, 180, theta.size), (2, 1))
# Label in the center of the semicircle
cax.text(np.pi/2, 0.2, r'$\rho$ (°)', ha='center', va='center', fontsize=14)
cax.pcolormesh(Theta, R, C, cmap='hsv', shading='auto', edgecolors='none', linewidth=0)
cax.set_theta_zero_location('E')   # start at right
cax.set_theta_direction(1)         # counterclockwise
cax.set_thetamin(0)
cax.set_thetamax(180)
cax.set_rticks([])
cax.grid(False)
ticks_deg = np.arange(0, 181, 45)
cax.set_thetagrids(ticks_deg, labels=[f"{d}°" for d in ticks_deg])
cax.spines['polar'].set_visible(False)
ticks = np.linspace(0, 1, 7)
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
plt.rcParams["figure.figsize"] = [12, 5]
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
#%%colormap
hls_colors = sns.color_palette("hls", 256)
cmap = mcolors.ListedColormap(hls_colors)
#%% plot eta
plt.rcParams['figure.figsize'] = [15, 5]
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
colors = cmap(norm(np.rad2deg(centers_folded)))
norm = plt.Normalize(30, 150)

# rebuild polar axis
ax[1].remove()
ax1 = fig.add_subplot(1, 2, 2, projection='polar')

# White angular tick labels
for label in ax1.get_xticklabels():
    label.set_color('black')

# White radial tick labels
for label in ax1.get_yticklabels():
    label.set_color('black')
ax1.bar(
    centers,
    counts,
    width=width,
    bottom=0,
    color=colors,
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
    params = torch.tensor([1000.,8800.,-15400.], requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=100)
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
    
params = find_params(x[new_zoom], y[new_zoom])
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
delta_rho = (rho[new_zoom] - rho_th + 90) % 180 - 90 -4
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
ax[0].set_ylim(-100, 100)
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

# usage
w, eigenvectors, eigenvalues, mean_xy = pca_xy(x[mask], y[mask], z[mask])

# w[:,0] — coordinate along first principal axis (most variance)
# w[:,1] — coordinate along second principal axis
print(f"Principal axis direction: {eigenvectors[:,0]}")
print(f"Explained variance: {eigenvalues/eigenvalues.sum()*100}")

# %% orientation of filaments
from scipy import stats

plt.rcParams['figure.figsize'] = [5, 7]
plt.rcParams.update({'font.size': 13})

fig = plt.figure()
ax0 = fig.add_subplot(2, 1, 1)
ax1 = fig.add_subplot(2, 1, 2, projection='polar')

# --- linear regression ---
slope, intercept, r_value, p_value, std_err = stats.linregress(w[:,0]/1000, z[mask]/1000)
slope_deg = np.rad2deg(np.arctan(slope))
print(f"Slope: {slope_deg:.2f}°")

x_fit = np.linspace(w[:,0].min()/1000, w[:,0].max()/1000, 500)
y_fit = slope * x_fit + intercept

# --- scatter plot ---
ax0.scatter(w[:,0]/1000, z[mask]/1000, s=0.1, color='steelblue', alpha=0.5, rasterized=True)
ax0.plot(x_fit, y_fit, color='red', linewidth=2, label=f'slope = {slope_deg:.2f}°')
#ax0.set_ylim((0.95,1.45))
#ax0.set_aspect('equal')
ax0.set_xlabel('PCA axis ($\\mu$m)')
ax0.set_ylabel('z ($\\mu$m)')
ax0.legend()
ax0.grid(True, alpha=0.4)

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

# --- line corresponding to slope ---
slope = np.median(eta[mask])
slope_rad = np.deg2rad(slope)  # convert to polar angle (0=North)
ax1.axvline(slope_rad, color='red', linewidth=2, label=f'{90-slope:.2f}°')
ax1.axvline(slope_rad + np.pi, color='red', linewidth=2)  # opposite direction

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

ax1.set_thetagrids(
    np.arange(0, 360, 60),
    labels=[f"{d}°" for d in np.arange(0, 360, 60)]
)
ax1.set_ylabel("Count", labelpad=25)
ax1.legend(loc='upper right')

plt.tight_layout()
plt.show()
print(90-np.mean(eta[mask]))
print(90-np.median(eta[mask]))
# %%
