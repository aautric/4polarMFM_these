# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 2026

@author: Amaury Autric
amaury.autric@polytechnique.edu
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
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/'))
# %%

#data = pd.read_csv("\\\\NAS_LOCCO\\Amaury\\DATA\\4_polar_MFM_these\\test_jax.csv", delimiter=';')
#data = pd.read_csv("/mnt/z/DATA/4_polar_MFM_these/test_jax.csv", delimiter=';')
look_up_folder = '/mnt/d/Amaury/DATA'
path = filedialog.askopenfilename(
    initialdir=look_up_folder,
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

# Open file dialog
data = pd.read_csv(path, delimiter=';')
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

loss_thresh = -2*10**5
mask1 = (score<loss_thresh) & (delta<150) & (delta>50) & (N_photons>300) & (N_photons<10000) & (z<1500) & (z>0) #& (eta>30) & (eta<150)
#mask1 = (delta<150) & (delta>60)

#%% mask selection

%matplotlib qt
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
        
%matplotlib inline
plt.rcParams['figure.figsize'] = [3,3]
xcorr = gaussian_filter1d(mean_x, 7)

xcorr = xcorr-xcorr[0]
plt.plot(xcorr)
plt.show()
ycorr = gaussian_filter1d(mean_y, 11)
ycorr = ycorr-ycorr[0]
plt.plot(ycorr)
plt.show()
correct_z = False
#%% Drift correction with fiducial

%matplotlib qt
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
#%%
hh = plt.hist(stdx, bins=100)
plt.xlabel('std x (nm)')
#%%
plt.show()
hh = plt.hist(stdz, bins=100)
plt.xlabel('std z (nm)')
#%%
plt.show()
hh = plt.hist(stdrho, bins=100)
plt.xlabel('std $\\rho$ (degree)')
plt.show()
#%%
hh = plt.hist(stdeta, bins=100)
plt.xlabel('std $\\eta$ (degree)')
plt.show()
#%%
hh = plt.hist(stddelta, bins=100)
plt.xlabel('std $\\delta$ (degree)')
plt.show()
#%%
mask1 = (score<loss_thresh) & (delta<150) &  (delta>50) & (N_photons>300) & (N_photons<4000) & (z<1000) & (z>0)
#%% select mask
%matplotlib qt
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
#%% plot rho
%matplotlib qt
plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2, projection='polar')
ax0.set_facecolor('black')
#ax1.set_facecolor('black')
sc = ax0.scatter(x[mask]/1000, y[mask]/1000,
                 c=rho[mask] / 180.0,
                 cmap='hsv', s=0.001)
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
                 cmap='plasma', s=0.001)#, vmin=0.4, vmax=1.)
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
    s=0.01,
    vmin=50,
    vmax=150)
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
#%% plot eta
%matplotlib qt
plt.rcParams['figure.figsize'] = [12, 5]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1,2)
ax[0].set_facecolor('black')
sc = ax[0].scatter(x[mask]/1000, y[mask]/1000,
                 c=eta[mask] / 180.0,
                 cmap='spring', s=0.01, vmin=30/180, vmax=150/180)
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
    cmap='spring',
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
norm = plt.Normalize(30, 150)
colors = plt.cm.spring(norm(np.rad2deg(centers_folded)))

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

#%%
%matplotlib qt

plt.rcParams['figure.figsize'] = [5, 3]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()
ax.set_facecolor('black')
sc = ax.scatter(y[mask]/1000, z[mask]/1000,
                 c=eta[mask] / 180.0,
                 cmap='gnuplot2', s=3, vmin=0/180, vmax=180/180)
plt.axis('equal')
# radial colorbar inset
cax = fig.add_axes([0.7, 0.6, 0.16, 0.16], projection='polar')

theta = np.linspace(np.deg2rad(0), np.deg2rad(180), 512)
r = np.array([0.8, 1.0])

Theta, R = np.meshgrid(theta, r)
C = np.tile(np.linspace(0, 180, theta.size), (2, 1))

cax.pcolormesh(
    Theta,
    R,
    C,
    cmap='gnuplot2',
    vmin=0,
    vmax=180,
    shading='auto',
    edgecolors='none'
)

# orientation
cax.set_theta_zero_location('N')   # 0° at top
cax.set_theta_direction(-1)        # clockwise

# ticks
ticks = np.arange(0, 181, 60)
cax.set_thetagrids(ticks, labels=[f'{t}°' for t in ticks])

cax.set_facecolor('black')
cax.tick_params(axis='x', colors='white')
# cosmetics
cax.set_rticks([])
cax.grid(False)
cax.spines['polar'].set_visible(False)

# label
cax.text(
    np.deg2rad(90),
    0.2,
    r'$\eta$',
    ha='center',
    va='center',
    fontsize=14, color='white'
)
#%%
%matplotlib qt
plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(x[mask1] , y[mask1], z[mask1], c=eta[mask1] / 180.0 , cmap='hsv', s=0.005)
ax.axis('equal')
cbar = plt.colorbar(sc)
ticks = np.linspace(0, 1, 7)  # 0 → 1
cbar.set_ticks(ticks)
cbar.set_ticklabels((ticks * 180).astype(int))
cbar.set_label("$\\rho$")

#%%

path_filtered = path[:-4]+'_filtered.csv'

data = np.column_stack((
    frame[mask], x[mask], y[mask], z[mask], rho[mask],
    eta[mask], delta[mask], N_photons[mask], score[mask],
))

# Save to CSV with many digits and proper header
# Ensure newline="" to avoid issues when reading in Excel
np.savetxt(
    path_filtered,
    data,
    delimiter=";",
    header="frame;x;y;z;rho;eta;delta;N_photon;score",#"frame;x;y;z;rho;eta;delta;N_photon;score;x_start;y_start;z_start;rho_start;delta_start",
    comments='',
    fmt='%.15f'
)