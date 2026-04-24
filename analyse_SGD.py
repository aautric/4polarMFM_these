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
# %%

#data = pd.read_csv("\\\\NAS_LOCCO\\Amaury\\DATA\\4_polar_MFM_these\\sperm_souris_7.csv", delimiter=';')

root = Tk()
root.withdraw()

# Open file dialog
file_path = filedialog.askopenfilename()
data = pd.read_csv(file_path, delimiter=';')

if "background_array_found" in data.index: # case to process SGD data
    frame = data['frame'].to_numpy().astype(int)
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    z = data['z'].to_numpy()
    rho = data['rho'].to_numpy()
    eta = data['eta'].to_numpy()
    delta = data['delta'].to_numpy()
    N_photons = data['N_photon'].to_numpy()
    background_array_found = data['background_array_found'].to_numpy()
    score = data['score'].to_numpy()
    x_start = data['x_start'].to_numpy()
    y_start = data['y_start'].to_numpy()
    z_start = data['z_start'].to_numpy()
    rho_start = data['rho_start'].to_numpy()
    delta_start = data['delta_start'].to_numpy()
else: # case ratiometric only
    data = pd.read_csv(file_path, delimiter=',')
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
plt.rcParams['figure.figsize'] = [8,3]
fig, ax = plt.subplots(1,3)
hist = ax[0].hist(score, bins=50)
ax[0].set_xlabel('Finale loss per PSF')
ax[0].set_ylabel('Occurences')

hist = ax[1].hist(N_photons, bins=50)
ax[1].set_xlim((0, 10000))
ax[1].set_xlabel('Photon number per PSF')
ax[1].set_ylabel('Occurences')

hist = ax[2].hist(z, bins=80)
ax[2].set_xlabel('z')
ax[2].set_ylabel('Occurences')

# %% TO BE MODIFIED

loss_thresh = 200
mask1 = (score<loss_thresh) & (delta<150) & (delta>10) & (N_photons>500) & (N_photons<5000) & (z<2500) & (z>-2500) #& (eta>30) & (eta<150)

#%%DRIFT CORRECTION

%matplotlib qt
plt.rcParams['figure.figsize'] = [15,15]
fig = plt.figure()
ax = fig.add_subplot()
vals = frame
sc = ax.scatter(x , y , c=vals , cmap='coolwarm', s=1)
ax.set_title("select a point cloud")
ax.axis('equal')
cb = plt.colorbar(sc)
cb.ax.invert_yaxis()
points = np.column_stack((x, y))
mask = np.zeros(len(x), dtype=bool)

def onselect(verts):
    global mask
    path = Path(verts)
    mask = path.contains_points(points) 
    print(mask)

lasso = LassoSelector(ax, onselect)
plt.show()

#%% function
def mean_x_per_20_frames(x, frame, window=20):

    bins = (frame // window).astype(int)
    n_bins = bins.max() + 1

    sum_x = np.bincount(bins, weights=x, minlength=n_bins)
    count = np.bincount(bins, minlength=n_bins)

    mean_x = np.zeros(n_bins)

    nonzero = count > 0
    mean_x[nonzero] = sum_x[nonzero] / count[nonzero]

    mean_x[~nonzero] = np.nan  # only real empty bins

    return mean_x

#%%
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

frame_bins = np.linspace(0, np.max(frame), len(xcorr)).astype(int)

for ind, frame_ in tqdm(enumerate(frame)):
    idx = bisect.bisect_left(frame_bins, frame_)
    #print(frame_, idx, frame_bins[idx-1])
    x[ind]-=xcorr[idx-1]
    y[ind]-=ycorr[idx-1]
    #print(ind, frame_, x[ind], y[ind], xcorr[idx-1]*0.45, ycorr[idx-1]*0.6)

%matplotlib qt
plt.rcParams['figure.figsize'] = [15,15]
fig = plt.figure()
ax = fig.add_subplot()
vals = frame
sc = ax.scatter(x , y , c=vals , cmap='coolwarm', s=1)
ax.set_title("select a point cloud")
ax.axis('equal')
cb = plt.colorbar(sc)
cb.ax.invert_yaxis()
points = np.column_stack((x, y))
mask = np.zeros(len(x), dtype=bool)

def onselect(verts):
    global mask
    path = Path(verts)
    mask = path.contains_points(points) 
    print(mask)

lasso = LassoSelector(ax, onselect)
plt.show()

#%%
%matplotlib qt
plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams.update({'font.size': 15})
sc = plt.scatter(x[mask1]/1000, y[mask1]/1000,
                 c=rho[mask1] / 180.0,
                 cmap='hsv', s=0.5)
cbar = plt.colorbar(sc)
ticks = np.linspace(0, 1, 7)  # 0 → 1
cbar.set_ticks(ticks)
cbar.set_ticklabels((ticks * 180).astype(int))
cbar.set_label("$\\rho$")
plt.axis('equal')
#plt.xlim((10000, 22500))
#plt.ylim((2200, 14000))
plt.xlabel('x ($\\mu$m)')
plt.ylabel('y ($\\mu$m)')
plt.show()

#%%

%matplotlib qt
plt.rcParams['figure.figsize'] = [15,15]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc = ax.scatter(x[mask1] , y[mask1], z[mask1], c=rho[mask1] / 180.0 , cmap='hsv', s=0.5)
ax.axis('equal')
cbar = plt.colorbar(sc)
ticks = np.linspace(0, 1, 7)  # 0 → 1
cbar.set_ticks(ticks)
cbar.set_ticklabels((ticks * 180).astype(int))
cbar.set_label("$\\rho$")