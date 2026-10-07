# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 2026

@author: Amaury Autric
amaury.autric@polytechnique.edu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
import sys
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/'))

#%%
look_up_folder = '/mnt/d/Amaury/DATA'
folder = Path(filedialog.askdirectory(
    initialdir=look_up_folder,
    title="Select the NPZ folder of the run to post-process"))
# the fit keeps the date of the run it comes from: NPZ_2026-08-10_14h32 -> fit_2026-08-10_14h32.csv
date_of_the_run = folder.name.replace('NPZ', '')
storing_folder = folder.parent / ('fit'+date_of_the_run+'.csv')
print('Post-processing: '+str(folder))
print('Storing in: '+str(storing_folder))
#storing_folder = '/mnt/c/Users/Amaury/Desktop/DATA/2026_02_10_actin_Moein/sm/fit.csv'
#"\\\\NAS_LOCCO\\Amaury\\DATA\\4_polar_MFM_these\\test_jax.csv"
frame, x, y, z, N_photons, background_array_found, rho, eta, delta, score, x_start, y_start, z_start, rho_start, delta_start = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
# aberrations fitted in SGD2: one set per batch (3 planes x 15 Noll modes, in rad, per polarisation),
# copied on each localisation of the batch
batch, zern_found = [], []
# deviance per pixel of each PSF (goodness of fit), absent from the runs before it was added
deviance = []
for filename in sorted(folder.glob('*.npz'), key=lambda p: int(p.stem)):
    data = np.load(filename)
    n_loc = len(data['frame'])
    batch.append(np.full(n_loc, int(filename.stem)))
    if 'zernx_found' in data.files: # absent from the runs without aberration fit
        zern_found.append(np.tile(np.concatenate((data['zernx_found'].flatten(), data['zerny_found'].flatten())), (n_loc, 1)))
    else:
        zern_found.append(np.full((n_loc, 2*3*15), np.nan))
    frame = np.concatenate((frame, data['frame']))
    x = np.concatenate((x, data['x']))
    y = np.concatenate((y, data['y']))
    
    z = np.concatenate((z, data['z'])) 
    background_array_found = np.concatenate((background_array_found, data['background_array_found'])) 
    N_photons = np.concatenate((N_photons, data['N_photons']))
    rho = np.concatenate((rho, data['rho']))
    eta = np.concatenate((eta, data['eta']))
    delta = np.concatenate((delta, data['delta']))
    score = np.concatenate((score, data['score']))
    deviance.append(data['deviance'] if 'deviance' in data.files else np.full(n_loc, np.nan))
    #x_start = np.concatenate((x_start, data['x_start']))
    #y_start = np.concatenate((y_start, data['y_start']))
    #z_start = np.concatenate((z_start, data['z_start']))    
    #rho_start = np.concatenate((rho_start, data['rho_start']))
    #delta_start = np.concatenate((delta_start, data['delta_start']))
frame = np.array(frame)
x = np.array(x)
y = np.array(y)
z = np.array(z)
N_photons = np.array(N_photons)
rho = np.array(rho)
eta = np.array(eta)
delta = np.array(delta)
score = np.array(score)
rho_start = np.array(rho_start)
delta_start = np.array(delta_start)
z_start = np.array(z_start)
x_start = np.array(x_start)
y_start = np.array(y_start)
background_array_found = np.array(background_array_found)
batch = np.concatenate(batch)
zern_found = np.concatenate(zern_found)
with_zernike = not np.all(np.isnan(zern_found))
#%%
threshold = (N_photons>0)

plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams.update({'font.size': 15})
sc = plt.scatter(x[threshold]/1000, y[threshold]/1000,
                 c=rho[threshold] / 180.0,
                 cmap='hsv', s=.001)
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

#%% storing as a csv
'''
data = np.column_stack((
    frame[threshold], x[threshold], y[threshold], z[threshold], rho[threshold],
    eta[threshold], delta[threshold], N_photons[threshold], score[threshold],
    x_start[threshold], y_start[threshold], z_start[threshold],
    rho_start[threshold], delta_start[threshold]
))'''
data = np.column_stack((
    frame[threshold], x[threshold], y[threshold], z[threshold], rho[threshold],
    eta[threshold], delta[threshold], N_photons[threshold], score[threshold], batch[threshold],
))
columns = "frame;x;y;z;rho;eta;delta;N_photon;score;batch"
deviance = np.concatenate(deviance)
if not np.all(np.isnan(deviance)):
    data = np.column_stack((data, deviance[threshold]))
    columns += ';deviance'
if with_zernike:
    # column x_p1_Z6 = x polarisation, plane 1, Noll index 6, in rad
    data = np.column_stack((data, zern_found[threshold]))
    columns += ';' + ';'.join(pol+'_p'+str(p)+'_Z'+str(m+1) for pol in ['x', 'y'] for p in range(3) for m in range(15))

# the whole config of the run is copied on top of the csv, as commented lines
config_path = folder / 'config.txt'
if config_path.is_file():
    config_lines = config_path.read_text().splitlines()
else:
    config_lines = ['no config.txt found in '+str(folder)]
    print('Warning: '+str(config_path)+' does not exist, the csv will have no config header')
header = '\n'.join('# '+line.lstrip('# ') for line in config_lines)
header += '\n' + columns#"frame;x;y;z;rho;eta;delta;N_photon;score;x_start;y_start;z_start;rho_start;delta_start"

# Save to CSV with many digits and proper header
# Ensure newline="" to avoid issues when reading in Excel
np.savetxt(
    storing_folder,
    data,
    delimiter=";",
    header=header,
    comments='',
    fmt='%.15f'
)

# %%
