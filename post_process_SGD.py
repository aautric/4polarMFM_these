# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 2026

@author: Amaury Autric
amaury.autric@polytechnique.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import hsv_to_rgb

#%%
folder = '\\\\NAS_LOCCO\\Amaury\\DATA\\polMFM_experimental_processed\\these_4polar_MFM\\test_jax'
storing_folder = "\\\\NAS_LOCCO\\Amaury\\DATA\\4_polar_MFM_these\\test_jax.csv"
frame, x, y, z, N_photons, background_array_found, rho, eta, delta, score, x_start, y_start, z_start, rho_start, delta_start = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for filename in sorted(os.listdir(folder), key=lambda x: int(x.replace('.npz',''))):
    data = np.load(rf"{folder}\{filename}")
    #frame = np.concatenate((frame, data['frame']))
    x = np.concatenate((x, data['x']))
    y = np.concatenate((y, data['y']))
    z = np.concatenate((z, data['z'])) 
    background_array_found = np.concatenate((background_array_found, data['background_array_found'])) 
    N_photons = np.concatenate((N_photons, data['N_photons']))
    rho = np.concatenate((rho, data['rho']))
    eta = np.concatenate((eta, data['eta']))
    delta = np.concatenate((delta, data['delta']))
    score = np.concatenate((score, data['score']))
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
#%%
threshold = (N_photons>0)
#%%
plt.scatter(x , y , c = rho / 180.0, cmap='hsv', s=1)
plt.axis('equal')
#%%
'''
data = np.column_stack((
    frame[threshold], x[threshold], y[threshold], z[threshold], rho[threshold],
    eta[threshold], delta[threshold], N_photons[threshold], score[threshold],
    x_start[threshold], y_start[threshold], z_start[threshold],
    rho_start[threshold], delta_start[threshold]
))'''
data = np.column_stack((
    x[threshold], y[threshold], z[threshold], rho[threshold],
    eta[threshold], delta[threshold], N_photons[threshold], score[threshold],
))

# Save to CSV with many digits and proper header
# Ensure newline="" to avoid issues when reading in Excel
np.savetxt(
    storing_folder,
    data,
    delimiter=";",
    header="x;y;z;rho;eta;delta;N_photon;score",#"frame;x;y;z;rho;eta;delta;N_photon;score;x_start;y_start;z_start;rho_start;delta_start",
    comments='',
    fmt='%.15f'
)
