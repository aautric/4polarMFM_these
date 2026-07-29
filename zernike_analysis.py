#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:44:12 2026

@author: Amaury Autric
amaury.autric@polytechnique.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog
from matplotlib.colors import hsv_to_rgb
import sys
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/'))

#%%
look_up_folder = '/mnt/c/Users/Amaury/Desktop/DATA/'#'/mnt/z/DATA/4_polar_MFM_these/'
file = filedialog.askopenfilename(
     initialdir=look_up_folder,
     title="",
     filetypes=[("fits", "*.npz"), ("Tous", "*.*")]
     )
data = np.load(file, allow_pickle=True)
#%%
plt.scatter(range(15), data['zernx_found'][0]*1000/(2*np.pi), c='b')
plt.scatter(range(15), data['zernx_found'][1]*1000/(2*np.pi), c='r')
plt.scatter(range(15), data['zernx_found'][2]*1000/(2*np.pi), c='g')
plt.scatter(range(15), data['zerny_found'][0]*1000/(2*np.pi), c='c')
plt.scatter(range(15), data['zerny_found'][1]*1000/(2*np.pi), c='y')
plt.scatter(range(15), data['zerny_found'][2]*1000/(2*np.pi), c='orange')

for a in [data['zernx_found'][0], data['zernx_found'][1], data['zernx_found'][2],
          data['zerny_found'][0],data['zerny_found'][1],data['zerny_found'][2]]:
    rms_lambda = np.sqrt(np.sum(a[:-1]**2)) / (2*np.pi)
    rms_mlambda = rms_lambda * 1000
    
    print(f"RMS = {rms_mlambda:.1f} mλ")