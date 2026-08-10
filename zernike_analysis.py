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
look_up_folder = '/mnt/d/Amaury/DATA'
files = filedialog.askopenfilenames(
    initialdir=look_up_folder,
    title="",
    filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")]
)
data_ = []
for file in files:
    data_.append(np.load(file, allow_pickle=True))
#%%
s= 5
for data in data_:
    plt.scatter(range(15), data['zernx_found'][0]*1000/(2*np.pi), c='b', marker='x', s=s)
    plt.scatter(range(15), data['zernx_found'][1]*1000/(2*np.pi), c='r', marker='x', s=s)
    plt.scatter(range(15), data['zernx_found'][2]*1000/(2*np.pi), c='g', marker='x', s=s)
    plt.scatter(range(15), data['zerny_found'][0]*1000/(2*np.pi), c='c', marker='x', s=s)
    plt.scatter(range(15), data['zerny_found'][1]*1000/(2*np.pi), c='y', marker='x', s=s)
    plt.scatter(range(15), data['zerny_found'][2]*1000/(2*np.pi), c='orange', marker='x', s=s)
    #plt.show()
    for a in [data['zernx_found'][0], data['zernx_found'][1], data['zernx_found'][2],
              data['zerny_found'][0],data['zerny_found'][1],data['zerny_found'][2]]:
        rms_lambda = np.sqrt(np.sum(a[:-1]**2)) / (2*np.pi)
        rms_mlambda = rms_lambda * 1000
        print(f"RMS = {rms_mlambda:.1f} mλ")

#%%
