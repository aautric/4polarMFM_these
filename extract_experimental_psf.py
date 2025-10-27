# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:23:31 2025

@author: LOCCO_Louise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import tifffile

l_pixel = 16000 # nm
l_pixel_field = (l_pixel/100)/((200/150))

def pos_from_csv(path_csv):
    data = pd.read_csv(path_csv)
    return data

def xy_from_data(data, frame):
    x = data[data['frame']==frame]['x [nm]'].to_numpy()
    y = data[data['frame']==frame]['y [nm]'].to_numpy()
    return x, y

def position_from_data(data, frame):
    x = data[data['frame']==frame]['x [nm]'].to_numpy()
    y = data[data['frame']==frame]['y [nm]'].to_numpy()
    z = data[data['frame']==frame]['z [nm]'].to_numpy()
    rho = data[data['frame']==frame]['rho'].to_numpy()
    delta = data[data['frame']==frame]['delta'].to_numpy()
    return x, y, z, rho, delta

def extract_raw(path_raw):
    if not os.path.exists(path_raw):
        raise FileNotFoundError(f"File not found: {path_raw}")
    
    try:
        with tifffile.TiffFile(path_raw) as tif:
            if len(tif.pages) < 6:
                raise ValueError(f"File {path_raw} has only {len(tif.pages)} frames")
            raw = np.zeros((6, 214, 129))
            for i in range(6):
                raw[i] = tif.pages[i].asarray()
        return raw
    except Exception as e:
        print(f"Error reading {path_raw}: {e}")
        return None
'''
def plot_raw(raw):
    plt.rcParams['figure.figsize'] = [12, 13]
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(2,3)
    for i in range(2):
        for j in range(3):
            ax[i,j].pcolormesh(raw[i*j], cmap='gray')
            ax[i,j].set_aspect('equal')
    plt.show()
    '''
def plot_raw_xy(raw, x, y, box=13, number=-1):
    plt.rcParams['figure.figsize'] = [20, 8]
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(2,3)
    vmax = np.max(raw)
    for i in range(2):
        for j in range(3):
            im = ax[i,j].pcolormesh(raw[2*j+i].T, cmap='gray', vmin=0., vmax=vmax)
            ax[i,j].set_aspect('equal')
            ax[i,j].set_title(str(i)+','+str(j)+'-->'+str(2*j+i))
            for k in range(len(x)):
                if k==number:
                    col = 'r'
                else:
                    col='y'
                rect = patches.Rectangle((x[k]/l_pixel_field-int(box/2), y[k]/l_pixel_field-int(box/2)), box, box, linewidth=2, edgecolor=col, facecolor='none')
                ax[i,j].add_patch(rect)
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8)
    plt.show()

def extract_raw_xy(raw, x, y, half_box=6):
    res = np.zeros((len(x), 3, 2, half_box*2+1, half_box*2+1))
    for i in range(2):
        for j in range(3):
            for k in range(len(x)):
                res[k, j, i] = raw[2*j+i, int(x[k]/l_pixel_field)-half_box:int(x[k]/l_pixel_field)+half_box+1, int(y[k]/l_pixel_field)-half_box:int(y[k]/l_pixel_field)+half_box+1]
    return res