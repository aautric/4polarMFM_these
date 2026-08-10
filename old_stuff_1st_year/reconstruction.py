# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:20:29 2026

@author: LOCCO_Louise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import tifffile 
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, label, find_objects
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def extract_raw_ome(path_raw, first_frame=None, last_frame=None):
    if not os.path.exists(path_raw):
        raise FileNotFoundError(f"File not found: {path_raw}")
    try:
        with tifffile.TiffFile(path_raw) as tif:
            if first_frame is None:
                first_frame=0
                last_frame = len(tif.pages)
            raw = np.zeros((last_frame-first_frame, 512, 512))
            for i in range(last_frame-first_frame):
                raw[i] = tif.pages[first_frame+i].asarray()
        return raw
    except Exception as e:
        print(f"Error reading {path_raw}: {e}")
        return None
    
def detect_plateau_0(img):
    sigma = 3
    line = np.mean(img, axis=0)
    der = line[1:]-line[:-1]
    smooth = gaussian_filter1d(der, sigma=sigma)
    max1 = np.argmax(smooth[0:50])
    min2 = 150+np.argmin(smooth[150:200])
    max2 = 150+np.argmax(smooth[150:200])
    min3 = 300+np.argmin(smooth[300:400])
    max3 = 300+np.argmax(smooth[300:400])
    min4 = 420+np.argmin(smooth[420:-1])
    return max1-1, min2+1, max2-1, min3+1, max3-1, min4+1
    
def detect_plateau_1(img):
    sigma = 3
    line = np.mean(img, axis=1)
    der = line[1:]-line[:-1]
    smooth = gaussian_filter1d(der, sigma=sigma)
    
    max1 = np.argmax(smooth[0:80])
    min2 = 200+np.argmin(smooth[200:300])
    max2 = 200+np.argmax(smooth[200:300])
    min3 = 420+np.argmin(smooth[420:-1])
    return max1-1, min2+1, max2-1, min3+1
    
def gauss(coords, A, mu1, mu2, s, c):
    x, y = coords
    return (16**2)*A*(1/(2*np.pi*(s**2)))*np.exp(-0.5*(((x-mu1)**2+(y-mu2)**2)/((s**2)))) + c

plt.ion()
def select_points_subimage(img, x1, x2, y0, y1):

    # Extract the subimage
    subimg = img[y0:y1, x1:x2]

    # Display the subimage
    fig, ax = plt.subplots()
    ax.imshow(subimg, cmap='gray')
    ax.set_title("Click to select points, press Enter when done")

    # Let the user select points interactively
    points = plt.ginput(n=-1, timeout=0)  # n=-1 allows unlimited points
    plt.close(fig)

    # Convert to coordinates in the original image
    points = np.array([(x + x1, y + y0) for x, y in points])

    return points

def fit_one_image(img, points):
    X, Y = np.meshgrid(np.arange(512), np.arange(512))  # 512x512 grids
    xdata = np.vstack((X.ravel(), Y.ravel()))  # shape (2, 262144)
    zdata = img.ravel()
    p_list = []
    for i in range(len(points)):
        p, pcov = curve_fit(gauss, xdata, zdata, p0=(70, points[i,0], points[i,1], 2, 150))
        p_list.append(p)
    return np.array(p_list)
    