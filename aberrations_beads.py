#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:41:40 2026

@author: amaury
"""

import matplotlib
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
import tifffile 
import tkinter as tk
import cv2
from tkinter import filedialog, messagebox
from scipy import ndimage
from pathlib import Path
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector

def pick_threshold(img, cmap="gray", vmax=None, initial_vmin=None):
    """
    Iteratively choose a vmin threshold for displaying a 2D image.
 
    Shows the image with the current vmin, then prompts for a new value.
    Press Enter (empty input) to accept the current threshold and stop.
    Type 'q' to abort and return None.
 
    Parameters
    ----------
    img : 2D array-like
    cmap : str, optional
    vmax : float, optional
        Fixed vmax. Defaults to img.max().
    initial_vmin : float, optional
        Starting threshold. Defaults to img.min().
 
    Returns
    -------
    float or None
        The accepted vmin, or None if aborted.
    """
    img = np.asarray(img)
    img_min, img_max = float(np.nanmin(img)), float(np.nanmax(img))
    if vmax is None:
        vmax = img_max
 
    vmin = initial_vmin if initial_vmin is not None else img_min
 
    print(f"Image range: [{img_min:.3g}, {img_max:.3g}]  (vmax fixed at {vmax:.3g})")
    print("After each plot closes, type a new vmin and press Enter,")
    print("or just press Enter to accept, or 'q' to abort.\n")
 
    while True:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmin+(vmax-vmin)/5)
        ax.set_title(f"vmin = {vmin:.4g}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.show()  # blocks until the window/figure is closed (or renders inline in notebooks)
 
        raw = input(f"Current vmin = {vmin:.4g}. New vmin (Enter=accept, q=abort): ").strip()
 
        if raw == "":
            print(f"Accepted vmin = {vmin:.4g}")
            return vmin
        if raw.lower() == "q":
            print("Aborted.")
            return None
 
        try:
            vmin = float(raw)
        except ValueError:
            print("Could not parse that as a number, try again.")
            
def find_island_centers(img, vmin, connectivity=8, min_size=4, max_size=20, weighted=True):
    """
    Find center coordinates of connected pixel islands where img > vmin.
 
    Parameters
    ----------
    img : 2D array-like
        The image.
    vmin : float
        Threshold; pixels strictly greater than this are considered part
        of an island.
    connectivity : {4, 8}, optional
        4 = only up/down/left/right neighbors connect a region.
        8 = diagonal neighbors also connect a region (default).
    min_size : int, optional
        Minimum number of pixels for an island to be kept (filters noise).
    weighted : bool, optional
        If True (default), the center is the intensity-weighted centroid
        (center of mass using pixel values as weights) — generally more
        accurate for blobs like beads/PSFs.
        If False, the center is the unweighted geometric centroid of the
        island's pixels.
 
    Returns
    -------
    centers : (N, 2) ndarray
        Row, col (i.e. y, x) coordinates of each island's center, sorted
        by row then col.
    labeled : 2D ndarray
        The label image (0 = background, 1..N = island id), useful for
        debugging / visualization.
    """
    img = np.asarray(img)
    mask = img > vmin
 
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    elif connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        raise ValueError("connectivity must be 4 or 8")
 
    labeled, n_islands = ndimage.label(mask, structure=structure)
    print(labeled.shape)
    if n_islands == 0:
        return np.empty((0, 2)), labeled
 
    # filter out islands smaller than min_size
    if min_size > 1:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_islands + 1))
        keep_labels = np.where((sizes >= min_size)&(sizes <= max_size))[0] + 1
        # relabel keeping only the surviving islands, in order
        new_mask = np.isin(labeled, keep_labels)
        labeled, n_islands = ndimage.label(new_mask, structure=structure)
        if n_islands == 0:
            return np.empty((0, 2)), labeled
 
    if weighted:
        centers = ndimage.center_of_mass(
            img, labeled, index=np.arange(1, n_islands + 1)
        )
    else:
        centers = ndimage.center_of_mass(
            mask, labeled, index=np.arange(1, n_islands + 1)
        )
 
    centers = np.array(centers)  # (N, 2) as (row, col)
    
    centers_corrected = []
    for c in centers:
        if not((c[0]<10) | (c[0]>206 )| (c[1]<10) | (c[1]>146)):
            centers_corrected.append(c)
    centers_corrected = np.array(centers_corrected)
    
    # print island sizes, mean intensities and positions for diagnostics
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_islands + 1))
    means = ndimage.mean(img, labeled, index=np.arange(1, n_islands + 1))
    raw_centers = ndimage.center_of_mass(img, labeled, index=np.arange(1, n_islands + 1))
    print(f"{'Island':>8} {'Size':>8} {'Mean intensity':>16} {'Row':>8} {'Col':>8}")
    for i, (s, m, c) in enumerate(zip(sizes, means, raw_centers)):
        print(f"{i+1:>8} {int(s):>8} {m:>16.1f} {c[0]:>8.1f} {c[1]:>8.1f}")
    return centers_corrected, labeled

 
#%%
def extract_reconstructed(path_raw, first_frame=None, last_frame=None):
    if not os.path.exists(path_raw):
        raise FileNotFoundError(f"File not found: {path_raw}")
    try:
        with tifffile.TiffFile(path_raw) as tif:
            if first_frame is None:
                first_frame=0
                last_frame = len(tif.pages)
            raw = np.zeros((last_frame-first_frame, 216, 160))
            for i in range(last_frame-first_frame):
                raw[i] = tif.pages[first_frame+i].asarray()
        return raw
    except Exception as e:
        print(f"Error reading {path_raw}: {e}")
        return None
#%% 
def detection_beads():
    start_folder = Path(r"/mnt/c/Users/Amaury/Desktop/DATA")

    filename = filedialog.askopenfilename(
        initialdir=start_folder,
        title="Select the file to reconstruct",
        filetypes=[("Images", "*.tif"), ("All files", "*.*")]
    )
    parent = Path(filename).parent
    reference = extract_reconstructed(filename)
    reference = reference[2]+reference[3]
    return reference, parent

#%%
def predetection_zstack_beads():
    ref, parent = detection_beads()
    vmin = pick_threshold(ref) 
    centers, labeled = find_island_centers(ref, vmin)
    print("Found centers (row, col):")
    print(centers)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ref, cmap="gray", vmin=vmin)
    axes[0].set_title("Thresholded image")
    axes[1].imshow(labeled, cmap="nipy_spectral")
    axes[1].set_title(f"{len(centers)} islands found")
    for cy, cx in centers:
        axes[0].plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
    plt.tight_layout()
    plt.show()

    return centers, parent