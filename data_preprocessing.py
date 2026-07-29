#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:56:09 2026

@author: amaury, juliette
"""
import sys
import os
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these'))
os.chdir('/mnt/d/Amaury/DATA')
from reconstruction_juliette import *

cwd = '/mnt/d/Amaury/DATA'
#%% calibration file
deltaz = calibration(cwd)
#%% check the quality of the calibration
check_calib(cwd)
#%% reconstruction
reconstruction_stack(cwd)
#%% pre-localization - adjust the vmin
vmin = find_param_preloc(nb_frames_threshold=1, connectivity=8, min_size=4
                         , max_size=30, weighted=True, bg_subtract=True, bg_kernel_size=100, cwd=cwd)
#%% pre-localization
matplotlib.use("module://matplotlib_inline.backend_inline")
pre_loc_dataset(vmin, deltaz, first_frame=2000, last_frame=None,
                    connectivity=8, min_size=4, max_size=30,
                     weighted=True, plot=False, bg_kernel_size=100, cwd=cwd)
 #%% check the quality - choice of directory
matplotlib.use("module://matplotlib_inline.backend_inline")
directory = (
    Path(filedialog.askdirectory(initialdir=cwd, title="Select the directory containing the tiff files"))
    / "reconstruction"
)
if not directory.is_dir():
    raise NotADirectoryError(f"{directory} is not a valid directory")
#%% check the quality - choice of frame and plot
for frame_number in np.linspace(0, 49999, 10).astype(int):
    verify_preloc(directory, frame_number, cwd)
#%% check pre-localization 
matplotlib.use("module://matplotlib_inline.backend_inline")
directory = (
    Path(filedialog.askdirectory(initialdir=cwd, title="Select the directory containing the tiff files"))
    / "reconstruction"
)
csv_files = list(directory.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        f"Expected exactly one CSV file in {directory}, found {len(csv_files)}."
    )

df = pd.read_csv(csv_files[0])
plt.scatter(df['row'], df['col'], s=0.01, marker='.')
plt.show()
print(df.shape[0])