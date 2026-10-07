# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:42:21 2026

@author: Amaury
amaury.autric@polytechnique.edu
"""
#%%
import sys
import os
import jax
from tkinter import Tk, filedialog
import jax.numpy as jnp
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simu_PSF_polarMFM_JAX import *
import matplotlib.pyplot as plt
from extract_experimental_psf import *
import gc
import optax
import copy
from tqdm import tqdm
import functools
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import ast
import tifffile
import re
# %% PARAMETERS TO BE DEFINED

total_n_frame = 100000
QE = 0.92
EM = 200
# electrons per ADU before the EM register. 15.4 was the value used until October 2026, but the variance vs level
# of the raw frames (photon transfer curve) gives 3.7 to 4.2 e-/ADU on three independent data sets (white lamp at
# EM gain 3, cell background at EM gain 200, actin + fiducial bead at EM gain 250): about 4 times less, so the
# photon numbers were about 4 times too high. The fitted orientations and positions do not depend on this value.
#sensitivity = 15.4
sensitivity = 3.9

look_up_folder = '/mnt/d/Amaury/DATA'
save_folder = Path(filedialog.askdirectory(initialdir=look_up_folder, title="Select the directory containing the tiff files"))
path=(save_folder / "reconstruction")
save_folder=(save_folder / f"NPZ_{datetime.now():%Y-%m-%d_%Hh%M}")
save_folder.mkdir(parents=True, exist_ok=True)
print('Saving in: '+str(save_folder))
# %% functions

def extract_frames(frame_0, N_frame, dimensions):
    error_indices = []
    print('extracting frame '+str(frame_0)+' to '+str(frame_0+N_frame-1))

    def load_single(i, dimensions):
        path_data = str(path) + '/' +str(frame_0+i) + '.tif'
        return i, extract_raw2(path_data, dimensions)

    with ThreadPoolExecutor(max_workers=Nframe) as executor:
        results = list(executor.map(lambda i: load_single(i, dimensions), range(N_frame)))

    for i, raw_ in results:
        if raw_ is None:
            error_indices.append(i)
        else:
            raw[i] = raw_

    return raw, error_indices

def extract_positions(frame_0, N_frame, error_indices):
    index_frame = []
    x, y = [], []
    ind = 0
    for i in range(N_frame):
        if i not in error_indices:
            x__, y__ = position_from_data2(data, frame_0+i)
            x = np.concatenate((x, x__))
            y = np.concatenate((y, y__))
            for k in range(len(x__)):
                index_frame.append(ind)
        ind+=1
    index_frame=np.array(index_frame)
    return x, y, index_frame

def measure_baseline(raw_folder, n_frames=100):
    # camera baseline in ADU (value of a pixel without light), measured on the rows of the raw frames outside the
    # 6 channels: the rows whose level is close to the one of the darkest row, over n_frames frames taken in the
    # middle of the first raw file. Checked on the variance vs level curve of a fiducial bead (zero variance at
    # 178 ADU for a baseline measured at 177 ADU)
    raw_file = sorted(Path(raw_folder).glob('*.ome.tif'))[0]
    with tifffile.TiffFile(raw_file) as tif:
        n = len(tif.pages)
        start = max(0, n//2 - n_frames//2)
        frames = np.stack([tif.pages[i].asarray() for i in range(start, min(n, start+n_frames))]).astype(float)
    rows = np.median(np.median(frames, axis=0), axis=1) # level of each row of the sensor
    dark = rows < rows.min() + 0.2*(np.median(rows) - rows.min())
    print('Camera baseline: '+str(np.median(frames[:, dark]))+' ADU, measured on '+str(np.sum(dark))+' dark rows of '+raw_file.name)
    return np.median(frames[:, dark])

def noise_model(raw, shift_value, n_bins=10):
    # noise of each of the 6 channels of a chunk of frames (frames, channels, H, W), in the units of the data:
    # var(v) = gain*(v + shift). shift = -baseline in these units (value of v without light, given by
    # measure_baseline), the same for the 6 channels. gain is fitted on the photon transfer curve of the chunk:
    # the pixels are sorted by their median over time in n_bins groups, the variance of each group is measured
    # on the differences between consecutive frames (removes the static structure, /2 because a difference has
    # twice the variance) with a MAD (robust to the emitters that blink in a pixel), and gain is the slope of the
    # variance vs (level + shift) through 0. gain includes everything that scales the noise: EM excess noise,
    # real EM gain, normalisation and interpolation of the reconstruction. level is the median background.
    gain, shift, level = np.zeros(raw.shape[1]), np.full(raw.shape[1], shift_value), np.zeros(raw.shape[1])
    for c in range(raw.shape[1]):
        med = np.median(raw[:, c], axis=0)
        dif = np.diff(raw[:, c], axis=0)
        level[c] = np.median(med)
        keep = (med > np.percentile(med, 5)) & (med < np.percentile(med, 90)) # no border, no bright structure
        edges = np.percentile(med[keep], np.linspace(0, 100, n_bins+1))
        levels, variances = [], []
        for k in range(n_bins):
            d = dif[:, keep & (med >= edges[k]) & (med < edges[k+1])]
            levels.append(np.median(med[keep & (med >= edges[k]) & (med < edges[k+1])]))
            variances.append((1.4826*np.median(np.abs(d - np.median(d))))**2 / 2)
        signal = np.array(levels) + shift_value # value above the baseline
        gain[c] = np.sum(signal*np.array(variances)) / np.sum(signal**2) # least squares slope through 0
    return gain, shift, level

def limit(x, lim, slope, upper=True):
    if upper:
        return jnp.sum(jnp.exp((x-lim)*slope))
    else:
        return jnp.sum(jnp.exp(-1*(x-lim)*slope))
    
def loss_pos(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, noise, dim_simu, d_):
    Mj = compute_M_jax(xp=params['xp'], yp=params['yp'], zp=params['zp'], d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    dim_data = 6
    dim_simu = int(dim_simu)
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*Nphotons_speed1)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]

    loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(h+jnp.reshape(params['background']*background_speed, (h.shape[0],3,2))[:, :, :, None, None], -data), axis=(2,)), 2))

    x_bound = limit(params['xp'], 5*0.12, 100, upper=True) + limit(params['xp'], -5*0.12, 100, upper=False)
    y_bound = limit(params['yp'], 5*0.12, 100, upper=True) + limit(params['yp'], -5*0.12, 100, upper=False)
    z_bound = limit(params['zp'], 50., 100, upper=True) + limit(params['zp'], 0, 100, upper=False)
    N_bound = limit(params['N_photons'], 0., 10000, upper=False)
    return (loss +x_bound+y_bound+z_bound+N_bound).astype(jnp.float32)

def loss_angle_with_M(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, zern_speed2_x, zern_speed2_y, data, background, noise, dim_simu, d_):
    # remove plot argument entirely
    dim_data = 6
    dim_simu = int(dim_simu)
    # zernx/zerny are the starting aberrations, params['zern_x'/'zern_y'] the fitted offset shared by the whole batch
    # (an offset rather than a ratio so that a speed of 0 freezes a coefficient at its starting value)
    Mj = compute_M_jax(xp=params['x']*xy_speed2, yp=params['y']*xy_speed2, zp=params['z']*z_speed2, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                   Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base,
                   zernike_coefs_x=jnp.reshape(zernx + params['zern_x']*zern_speed2_x, (3,15)), zernike_coefs_y=jnp.reshape(zerny + params['zern_y']*zern_speed2_y, (3,15)),
                   second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)

    h = PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    model = h + jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None] # expected value per pixel
    # noise model of each channel, var(v) = gain*(v + shift), see noise_model
    gain = noise[:, 0, :, :, None, None]
    shift = noise[:, 1, :, :, None, None]
    if loss2 == 'poisson':
        # (v + shift)/gain follows a Poisson law: its likelihood, sum over the pixels of
        # (mu - (d+shift) log(mu+shift)) / gain. Clipped to keep the log defined
        loss = jnp.sum((model - (data+shift)*jnp.log(jnp.maximum(model+shift, 1e-3))) / gain)
    else:
        # least squares normalised by the variance of each pixel: sum over the pixels of (mu - d)^2 / var.
        # var = gain*(d + shift) is estimated from the data, so it does not depend on the fitted parameters:
        # the weights are fixed during the descent. Clipped at 1 to avoid dividing by ~0
        variance = jnp.maximum(gain*(data + shift), 1.)
        loss = jnp.sum(jnp.pow(model - data, 2) / variance)
    delta_bound = limit(params['delta'], 180, 100, upper=True) + limit(params['delta'], 1, 100, upper=False)
    # N_photons >= 0 (as in SGD1): without it the fit can make the PSF negative to lower the model when the
    # background, fixed after SGD1, is too high. Slope 10 on the scaled parameter (N/nphotons_speed2): the penalty
    # is negligible above ~0.5*nphotons_speed2 photons and stays finite in float32 for an overshoot of a few steps
    N_bound = limit(params['N_photons'], 0., 10, upper=False)
    #rho_bound = limit(params['rho'], 0, 50, upper=False)
    return (loss + 1000.*(delta_bound + N_bound)).astype(jnp.float32)

def plot_results(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, noise, dim_simu, d_):
    dim_data = 6
    dim_simu = int(dim_simu)
    x_fine = np.array(params['x'] * xy_speed2)
    y_fine = np.array(params['y'] * xy_speed2)
    Mj = compute_M_jax(xp=params['x']*xy_speed2, yp=params['y']*xy_speed2, zp=params['z']*z_speed2, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                    Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, 
                    zernike_coefs_x=jnp.reshape(zernx, (3,15)), zernike_coefs_y=jnp.reshape(zerny, (3,15)),
                    second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    h = np.array(PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1])
    data = np.array(data)
    
    rho = np.array(params['rho'])
    eta = np.array(params['eta'])
    delta = np.array(params['delta'] * delta_speed)
    N_photons = np.array(params['N_photons'] * nphotons_speed2)
    z = np.array(params['z'] * z_speed2)
    background_arr = np.array(jnp.reshape(background, (h.shape[0],3,2)))
    for nb in range(data.shape[0]):
        # one figure per PSF: rows are the planes, columns data x, data y, fit x, fit y
        if N_photons[nb]>1520: # fiducial (6000 with sensitivity 15.4), each image with its own scale, the fit without background
            fit = h[nb]
            scale = {}
            name = 'fiducial'
        else: # same scale for data and fit, the fit with the background
            fit = background_arr[nb].mean()+h[nb]
            scale = {'vmin': min(np.min(data[nb]), np.min(fit)), 'vmax': max(np.max(data[nb]), np.max(fit))}
            name = 'PSF'
        fig, ax = plt.subplots(3, 4, figsize=(10, 7))
        for p in range(3):
            for c, (image, title) in enumerate([(data[nb,p,0], 'data x'), (data[nb,p,1], 'data y'),
                                                (fit[p,0], 'fit x'), (fit[p,1], 'fit y')]):
                ax[p,c].imshow(image, cmap='gray', **scale)
                ax[p,c].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
                ax[p,c].set_xticks([])
                ax[p,c].set_yticks([])
                if p == 0:
                    ax[p,c].set_title(title)
            ax[p,0].set_ylabel('plane '+str(p))
        plt.suptitle(f'{name} {nb} | rho={float(rho[nb]):.3f} eta={float(eta[nb]):.3f} delta={float(delta[nb]):.2f} N={float(N_photons[nb]):.0f} z={float(z[nb]):.2f} bg={float(background_arr[nb].mean()):.2f}')
        plt.show()

@functools.partial(jax.jit, static_argnames=['dim_simu'])
def score_eval(M_, rho, eta, delta, N_photons, data, background, noise, dim_simu):
    dim_data = 6
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    # loss of SGD2 for each PSF, with the loss chosen by loss2 and the noise model of its channels
    # (same formulas as loss_angle_with_M, summed over the pixels of each PSF only)
    model = h + jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]
    gain = noise[:, 0, :, :, None, None]
    shift = noise[:, 1, :, :, None, None]
    if loss2 == 'poisson':
        score = jnp.sum((model - (data+shift)*jnp.log(jnp.maximum(model+shift, 1e-3))) / gain, axis=(1,2,3,4))
    else:
        score = jnp.sum(jnp.pow(model - data, 2) / jnp.maximum(gain*(data + shift), 1.), axis=(1,2,3,4))
    # Poisson deviance with the same noise model, per pixel: 2 sum[(mu+shift) - (d+shift) + (d+shift) log((d+shift)/(mu+shift))]/gain
    # divided by the number of pixels. Close to 1 when the model describes the data whatever the photon number and
    # the background (unlike the score), larger where the model does not fit. A pixel with d+shift <= 0 has no log term
    data_shift = data + shift
    model_shift = jnp.maximum(model + shift, 1e-3)
    log_term = jnp.where(data_shift > 0, data_shift*jnp.log(jnp.maximum(data_shift, 1e-3)/model_shift), 0.)
    deviance = 2*jnp.sum((model_shift - data_shift + log_term) / gain, axis=(1,2,3,4)) / (data[0].size)
    return score, deviance

@functools.partial(jax.jit, static_argnames=['dim_simu'])
def eval_batch(x_found, y_found, z_found, zernx, zerny, rho_found, eta_found, delta_found, N_found2, noisy_psf, background, noise, dim_simu):
    M = compute_M_jax(xp=x_found, yp=y_found, zp=z_found, d=d_, x=xx, y=yy, th1=th1, phi=phi, 
                      Ex0=Ex0, Ex1=Ex1, Ex2=Ex2, Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, 
                      zernike_base=zernike_base, zernike_coefs_x=zernx, zernike_coefs_y=zerny,
                      second_plane=second_plane, polar_projections=polar_projections, 
                      lambd=lambd, f_tube=f_tube)
    return score_eval(M, rho_found, eta_found, delta_found, N_found2, noisy_psf, background, noise, dim_simu)
def parse_config(config_path):
    cfg = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            try:
                value = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                value = value.strip() # date and paths are kept as strings
            cfg[key.strip()] = np.array(value) if isinstance(value, list) else value
    return cfg

# the keys of config.txt are the variable names themselves, so they can be reinjected directly
reloaded = ['QE', 'EM', 'sensitivity',
            'lambda_emission', 'middle_plane', 'interplane', 'd',
            'rotation', 'J1', 'J2', 'J_dichroic',
            'polar_projections', 'N', 'l_pixel', 'NA', 'mag', 'f_tube', 'MAG', 'SAF',
            'zernike_coefs_x', 'zernike_coefs_y', 'zern_x', 'zern_y',
            'LR1', 'num_epochs_max1', 'Nphotons_speed1', 'background_speed',
            'LR2', 'num_epochs_max2', 'delta_speed', 'nphotons_speed2', 'xy_speed2', 'z_speed2',
            'zern_speed2_x', 'zern_speed2_y', 'zern_start_epoch2', 'fit_zernike', 'loss2',
            'Nframe', 'last_frame_processed', 'NPSF', 'n_photons_filtering', 'batch_nb', 'dimensions']

def reload_config(config_file):
    # overwrites the parameters of the script with the ones stored in a config.txt, returns the whole config
    old_config = parse_config(config_file)
    print('Reloading the parameters of the run of '+str(old_config.get('date', 'unknown date')))
    for key in reloaded:
        if key in old_config:
            globals()[key] = old_config[key]
            print('  '+key+' = '+str(old_config[key]))
        else:
            print('  '+key+' is missing from the file, keeping the value defined above')
    return old_config

#%% extracting positions/pre-loc
csv_files = list(path.glob("*.csv"))
data = pos_from_csv(csv_files[0])
match = re.search(r'_(\d+)\.csv$', csv_files[0].name)
if match:
    number = int(match.group(1))
else:
    raise ValueError("No ending number found")
interplane = jax.device_put(number/1000)
print('Interplane: '+str(interplane)+'um')
# camera baseline (ADU), from the raw files next to the reconstruction, used by the noise model
baseline_adu = measure_baseline(path.parent)
#%% defining useful variables

lambda_emission = jax.device_put(620) # nm
middle_plane = jax.device_put(1.)
d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])


# %% calibration data

def rot(angle):
    angle=angle*np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
'''
J1 = np.array([[ 0.77294344        ,             -0.37847298 + 1j*  -0.5097466 ],[
      -0.24436265 + 1j*  0.58565116  ,   -0.7503899 + 1j*  0.18626373 ]])
J2 = np.array([[ 0.22273345               ,      -0.8014731 + 1j*  -0.55417156 ],[
      0.48960716 + 1j*  0.84284395   ,  -0.017539864 + 1j*  0.2226117 ]])
'''
'''
J1 = np.array([[ 0.86687591+0.j        , -0.57183764-0.13293168j],
       [ 0.47786187+0.14203586j,  0.74024064+0.32768077j]])
J2 = np.array([[ 0.31561277+0.j        , -0.95752769-0.05599034j],
       [-0.65325986-0.68821518j, -0.19602931-0.2039076j ]])
'''

J1 = np.array([[ 0.89097912+0.j        , -0.25009759-0.43946158j],
       [ 0.43033225-0.14481147j,  0.6335862 +0.58557087j]])
J2 = np.array([[ 0.31676515+0.j        , -0.93297585-0.07330244j],
       [-0.58381747+0.74754063j, -0.26412184+0.23328625j]])

rotation = 7
rotation2 =  2
J_dichroic = np.array([J1@rot(rotation), J2@rot(-rotation2), J1@rot(rotation)])
rho_offset = -4

# %% SGD PARANETERS TO DEFINE
# the settings in photons (speeds of SGD1 and SGD2, starting N in load_batch, fiducial threshold in plot_results)
# were tuned with sensitivity = 15.4 (2000, 100, 50, 3000, 6000) and are multiplied by 3.9/15.4 for sensitivity = 3.9.
# With the data ~4 times smaller and the same numbers, SGD1 started with ~2 times too many photons and pushed the
# PSF out of focus to remove them from the patch: z ended ~1.3 um too high (SLB 2026_02_02: 1.91 um instead of
# 0.40 um, 0.40 um again with the scaled values). To change together with sensitivity
Nphotons_speed1 = jax.device_put(506.)
background_speed = jax.device_put(25.3)
LR1 = jax.device_put(0.05)
num_epochs_max1 = 80

num_epochs_max2 = 120
LR2 = jax.device_put(1.2)
delta_speed = jax.device_put(1.8)
nphotons_speed2 = jax.device_put(12.7)
xy_speed2 = jax.device_put(1/70)
z_speed2=jax.device_put(1/70)
# relative learning rates of the aberrations fitted in SGD2, one per coefficient: 3 planes x 15 Noll modes,
# index = 15*plane + mode, for each polarisation channel. Coefficients in radians, fitted as an offset
# shared by all the PSF of the batch around zern_x/zern_y. 0 freezes a coefficient (all 0 = no aberration fit)
# only primary astigmatism (Noll 5, 6) and primary spherical (Noll 11) are fitted, the other modes stay at 0
fitted_noll = [5, 6, 11]
zern_fitted = jnp.tile(jnp.isin(jnp.arange(1, 16), jnp.array(fitted_noll)), 3).astype(jnp.float32) # 1 for a fitted coefficient
zern_speed2_x = jax.device_put(0.007*zern_fitted)
zern_speed2_y = jax.device_put(0.007*zern_fitted)
# False: no aberration in SGD2, all the Zernike coefficients stay at 0 whatever zern_x/zern_y and the speeds
fit_zernike = False
zern_start_epoch2 = 50 # the aberrations stay at zern_x/zern_y during the first epochs of SGD2, then are fitted
# loss of SGD2: 'poisson' = Poisson likelihood (default), 'lms' = least squares normalised by the variance of
# each pixel estimated from the data. Both use the noise model of noise_model. The saved score (eval_batch) is
# this loss for each PSF. The cells defining step2 and eval_batch must be run again after a change (jit).
# 'lms' is biased at low photon numbers: a pixel that fluctuates low gets a small variance, so a large weight,
# and pulls the model down (about -1 photon per pixel for a background of a few photons, N_photons went to
# -3000 on the SLB data of 2026_02_02 with a background of ~3 photons above the baseline). Fine for backgrounds
# of tens of photons (cells). The Poisson likelihood has no such bias
loss2 = 'poisson'

# extraction parameters
Nframe= 20 # nb of frame per batch of extraction
last_frame_processed=2000 #starting point 
NPSF = 100 # nb of PSF per batch

n_photons_filtering = 100

# nb of batch of SGD
batch_nb = 30000
# changed only by the resume sub-option of the OPTIONAL cell, keep these values for a new run
first_batch = 0 # index of the first npz file written
n_psf_to_skip = 0 # PSF of the first frame that were already saved by the resumed run
config_name = 'config.txt'
dimensions = [215,160] # the dimension of the channels can slightly vary depending on the reconstruction file

# microscope parameters
polar_projections = jax.device_put(jnp.array([0, 45, 0]))
N=jax.device_put(jnp.array(80))
l_pixel=jax.device_put(jnp.array(16))
NA=jax.device_put(jnp.array(1.4))
mag=jax.device_put(jnp.array(100))
f_tube=jax.device_put(jnp.array(200))
MAG=jax.device_put(jnp.array(200/150))

SAF = True

# aberrations, 0 everywhere means a perfect system. zernike_coefs_x/y are used by the
# first SGD, zern_x/zern_y by the second one (starting point of the aberration fit), in rad
zernike_coefs_x = jnp.zeros((3,15)).astype(jnp.complex64)
zernike_coefs_y = jnp.zeros((3,15)).astype(jnp.complex64)
# mean over the 221 batches of the SGD2 aberration fit of fit_2026-10-02_19h42.csv
# (2026_02_02_SLB_1um_new_process/SM_tres_haut), one row per plane, Noll index 1 to 15,
# only primary astigmatism (Noll 5, 6) and primary spherical (Noll 11) are kept
zern_x = jnp.zeros(3*15) # start from a perfect system, the means of the previous fit are kept below
''' jnp.array([
    0.000000, 0.000000, 0.000000, 0.000000, 0.064880, -0.307965, 0.000000, 0.000000, 0.000000, 0.000000, 0.053822, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.420010, -0.011255, 0.000000, 0.000000, 0.000000, 0.000000, 0.205577, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, -0.001132, -0.183550, 0.000000, 0.000000, 0.000000, 0.000000, 0.203165, 0.000000, 0.000000, 0.000000, 0.000000])
'''
zern_y = jnp.zeros(3*15)
''' jnp.array([
    0.000000, 0.000000, 0.000000, 0.000000, -0.091603, -0.102861, 0.000000, 0.000000, 0.000000, 0.000000, 0.135169, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.185555, -0.316491, 0.000000, 0.000000, 0.000000, 0.000000, 0.281857, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.214820, -0.010762, 0.000000, 0.000000, 0.000000, 0.000000, 0.184550, 0.000000, 0.000000, 0.000000, 0.000000])
'''
#%%   #### OPTIONAL - reloading the parameters of a previous run instead of the cells above ####
# run this cell only if you want to reproduce an old run: it overwrites the parameters
# defined above with the ones stored in the config.txt of that run

config_file = filedialog.askopenfilename(initialdir=look_up_folder,
                                         title="Select the config.txt of the run to reproduce",
                                         filetypes=[("Config file", "config*.txt"), ("All files", "*.*")])
old_config = reload_config(config_file)

#%%   #### OPTIONAL - resuming a run that stopped ####
# run this cell (and not the one above) to continue a run that stopped, in its own NPZ folder and from the
# frame where it stopped: it reloads the parameters of the run and finds in its npz files where it stopped.
# The raw files are the ones selected at the beginning of the script. Do not run the parameter cells
# above after this one, they would set the starting point back to a new run

resume_folder = Path(filedialog.askdirectory(initialdir=look_up_folder,
                                             title="Select the NPZ folder of the run to resume"))
old_config = reload_config(resume_folder / 'config.txt')
if old_config.get('reconstruction_path', str(path)) != str(path):
    print('WARNING: the stopped run processed '+str(old_config['reconstruction_path'])+' but '+str(path)+' is selected')

frames_done, batches_done = [], []
for npz_file in resume_folder.glob('*.npz'):
    if not npz_file.stem.isdigit():
        continue
    try:
        with np.load(npz_file) as npz:
            frames_done.append(npz['frame'])
        batches_done.append(int(npz_file.stem))
    except Exception as error: # typically the file being written when the run stopped
        print('  '+npz_file.name+' cannot be read, its batch will be processed again ('+str(error)+')')
if len(batches_done) == 0:
    raise ValueError('No readable npz file in '+str(resume_folder)+', nothing to resume')
frames_done = np.concatenate(frames_done)
last_frame_saved = int(np.max(frames_done))
# a batch ends in the middle of a frame: the last saved frame is extracted again and its PSF
# already saved are skipped, the order of the PSF inside a frame being the one of the csv
last_frame_processed = last_frame_saved - 1
n_psf_to_skip = int(np.sum(frames_done == last_frame_saved))
first_batch = max(batches_done) + 1
if save_folder != resume_folder and save_folder.is_dir() and not any(save_folder.iterdir()):
    save_folder.rmdir() # the empty NPZ folder created at the beginning
save_folder = resume_folder
config_name = f'config_resumed_{datetime.now():%Y-%m-%d_%Hh%M}.txt' # the config.txt of the run is kept
print('Resuming in '+str(save_folder)+' at batch '+str(first_batch)+', frame '+str(last_frame_saved)
      +' ('+str(n_psf_to_skip)+' PSF of this frame already saved)')

#%%   #################### saving the configuration of the run ##################


config = {
    'date': f'{datetime.now():%Y-%m-%d %H:%M:%S}',
    'reconstruction_path': str(path),
    # camera
    'QE': QE,
    'EM': EM,
    'sensitivity': sensitivity,
    'baseline_adu': baseline_adu, # measured on the raw files, not reloaded
    # microscope and planes
    'lambda_emission': lambda_emission,
    'middle_plane': middle_plane,
    'interplane': interplane,
    'd': d,
    # polarisation calibration
    'rotation': rotation,
    'rotation2': rotation2,
    'rho_offset': rho_offset,
    'J1': J1,
    'J2': J2,
    'J_dichroic': J_dichroic,
    # objective and back focal plane
    'polar_projections': polar_projections,
    'N': N,
    'l_pixel': l_pixel,
    'NA': NA,
    'mag': mag,
    'f_tube': f_tube,
    'MAG': MAG,
    'SAF': SAF,
    # aberrations
    'zernike_coefs_x': zernike_coefs_x,
    'zernike_coefs_y': zernike_coefs_y,
    'zern_x': zern_x,
    'zern_y': zern_y,
    # SGD 1 - position, photons, background
    'LR1': LR1,
    'num_epochs_max1': num_epochs_max1,
    'Nphotons_speed1': Nphotons_speed1,
    'background_speed': background_speed,
    # SGD 2 - orientation
    'LR2': LR2,
    'num_epochs_max2': num_epochs_max2,
    'delta_speed': delta_speed,
    'nphotons_speed2': nphotons_speed2,
    'xy_speed2': xy_speed2,
    'z_speed2': z_speed2,
    'zern_speed2_x': zern_speed2_x,
    'zern_speed2_y': zern_speed2_y,
    'zern_start_epoch2': zern_start_epoch2,
    'fit_zernike': fit_zernike,
    'loss2': loss2,
    # extraction
    'Nframe': Nframe,
    'last_frame_processed': last_frame_processed,
    'NPSF': NPSF,
    'n_photons_filtering': n_photons_filtering,
    'batch_nb': batch_nb,
    'dimensions': dimensions,
    # resume, 0 for a new run
    'first_batch': first_batch,
    'n_psf_to_skip': n_psf_to_skip,
}

with open(save_folder / config_name, 'w') as f:
    f.write('# 4polarMFM SGD run configuration, reloadable by the OPTIONAL cell of SGD_jax.py\n')
    f.write('# lengths in um, angles in degrees, wavelengths in nm\n')
    for key, value in config.items():
        if isinstance(value, str):
            line = value
        else:
            line = np.array2string(np.asarray(value), separator=', ', max_line_width=10**6, floatmode='unique')
            line = line.replace('\n', '') # 2D arrays are printed over several rows, one key per line is needed to reload
        f.write(f'{key} = {line}\n')
print('Config saved in: '+str(save_folder / config_name))

#%%   #################### gradient descent ##################

# quantities derived from the parameters above
d_ = jax.device_put(jnp.array([-float(d[1]) for k in range(NPSF)]))
second_plane = jax.device_put(jnp.array([d[1]-d[0], 0, d[1]-d[2]]))
lambd=jax.device_put(jnp.array(lambda_emission))

if SAF:
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k_, f_o, costh2 = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic, SAF=SAF)
else:
    costh2=None
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k_, f_o = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic, SAF=SAF)

u, v, Npadding = padding_jax(r, r_cut, k_, f_o,  N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, 
           f_tube=f_tube, MAG=MAG)

phase_mask = jnp.stack([jnp.ones((N,N)), jnp.ones((N,N)), jnp.ones((N,N))])
zernike_base = generate_zernike_base_jax(r_cut=r_cut, N=N, zernike_order=4, skip_indices={0, 1, 2, 3}) # piston, tip, tilt and defocus are degenerate with x, y, z
active_modes = np.where(np.any(np.array(zernike_base)!=0, axis=(1,2)))[0] # fitted modes, used for the plot of SGD2

xx = pad_jax(xx, Npadding).astype(jnp.complex64)
yy = pad_jax(yy, Npadding).astype(jnp.complex64)
th1 = pad_jax(th1, Npadding).astype(jnp.complex64)
phi = pad_jax(phi, Npadding).astype(jnp.complex64)
Ex0 = pad_jax(Ex0, Npadding).astype(jnp.complex64)
Ex1 = pad_jax(Ex1, Npadding).astype(jnp.complex64)
Ex2 = pad_jax(Ex2, Npadding).astype(jnp.complex64)
Ey0 = pad_jax(Ey0, Npadding).astype(jnp.complex64)
Ey1 = pad_jax(Ey1, Npadding).astype(jnp.complex64)
Ey2 = pad_jax(Ey2, Npadding).astype(jnp.complex64)
phase_mask = pad_jax(phase_mask, Npadding).astype(jnp.complex64)
zernike_base = pad_jax(zernike_base, Npadding).astype(jnp.complex64)
if SAF:
    costh2 = pad_jax(costh2, Npadding).astype(jnp.complex64)

# strating parameters (could be a first evaluation with coarse algo)
x_start = jax.device_put(jnp.array([0. for k in range(NPSF)])).astype(jnp.float32)
y_start = jax.device_put(jnp.array([0. for k in range(NPSF)])).astype(jnp.float32)
z_exp =  jax.device_put(jnp.array([float(d[1])*0.7 for k in range(NPSF)])).astype(jnp.float32)

# gradient descent parameters
rho_start = jnp.array([90. for k in range(NPSF)]).astype(jnp.float32)
eta_start = jnp.array([90. for k in range(NPSF)]).astype(jnp.float32)
delta_start = jnp.array([80. for k in range(NPSF)]).astype(jnp.float32)
Nstart_test = jnp.array([3000. for k in range(NPSF)]).astype(jnp.float32)

Mtest = compute_M_jax(xp=x_start, yp=y_start, zp=z_exp, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                , zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                ,  second_plane=second_plane
              , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
htest = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=Nstart_test).astype(jnp.float32)
dim_simu = int(htest.shape[-1]//2)


def load_batch(last_frame_processed, buffer, psf_buffer, NPSF, result, n_skip=0):
    # n_skip: number of first PSF to drop, already saved by the run that is resumed
    bufferx, buffery, bufferindex, buffernoise = buffer
    x, y, index_frame = bufferx, buffery, bufferindex
    # noise of each PSF: gain, shift and background level of its 3 planes x 2 polarisations (see noise_model),
    # measured on the chunk of frames the PSF comes from and kept with it in the buffer
    psf_noise = buffernoise
    single_psf = psf_buffer
    while len(x)<NPSF+n_skip:
        #t0 = time.time()
        raw, error_indices = extract_frames(last_frame_processed+1, Nframe, dimensions)
        #print(f'extract_frames: {time.time()-t0:.2f}s')
        x_, y_, index_frame_ = extract_positions(last_frame_processed+1, Nframe, error_indices)
        #print(f'extract_positions: {time.time()-t0:.2f}s')
        # converting to photon count
        raw = raw*sensitivity/(QE*EM)
        gain, shift, level = noise_model(raw, -baseline_adu*sensitivity/(QE*EM))
        background = np.mean(level)
        print('noise of frames '+str(last_frame_processed+1)+' to '+str(last_frame_processed+Nframe)+', var = gain*(value + shift), shift = -baseline')
        print('  gain  '+str(np.round(gain, 2))+'\n  shift '+str(np.round(shift, 1))+'\n  background '+str(np.round(level, 1)))
        L = raw.shape[2]*120
        W = raw.shape[3]*120
        # removing all the PSF where a parameter is evaluated to nan in Louise pipeline
        nb = len(x_)
        
        L = raw.shape[2]*120
        W = raw.shape[3]*120
        for k in range(nb-1, -1, -1):  # iterate backwards to safely delete
            if np.isnan(x_[k]) or np.isnan(y_[k]) or \
               (y_[k]<7*120) or (x_[k]<7*120) or (x_[k]>L-7*120) or (y_[k]>W-7*120):
                x_ = np.delete(x_, k, 0)
                y_ = np.delete(y_, k, 0)
                index_frame_ = np.delete(index_frame_, k, 0)
           
        index_frame_ = (last_frame_processed+index_frame_+1).astype(int)
        
        # extracting the psf from the files
        single_psf_ = extract_raw_xy(raw[0], x_[index_frame_==last_frame_processed+1], y_[index_frame_==last_frame_processed+1])

        for i in range(1, Nframe):
            frame_id = last_frame_processed + 1 + i
            single_psf_ = np.concatenate((single_psf_, extract_raw_xy(raw[i], x_[index_frame_==frame_id], y_[index_frame_==frame_id])))
        
        # dimenstion matching to have x in horizontal and y in vertical when considering what appears in a tiff file
        single_psf_ = single_psf_[:,::-1,:,::-1,:]
        x_, y_ = y_, -x_
        
        x = np.concatenate((x, x_))
        y = np.concatenate((y, y_))
        index_frame = np.concatenate((index_frame, index_frame_))
        single_psf = np.concatenate((single_psf, single_psf_))
        # channel 2*plane+polarisation of raw, as in extract_raw_xy, then planes reversed as single_psf_
        chunk_noise = np.stack((gain, shift, level))[:, np.arange(6).reshape(3, 2)[::-1]]
        psf_noise = np.concatenate((psf_noise, np.tile(chunk_noise, (len(single_psf_), 1, 1, 1))))
        
        # filter after concatenation, inside while loop
        n_pixels = single_psf.shape[2] * single_psf.shape[3] * single_psf.shape[4]
        Nstart_by_plane_full = np.sum(single_psf, axis=(2,3,4)) - background * n_pixels
        nb_full = len(x)
        '''
        for k in range(nb_full-1, -1, -1):
            if ((Nstart_by_plane_full[k,0] > Nstart_by_plane_full[k,1]) and \
                (Nstart_by_plane_full[k,2] > Nstart_by_plane_full[k,1])) or \
               (Nstart_by_plane_full[k,0] + Nstart_by_plane_full[k,1] + Nstart_by_plane_full[k,2] < n_photons_filtering):
                x = np.delete(x, k, 0)
                y = np.delete(y, k, 0)
                index_frame = np.delete(index_frame, k, 0)
                single_psf = np.delete(single_psf, k, 0)
        '''
        last_frame_processed+=Nframe

    x, y, index_frame, single_psf, psf_noise = x[n_skip:], y[n_skip:], index_frame[n_skip:], single_psf[n_skip:], psf_noise[n_skip:]
    buffer = x[NPSF:], y[NPSF:], index_frame[NPSF:], psf_noise[NPSF:]
    psf_buffer = single_psf[NPSF:]
    noisy_psf = single_psf[:NPSF]
    x, y = x[:NPSF], y[:NPSF]
    index_frame = index_frame[:NPSF]
    psf_noise = psf_noise[:NPSF]
    
    result['buffer'] = buffer        
    result['psf_buffer'] = psf_buffer 
    result['noisy_psf'] = jnp.array(noisy_psf)
    result['x'] = jnp.array(x)
    result['y'] = jnp.array(y)
    result['Nstart'] = jnp.array([760. for i in range(NPSF)]).astype(jnp.float32) # 3000 with sensitivity 15.4. Other start: #jnp.array(jnp.sum(Nstart_by_plane, axis=1)).astype(jnp.float32)
    result['background_array'] = jnp.array(psf_noise[:, 2]).astype(jnp.float32) # starting background of each channel
    result['noise'] = jnp.array(psf_noise[:, :2]).astype(jnp.float32) # (NPSF, gain/shift, 3, 2)
    result['frame'] = index_frame
    result['last_frame_processed'] = last_frame_processed

# functions for the SGD steps
@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step1(params, opt_state, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, noise, dim_simu, d_):
    loss, grads = jax.value_and_grad(loss_pos)(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, noise, dim_simu, d_)
    updates, opt_state = optimizer1.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step2(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, zern_speed2_x, zern_speed2_y, data, background, noise, dim_simu, d_):
    loss, grads = jax.value_and_grad(loss_angle_with_M)(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, zern_speed2_x, zern_speed2_y, data, background, noise, dim_simu, d_)
    updates, opt_state = optimizer2.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

optimizer1 = optax.adam(learning_rate=LR1)
optimizer2 = optax.adam(learning_rate=LR2)


################# main loop ########################
if not fit_zernike: # no aberration in SGD2, applied here so that it also holds after reloading an old config
    zern_x, zern_y = jnp.zeros(3*15), jnp.zeros(3*15)
    zern_speed2_x, zern_speed2_y = jnp.zeros(3*15), jnp.zeros(3*15)
    print('Zernike aberrations off: all the coefficients stay at 0 in SGD2')

# working buffers of the extraction, empty at the start of a launch (new or resumed run)
raw = np.zeros((Nframe,6,dimensions[0],dimensions[1]))
buffer = (np.array([]), np.array([]), np.array([]), np.empty((0, 3, 3, 2))) # x, y, frame, noise of each PSF
psf_buffer = np.empty((0, 3, 2, 13, 13))  # adjust shape to match your PSF dimensions
first_loop_of_the_launch = True
print('Starting at batch '+str(first_batch)+' after frame '+str(last_frame_processed)+', saving in '+str(save_folder))
for batch in range(first_batch, batch_nb):

    current = {}
    load_batch(last_frame_processed, buffer, psf_buffer, NPSF, current, n_skip=n_psf_to_skip)
    n_psf_to_skip = 0 # only the first batch of a resumed run skips PSF
    buffer = current.get('buffer', buffer)
    psf_buffer = current.get('psf_buffer', psf_buffer)
    last_frame_processed = current.get('last_frame_processed', last_frame_processed)

    # build params from current batch
    noisy_psf = current['noisy_psf']  
    noise = current['noise']        
    x = current['x']  
    y = current['y']  
    frame = current['frame'] 
    params = {
        'xp': x_start,
        'yp': y_start,
        'zp': z_exp,
        'N_photons': current['Nstart'] / Nphotons_speed1,
        'background': current['background_array'].flatten() / background_speed
    }

    angle_rd1 = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
    angle_rd2 = jnp.array([45. for k in range(NPSF)]).astype(jnp.float32)
    optimizer = optax.adam(learning_rate=LR1)
    opt_state = optimizer.init(params)
    
    loss_ = []
    z__ = []
    N__ = []
    x__ =[]
    bck = []
    
    for i in tqdm(range(num_epochs_max1)):
        params, opt_state, loss = step1(params, opt_state, Nphotons_speed1, background_speed, angle_rd2, angle_rd2, angle_rd1, noisy_psf, second_plane, noise, dim_simu, d_)
        loss_.append(float(loss))
        z__.append(np.array(params['zp']))
        N__.append(np.array(params['N_photons'] * Nphotons_speed1))
        x__.append(np.array(params['xp']))
        bck.append(np.array(params['background'] * background_speed))
    if batch<30:
        fig, ax = plt.subplots(2,3)
        ax[0,0].plot(loss_)
        ax[0,0].set(title='loss', xlabel='epoch', ylabel='loss (a.u.)')
        ax[0,1].plot(z__)
        ax[0,1].set(title='axial position', xlabel='epoch', ylabel='z ($\\mu$m)')
        ax[0,2].plot(N__)
        ax[0,2].set(title='photon budget', xlabel='epoch', ylabel='N photons')
        ax[1,0].plot(x__)
        ax[1,0].set(title='lateral position', xlabel='epoch', ylabel='x ($\\mu$m)')
        ax[1,1].plot(bck)
        ax[1,1].set(title='background', xlabel='epoch', ylabel='background (photons/pixel)')
        ax[1,2].axis('off')
        fig.suptitle('SGD 1 - position, photons, background')
        #fig.tight_layout()
        plt.show()
        del(ax, loss_)
    del(z__, N__, x__, bck)

    x_found = params['xp']
    y_found = params['yp']
    z_found = params['zp']
    N_found = params['N_photons'] * Nphotons_speed1
    background_array_found = params['background'] * background_speed
    del(params, loss)
    

################################ second SGD on orientation #########################################################################################
    
    params = {
    'rho': rho_start,
    'eta': eta_start,
    'delta': delta_start/delta_speed,
    'N_photons': N_found/nphotons_speed2,
    'x': x_found/xy_speed2,
    'y': y_found/xy_speed2,
    'z': z_found/z_speed2,
    'zern_x': jnp.zeros(3*15),
    'zern_y': jnp.zeros(3*15)
    }
    optimizer = optax.adam(learning_rate=LR2)
    opt_state = optimizer.init(params)
    
    loss_ = []
    eta_ = []
    rho_ = []
    delta_ = []
    x_ = []
    z_ = []
    Np_ = []
    zx_ = []
    zy_ = []
    
    for i in tqdm(range(num_epochs_max2)):
        # a speed of 0 gives a gradient of 0: the aberrations do not move before zern_start_epoch2
        zern_on = float(i >= zern_start_epoch2)
        params, opt_state, loss = step2(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y, zern_speed2_x*zern_on, zern_speed2_y*zern_on, noisy_psf, background_array_found, noise, dim_simu, d_)
        loss_.append(float(loss))
        rho_.append(np.array(params['rho']))
        eta_.append(np.array(params['eta']))
        delta_.append(np.array(params['delta'] * delta_speed))
        x_.append(np.array(params['x'] * xy_speed2))
        z_.append(np.array(params['z'] * z_speed2))
        Np_.append(np.array(params['N_photons'] * nphotons_speed2))
        zx_.append(np.array(zern_x + params['zern_x'] * zern_speed2_x) * 1000/(2*np.pi)) # rad to milli-waves
        zy_.append(np.array(zern_y + params['zern_y'] * zern_speed2_y) * 1000/(2*np.pi))
    #jax.debug.print("rho_found: {}", np.array(params['rho']))
    if batch<30:
        fig, ax = plt.subplots(4,2, figsize=(12,14))
        ax[0,0].plot(loss_)
        ax[0,0].set(title='loss', xlabel='epoch', ylabel='loss (a.u.)')
        ax[0,1].plot(eta_)
        ax[0,1].set(title='out-of-plane angle', xlabel='epoch', ylabel='$\\eta$ (deg)')
        rho_ = np.array(rho_)
        eta_ = np.array(eta_)
        Np_ = np.array(Np_)
        ax[1,0].plot(rho_)
        ax[1,0].set(title='in-plane angle', xlabel='epoch', ylabel='$\\rho$ (deg)')
        delta_ = np.array(delta_)
        ax[1,1].plot(delta_)
        ax[1,1].set(title='wobbling cone', xlabel='epoch', ylabel='$\\delta$ (deg)')
        ax[2,0].plot(x_)
        ax[2,0].set(title='lateral position', xlabel='epoch', ylabel='x ($\\mu$m)')
        ax[2,1].plot(z_)
        ax[2,1].set(title='axial position', xlabel='epoch', ylabel='z ($\\mu$m)')
        ax[3,0].plot(Np_)
        ax[3,0].set(title='photon budget', xlabel='epoch', ylabel='N photons')
        ax[3,1].hist((params['rho']%180)[Np_[-1]<10000])
        ax[3,1].hist((params['eta']%180)[Np_[-1]<10000], alpha=0.5)
        ax[3,1].set(title='final $\\rho$ and $\\eta$ (N < 10000)', xlabel='$\\rho$ mod 180 (deg)', ylabel='count')
        fig.suptitle('SGD 2 - orientation')
        #fig.tight_layout()
        del(fig, ax)
        plt.show()

        # aberrations, only the modes that exist in zernike_base (the skipped ones are 0 everywhere)
        zx_ = np.reshape(np.array(zx_), (-1,3,15))
        zy_ = np.reshape(np.array(zy_), (-1,3,15))
        fig, ax = plt.subplots(3,2, figsize=(12,10), sharex=True)
        for p in range(3):
            for c, (z_pol, pol) in enumerate([(zx_, 'x'), (zy_, 'y')]):
                for m in active_modes:
                    ax[p,c].plot(z_pol[:,p,m], label='Z'+str(m+1))
                ax[p,c].set(title='plane '+str(p)+' - '+pol+' polarisation', ylabel='coefficient (m$\\lambda$)')
        ax[2,0].set(xlabel='epoch')
        ax[2,1].set(xlabel='epoch')
        ax[0,1].legend(title='Noll index', fontsize=8, loc='upper left', bbox_to_anchor=(1.01,1))
        fig.suptitle('SGD 2 - aberrations')
        del(fig, ax)
        plt.show()
    
    '''
    mask_red = (rho_[-1, :] % 180 > 100) & (rho_[-1, :] % 180 < 130)
    mask_blue = ~mask_red
    plt.plot(rho_[:, mask_blue], color='b', alpha=0.7)
    plt.plot(rho_[:, mask_red], color='r')
    plt.show()
    plt.plot(eta_[:, mask_blue], color='b', alpha=0.7)
    plt.plot(eta_[:, mask_red], color='r')
    plt.show()
    plt.plot(Np_[:, mask_blue], color='b', alpha=0.4)
    plt.plot(Np_[:, mask_red], color='r')
    plt.show()
    plt.plot(delta_[:, mask_blue], color='b', alpha=0.7)
    plt.plot(delta_[:, mask_red], color='r')
    plt.show()
    #plt.plot(delta_)
    #plt.show()
    '''
    if first_loop_of_the_launch:
        plot_results(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x + params['zern_x']*zern_speed2_x, zern_y + params['zern_y']*zern_speed2_y, noisy_psf, background_array_found, noise, dim_simu, d_)
        first_loop_of_the_launch = False
    del(eta_, rho_, delta_, x_, z_, zx_, zy_)

    rho_found=params['rho']%360
    eta_found=params['eta']%180                                              
    
    delta_found=params['delta']*delta_speed
    N_found2 = params['N_photons']*nphotons_speed2
    x_found = params['x']*xy_speed2
    y_found = params['y']*xy_speed2
    z_found = params['z']*z_speed2
    zernx = jnp.reshape(zern_x + params['zern_x']*zern_speed2_x, (3,15))
    zerny = jnp.reshape(zern_y + params['zern_y']*zern_speed2_y, (3,15))
    del(params, loss)
    
    score, deviance = eval_batch(x_found, y_found, z_found, zernx, zerny, rho_found, eta_found, delta_found, N_found2, noisy_psf, background_array_found, noise, dim_simu)
    
    rho_found = np.array(rho_found)+rho_offset
    eta_found = np.array(eta_found)
    x_found = np.array(x_found)
    y_found = np.array(y_found)
    
    # the following manipulation is needed for consistency of the parametrization 
    # of the angles, because they are let free to be optimized without bound
    mask = (rho_found>180)
    eta_found[mask] = (180-eta_found[mask])%180
    rho_found = rho_found%180
    
    correction_dft = 0.9952298
    
    x_ = (x/120).astype(int)*120 + 1000*x_found/correction_dft
    y_ = (y/120).astype(int)*120 + 1000*y_found/correction_dft
    np.savez_compressed(str(save_folder)+'/'+str(int(batch))+'.npz', 
                        frame = frame, x=np.array(x_), 
                        y=np.array(y_), z=np.array(1000*z_found), N_photons=np.array(N_found2), 
                        rho=np.array(rho_found), eta=np.array(eta_found), 
                        delta=np.array(delta_found), score=np.array(score), deviance=np.array(deviance), x_start=np.array(x), 
                        y_start=np.array(y), z_start=np.nan,
                        rho_start=np.nan, delta_start=np.nan, 
                        background_array_found=np.array(background_array_found),
                        zernx_found=np.array(zernx), zerny_found=np.array(zerny),
                        noise_gain=np.array(noise[:, 0]), noise_shift=np.array(noise[:, 1]))

# %%
 