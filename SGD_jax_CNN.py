# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:42:21 2026

@author: Amaury
amaury.autric@polytechnique.edu

CNN version of SGD_jax.py (Omar, 2026-10-29): the same cells, model, loss (loss2), noise model, aberrations and saved
npz, but each PSF starts from the CNN (cnn_first_guess.py) and is fitted until it converges (stream_fit.py,
Levenberg-Marquardt or Adam), instead of the fixed starts and the SGD1 + SGD2 of SGD_jax.py. A change of the common
parts must be made in both files.
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
import time  # timers of the pools
from cnn_first_guess import cnn_first_guess, TRAIN_MIDDLE_PLANE, TRAIN_Z_RANGE  # CNN first guess, needs first_guess_net_jax.pt next to this file
import stream_fit  # per-PSF fit from the CNN start
# %% PARAMETERS TO BE DEFINED

total_n_frame = 100000
QE = 0.92
EM = 200
# electrons per ADU before the EM register. 15.4 was the value used until October 2026, but the variance vs level
# of the raw frames (photon transfer curve) gives 3.7 to 4.2 e-/ADU on three independent data sets (white lamp at
# EM gain 3, cell background at EM gain 200, actin + fiducial bead at EM gain 250): about 4 times less, so the
# photon numbers were about 4 times too high. The fitted orientations and positions do not depend on this value.
# 4.4 since 2026-10-22: single-electron (clock induced charge) exponential tail of the dark rows of beads acquired at
# EM gain 100-200 (D:/AMAURY/DATA/2026_10_22_test_cam, camera_calibration_summary.txt) gives G/S = 44.7-46.1 ADU per
# electron at EM 200, i.e. 4.43 +- 0.1 e-/ADU if the real EM gain is the set one, and F^2 = 1.8-1.9 (variance slope
# / (G/S), theory 2). Same readout settings (SENSITIVITY_SETTINGS). With 3.9 the photon numbers were ~12 % too low.
#sensitivity = 15.4
#sensitivity = 3.9
sensitivity = 4.4

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
    if len(error_indices) == N_frame:
        raise RuntimeError('none of the frames '+str(frame_0)+' to '+str(frame_0+N_frame-1)+' could be read: '
                           'check dimensions (size of the channels of the reconstruction, here '+str(dimensions)+')')

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

# Micro-Manager metadata of the raw files read by read_camera_metadata: name in the file -> name in the config
CAMERA_METADATA = {'Andor1-Gain': 'em_gain', 'Exposure-ms': 'exposure_ms', 'Andor1-Pre Amp Grain': 'preamp',
                   'Andor1-HS Speed': 'hs_speed', 'Andor1-VS Speed': 'vs_speed', 'Andor1-Output Amplifier': 'output_amplifier',
                   'Andor1-AD Converter': 'ad_converter', 'Andor1-FrameTransfer': 'frame_transfer',
                   'Andor1-Temperature Current': 'ccd_temperature', 'Binning': 'binning', 'Andor1-Crop Enabled': 'crop',
                   'Andor1-Head Model': 'camera_model', 'Andor1-Serial Number': 'camera_serial',
                   'Thorlabs_BSC201_Controller-Filter': 'filter_1', 'Thorlabs_BSC201_Controller2-Filter': 'filter_2',
                   'Nikon_TI2E_Microscope-Objective': 'objective'}
# readout settings at which sensitivity was measured (photon transfer curves, see the parameter cell)
SENSITIVITY_SETTINGS = {'preamp': 'Gain 1', 'hs_speed': '17.000MHz', 'output_amplifier': 'Electron Multiplying'}

def read_camera_metadata(raw_folder):
    # acquisition settings of the raw ome.tif files (Micro-Manager metadata of their first frame). Returns the values of
    # the first file (CAMERA_METADATA names) and prints a warning if a setting differs between the files
    values = []
    for raw_file in sorted(Path(raw_folder).glob('*.ome.tif')):
        with tifffile.TiffFile(raw_file) as tif:
            tag = tif.pages[0].tags.get('MicroManagerMetadata')
            page = tag.value if tag is not None and isinstance(tag.value, dict) else {}
        values.append({name: page[key] for key, name in CAMERA_METADATA.items() if key in page})
    if not values or not values[0]:
        print('No Micro-Manager metadata found in the raw files of '+str(raw_folder))
        return {}
    for name in values[0]:
        differing = sorted(set(str(v.get(name)) for v in values))
        if len(differing) > 1:
            print('WARNING: '+name+' differs between the raw files: '+', '.join(differing)+' (first file used)')
    print('Acquisition settings (metadata of the raw files): '+', '.join(f'{k} = {v}' for k, v in values[0].items()))
    return values[0]

def noise_model(raw, dark_variance, n_bins=10):
    # noise of each of the 6 channels of a chunk of frames (frames, channels, H, W), in the units of the data:
    # var(v) = gain*v + V. v = 0 without light (the reconstruction removes the camera baseline). V = dark_variance
    # (one value per channel): variance without light, read noise + clock induced charge, measured by
    # reconstruction_juliette.py on the dark rows of the raw frames and carried through the interpolation and the
    # normalisation of each channel (dark_variance.txt). gain is the part proportional to the light (EM excess noise,
    # real EM gain, normalisation and interpolation): fitted on the photon transfer curve of the whole channel, the
    # pixels sorted by their median over time in n_bins groups, the variance of each group measured on the differences
    # between consecutive frames (removes the static structure, /2 because a difference has twice the variance) with a
    # MAD (robust to the emitters that blink in a pixel), and gain the slope of the variance vs level with the
    # intercept fixed at V. shift = V/gain: (v + shift)/gain has a variance equal to its mean (shifted Poisson, see
    # pixel_loss). level is the median background
    gain, level = np.zeros(raw.shape[1]), np.zeros(raw.shape[1])
    dark_variance = np.broadcast_to(np.asarray(dark_variance, dtype=float), raw.shape[1])
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
        signal = np.array(levels)
        # least squares slope of variance - V vs level, through 0
        gain[c] = np.sum(signal*(np.array(variances) - dark_variance[c])) / np.sum(signal**2)
    shift = dark_variance/gain
    return gain, shift, level

def channel_gain():
    # relative gain of each (plane, polarisation) channel of the model: the y channel of the planes behind J1
    # (0 and 2) and of the plane behind J2 (1) is multiplied by y_gain_J1 and y_gain_J2 (calibration cell)
    return jnp.array([[1., y_gain_J1], [1., y_gain_J2], [1., y_gain_J1]])

def limit(x, lim, slope, upper=True):
    # exp of the overshoot, continued linearly (same value and slope) beyond an exponent of 30: exp(88.7) overflows
    # float32, and an inf loss gives nan parameters in Adam (seen with a PSF whose delta went to ~0.1 deg with the
    # bound at 1: exponent (1 - 0.06)*100 = 94). Beyond 30 the penalty is >1e13 and still pushes back
    e = (x-lim)*slope if upper else -1*(x-lim)*slope
    cap = 30.
    return jnp.sum(jnp.exp(jnp.minimum(e, cap)) * (1 + jnp.maximum(e - cap, 0.)))
    
def loss_pos(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, noise, dim_simu, d_):
    Mj = compute_M_jax(xp=params['xp'], yp=params['yp'], zp=params['zp'], d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube,
                    plane_shift=plane_shift_of_z(params['zp']))
    dim_data = 6
    dim_simu = int(dim_simu)
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*Nphotons_speed1)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    h = h * channel_gain()[None, :, :, None, None] # x/y gain correction of the channels, see y_gain_J1/J2

    loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(h+jnp.reshape(params['background']*background_speed, (h.shape[0],3,2))[:, :, :, None, None], -data), axis=(2,)), 2))

    x_bound = limit(params['xp'], 5*0.12, 100, upper=True) + limit(params['xp'], -5*0.12, 100, upper=False)
    y_bound = limit(params['yp'], 5*0.12, 100, upper=True) + limit(params['yp'], -5*0.12, 100, upper=False)
    z_bound = limit(params['zp'], 50., 100, upper=True) + limit(params['zp'], 0, 100, upper=False)
    N_bound = limit(params['N_photons'], 0., 10000, upper=False)
    return (loss +x_bound+y_bound+z_bound+N_bound).astype(jnp.float32)

def huber_outer(model, data, gain, shift):
    # half deviance r^2/2 of each pixel (shifted Poisson of the noise model) and the pixels beyond huber_c, see pixel_loss
    data_s = data + shift
    model_s = jnp.maximum(model + shift, 1e-3)
    log_term = jnp.where(data_s > 0, data_s*jnp.log(jnp.maximum(data_s, 1e-3)/model_s), 0.)
    half_deviance = jnp.maximum((model_s - data_s + log_term) / gain, 0.) # r^2/2
    outer = half_deviance > huber_c**2/2
    if huber_positive_only:
        outer = outer & (data > model)
    return half_deviance, outer

def pixel_loss(model, data, gain, shift):
    # loss of each pixel for SGD2 and for the saved score, chosen by loss2, with the noise model var = gain*v + V and
    # shift = V/gain (see noise_model)
    if loss2 == 'lms':
        # least squares normalised by the variance of each pixel: (mu - d)^2 / var. var = gain*d is estimated from the
        # data, so it does not depend on the fitted parameters: the weights are fixed during the descent. Clipped at 1
        # to avoid dividing by ~0
        return jnp.pow(model - data, 2) / jnp.maximum(gain*data, 1.)
    if loss2 == 'poisson':
        # shifted Poisson: (d + shift)/gain has the mean (mu + shift)/gain and the variance (gain*mu + V)/gain^2, equal
        # to it, so it is treated as a Poisson count: likelihood ((mu + shift) - (d + shift) log(mu + shift)) / gain
        # (Poisson noise of the light times the EM excess noise, plus the read noise and the clock induced charge V).
        # Each pixel is weighted by 1/(gain*mu + V). Clipped to keep the log defined
        return (model + shift - (data + shift)*jnp.log(jnp.maximum(model + shift, 1e-3))) / gain
    if loss2 == 'huber':
        # Poisson likelihood made robust with the Huber function of the deviance residual r of each pixel
        # (r^2 = deviance of the pixel). Half deviance = Poisson loss + constant, so for |r| <= huber_c the pixel is
        # treated exactly as with 'poisson'; beyond, the loss grows as huber_c*|r| instead of r^2/2: the pull of the
        # pixel is capped at the one of a pixel at huber_c. With huber_positive_only, only the pixels brighter than
        # the model are capped (light the model does not contain: neighbours, irregular background). Shifted
        # quantities as in 'poisson'
        half_deviance, outer = huber_outer(model, data, gain, shift)
        # sqrt of a value >= c^2 only: no infinite gradient at r = 0. Continuous with its derivative at |r| = c
        return jnp.where(outer, huber_c*jnp.sqrt(2*jnp.maximum(half_deviance, huber_c**2/2)) - huber_c**2/2, half_deviance)
    raise ValueError("loss2 must be 'poisson', 'lms' or 'huber', not "+repr(loss2))

def pixel_weight(model, data, gain, shift):
    # curvature of pixel_loss vs the model, weight of each pixel in the Gauss-Newton matrix of the Levenberg-Marquardt
    # (optim 'lm'). Expected value of the second derivative, so always > 0
    if loss2 == 'lms':
        return 2. / jnp.maximum(gain*data, 1.)
    fisher = 1. / (gain*jnp.maximum(model + shift, 1e-3)) # shifted Poisson: Fisher information of the pixel
    if loss2 == 'poisson':
        return fisher
    if loss2 == 'huber':
        # beyond huber_c the loss grows as huber_c*|r| instead of r^2/2: weight huber_c/|r| of the iteratively
        # reweighted least squares
        half_deviance, outer = huber_outer(model, data, gain, shift)
        return jnp.where(outer, fisher*huber_c/jnp.sqrt(2*jnp.maximum(half_deviance, huber_c**2/2)), fisher)
    raise ValueError("loss2 must be 'poisson', 'lms' or 'huber', not "+repr(loss2))

def fold_angles(rho, eta):
    # (rho, eta) and (rho + 180, 180 - eta) are the same orientation: rho (offset included) is brought back to
    # [0, 180[ and eta is flipped once for each half turn removed. Done after rho_offset, so that a rho moved
    # below 0 or above 180 by the offset is folded with its eta (with mask = rho > 180 before the fold, the rho
    # in [0, -rho_offset[ and [180, 180 - rho_offset[ were moved by 180 without flipping eta). Same result as the
    # fold at the end of the loop of SGD_jax.py
    half_turns = np.floor(rho/180)
    eta = np.where(half_turns % 2 == 1, (180 - eta) % 180, eta % 180)
    return rho - 180*half_turns, eta

def loss_angle_with_M(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, zern_speed2_x, zern_speed2_y, data, background, noise, dim_simu, d_):
    # remove plot argument entirely
    dim_data = 6
    dim_simu = int(dim_simu)
    # zernx/zerny are the starting aberrations, params['zern_x'/'zern_y'] the fitted offset shared by the whole batch
    # (an offset rather than a ratio so that a speed of 0 freezes a coefficient at its starting value)
    # jnp.where: a coefficient of speed 0 gets a gradient of exactly 0. With params*speed it would get nan*0 = nan
    # as soon as one PSF diverges, and these coefficients, shared by the batch, would make all its PSF nan
    zern_offset_x = jnp.where(zern_speed2_x == 0, 0., params['zern_x']*zern_speed2_x)
    zern_offset_y = jnp.where(zern_speed2_y == 0, 0., params['zern_y']*zern_speed2_y)
    Mj = compute_M_jax(xp=params['x']*xy_speed2, yp=params['y']*xy_speed2, zp=params['z']*z_speed2, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                   Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base,
                   zernike_coefs_x=jnp.reshape(zernx + zern_offset_x, (3,15)), zernike_coefs_y=jnp.reshape(zerny + zern_offset_y, (3,15)),
                   second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube,
                   plane_shift=plane_shift_of_z(params['z']*z_speed2))

    h = PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    h = h * channel_gain()[None, :, :, None, None] # x/y gain correction of the channels, see y_gain_J1/J2
    model = h + jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None] # expected value per pixel
    # noise model of each channel, var(v) = gain*v + V, shift = V/gain, see noise_model
    gain = noise[:, 0, :, :, None, None]
    shift = noise[:, 1, :, :, None, None]
    loss = jnp.sum(pixel_loss(model, data, gain, shift)) # loss chosen by loss2
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
                    second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube,
                    plane_shift=plane_shift_of_z(params['z']*z_speed2))
    h = np.array(PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1])
    h = np.array(h * channel_gain()[None, :, :, None, None]) # x/y gain correction of the channels, see y_gain_J1/J2
    data = np.array(data)
    
    rho = np.array(params['rho'])
    eta = np.array(params['eta'])
    delta = np.array(params['delta'] * delta_speed)
    N_photons = np.array(params['N_photons'] * nphotons_speed2)
    z = np.array(params['z'] * z_speed2)
    background_arr = np.array(jnp.reshape(background, (h.shape[0],3,2)))
    for nb in range(data.shape[0]):
        # one figure per PSF: rows are the planes, columns data x, data y, fit x, fit y
        if N_photons[nb]>3*6000.*sensitivity/15.4: # fiducial (6000 per plane with sensitivity 15.4), each image with its own scale, the fit without background
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
    h = h * channel_gain()[None, :, :, None, None] # x/y gain correction of the channels, see y_gain_J1/J2
    # loss of SGD2 for each PSF, with the loss chosen by loss2 and the noise model of its channels
    # (same formulas as loss_angle_with_M, summed over the pixels of each PSF only)
    model = h + jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]
    gain = noise[:, 0, :, :, None, None]
    shift = noise[:, 1, :, :, None, None]
    score = jnp.sum(pixel_loss(model, data, gain, shift), axis=(1,2,3,4))
    # deviance of the shifted Poisson of the same noise model, per pixel: 2 sum[mu' - d' + d' log(d'/mu')]/gain with
    # mu' = mu + shift, d' = d + shift, divided by the number of pixels. Close to 1 when the model describes the data
    # whatever the photon number and the background (unlike the score), larger where the model does not fit. A pixel
    # with d' <= 0 has no log term
    data_s = data + shift
    model_s = jnp.maximum(model + shift, 1e-3)
    log_term = jnp.where(data_s > 0, data_s*jnp.log(jnp.maximum(data_s, 1e-3)/model_s), 0.)
    deviance = 2*jnp.sum((model_s - data_s + log_term) / gain, axis=(1,2,3,4)) / (data[0].size)
    return score, deviance

@functools.partial(jax.jit, static_argnames=['dim_simu'])
def eval_batch(x_found, y_found, z_found, zernx, zerny, rho_found, eta_found, delta_found, N_found2, noisy_psf, background, noise, dim_simu):
    M = compute_M_jax(xp=x_found, yp=y_found, zp=z_found, d=d_, x=xx, y=yy, th1=th1, phi=phi, 
                      Ex0=Ex0, Ex1=Ex1, Ex2=Ex2, Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, 
                      zernike_base=zernike_base, zernike_coefs_x=zernx, zernike_coefs_y=zerny,
                      second_plane=second_plane, polar_projections=polar_projections, 
                      lambd=lambd, f_tube=f_tube, plane_shift=plane_shift_of_z(z_found))
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
            'zern_speed2_x', 'zern_speed2_y', 'zern_start_epoch2', 'fit_zernike', 'loss2', 'huber_c', 'huber_positive_only', 'y_gain_J1', 'y_gain_J2', 'unitary_J', 'registration_z',
            'Nframe', 'last_frame_processed', 'NPSF', 'n_photons_filtering', 'batch_nb', 'dimensions',
            'optim', 'batches_per_pool', 'lm_slots', 'cnn_zernike_rounds', 'cnn_zernike_steps']

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
# acquisition settings from the metadata of the raw files: EM gain used for the conversion to photons (the value of the
# parameter cell is kept if the metadata has none). sensitivity is not changed: the pre-amp label of the metadata
# does not give it (see the parameter cell), a warning is printed if the readout differs from the one it was measured with
camera_metadata = read_camera_metadata(path.parent)
if 'em_gain' in camera_metadata:
    EM = float(camera_metadata['em_gain'])
    print('EM gain from the metadata: '+str(EM))
else:
    print('EM gain not in the metadata, value of the parameter cell kept: '+str(EM))
for name, expected in SENSITIVITY_SETTINGS.items():
    if name in camera_metadata and str(camera_metadata[name]) != expected:
        print('WARNING: '+name+' = '+str(camera_metadata[name])+', sensitivity = '+str(sensitivity)+' was measured with '+expected+': check it')
# variance without light of each channel of the reconstruction (ADU^2, read noise + clock induced charge of the dark
# rows of the raw frames carried through the reconstruction), written by reconstruction_juliette.py
if (path / 'dark_variance.txt').exists():
    dark_variance_adu = np.loadtxt(path / 'dark_variance.txt').reshape(6)
    print('Variance without light of the channels (ADU^2): '+str(np.round(dark_variance_adu, 1)))
else:
    dark_variance_adu = np.zeros(6)
    print('No dark_variance.txt in the reconstruction folder: variance without light set to 0 (var = gain*v)')
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

# True: J1 and J2 replaced by their unitary part (polar decomposition J = U H, U = W Vh from the SVD J = W S Vh).
# The fit of Calibration_polar2 leaves them ~3 % off unitary (diattenuation ~5 %, unpolarised light split
# 1.050/0.950 behind J1 and 0.976/1.024 behind J2), while the reconstruction normalises the 6 channels with white
# light, so that unpolarised light gives the same intensity in x and y. U keeps the retardance (rotation and
# ellipticity of the polarisation) and removes the diattenuation. With it, y_gain_J1 = y_gain_J2 = 1
unitary_J = True
if unitary_J:
    W, _, Vh = np.linalg.svd(J1); J1 = W @ Vh
    W, _, Vh = np.linalg.svd(J2); J2 = W @ Vh

rotation = 5
rotation2 =  5
J_dichroic = np.array([J1@rot(rotation), J2@rot(-rotation2), J1@rot(rotation)])
rho_offset = -6.2
# relative gain of the y polarisation channel in the model, for the planes behind J1 (planes 0 and 2) and behind
# J2 (plane 1); 1 = no correction. The rho bias and the delta vs rho of a SLB bead (2026_02_02, rotation =
# rotation2 = 5) vary with a 180 deg period, which rotation/rotation2 cannot produce (90 deg period) but an x/y
# intensity ratio error can: the linearised fit of these curves gives the y channel of the data ~9 % brighter than
# the model behind J1 and ~11 % dimmer behind J2 (estimate, to be checked on the bead)
# with unitary_J = False, 1/(y/x of J) = 1.104 and 0.953 make unpolarised light even in x and y as after the
# normalisation; 1.09 and 0.89 were fitted on the bias curves of the bead
y_gain_J1 = 1.
y_gain_J2 = 0.93

# %% SGD PARANETERS TO DEFINE
# the settings in photons (speeds of SGD1 and SGD2, starting N in load_batch, fiducial threshold in plot_results)
# were tuned with sensitivity = 15.4 (2000, 100, 50, 3000, 6000) and are multiplied by sensitivity/15.4 (506, 25.3,
# 12.7, 760, 1520 with 3.9; 571, 28.6, 14.3, 857, 1714 with 4.4), so that they follow a change of sensitivity.
# N_photons = photons of the whole molecule (3 planes) since 2026-10-28 (PSF_jax normalised each plane to N before):
# the N settings (speeds of N in SGD1 and SGD2, starting N, fiducial threshold) are 3 x the values above, the
# background speed is unchanged (per pixel).
# With the data ~4 times smaller and the same numbers, SGD1 started with ~2 times too many photons and pushed the
# PSF out of focus to remove them from the patch: z ended ~1.3 um too high (SLB 2026_02_02: 1.91 um instead of
# 0.40 um, 0.40 um again with the scaled values). A resumed run reloads its own sensitivity and speeds (config.txt)
photon_scale = sensitivity/15.4
Nphotons_speed1 = jax.device_put(3*2000.*photon_scale) # x3: N of the whole molecule
background_speed = jax.device_put(100.*photon_scale)
LR1 = jax.device_put(0.05)
num_epochs_max1 = 80

num_epochs_max2 = 120
LR2 = jax.device_put(1.2)
delta_speed = jax.device_put(1.8)
nphotons_speed2 = jax.device_put(3*50.*photon_scale) # x3: N of the whole molecule
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
# Aberrations: fit_zernike False: none; fit_zernike True and all the zern_speed2 at 0: fixed at zern_x/zern_y (in the
# LM); fit_zernike True and zern_speed2 > 0: fitted by rounds (cnn_zernike_rounds: LM fit of each PSF, then
# cnn_zernike_steps Adam steps on the aberrations shared by the pool, as in SGD2), with optim 'lm' only
fit_zernike = False
zern_start_epoch2 = 50 # the aberrations stay at zern_x/zern_y during the first epochs of SGD2, then are fitted
# loss of the fit (and of the Adam steps on the aberrations): 'huber' (default, see below), 'poisson' = Poisson
# likelihood, 'lms' = least squares normalised by the variance of each pixel estimated from the data. All use the
# noise model of noise_model. The saved score (eval_batch) is this loss for each PSF. The cells defining step2 and
# eval_batch must be run again after a change (jit).
# 'lms' is biased at low photon numbers: a pixel that fluctuates low gets a small variance, so a large weight,
# and pulls the model down (about -1 photon per pixel for a background of a few photons, N_photons went to
# -3000 on the SLB data of 2026_02_02 with a background of ~3 photons above the baseline). Fine for backgrounds
# of tens of photons (cells). The Poisson likelihood has no such bias
loss2 = 'huber'
# loss2 = 'huber': Poisson likelihood with the pull of each pixel capped beyond huber_c (in noise units, deviance
# residual). huber_c = 2: pixels within 2 noise units (~98 % of the pixels of a well fitted PSF) are treated as with
# 'poisson'; brighter pixels count no more than a pixel at 2. huber_positive_only: only the pixels brighter than the
# model are capped (contamination always adds light)
huber_c = 2.
huber_positive_only = True

# fit of the CNN version
optim = 'lm'  # 'lm': Levenberg-Marquardt of each PSF (loss2); 'adam': one Adam fit of each PSF with the loss of SGD2 (loss2), its speeds and LR2, aberrations fixed
batches_per_pool = 20  # batches loaded together and given to the CNN at once, the GPU slots are refilled inside a pool
lm_slots = 25  # optim 'lm': PSF fitted together on the GPU (<= NPSF): the Jacobian of 100 PSF needs ~15 GB
cnn_zernike_rounds = 2  # optim 'lm', aberrations fitted only: rounds of cnn_zernike_steps Adam steps on the
cnn_zernike_steps = 30  # aberrations of the pool between the LM fits (1 + cnn_zernike_rounds LM fits)

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

# residual registration of the 3 planes vs the z of the emitter: registration_z.npz written by measure_registration_z
# of reconstruction_juliette.py (bead z-stacks reconstructed with the calibration of the data, offset of each channel
# vs the stage z). The image of each plane of the model is shifted laterally by the offset measured at the z of the
# emitter (see plane_shift_of_z in the gradient descent cell). The 2D warps of the reconstruction cannot correct it: it
# depends on z. With the actin2 calibration the outer planes move by 35-70 nm per um of z, which the fit reads as a
# tilt of the dipole (eta biased by up to ~3 deg). None: no correction, the same position in the 3 planes
registration_z = None

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
#%%   #### OPTIONAL - reloading the parameters of a previous run instead of the cells above ###########
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
    'dark_variance_adu': dark_variance_adu, # read from dark_variance.txt of the reconstruction, not reloaded
    # acquisition settings from the metadata of the raw files, not reloaded
    **{'metadata_'+k: str(v) for k, v in camera_metadata.items()},
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
    'registration_z': str(registration_z), # file of the residual registration of the planes vs z, None: no correction
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
    'unitary_J': unitary_J, # J1 and J2 above are already the unitary ones when True
    'y_gain_J1': y_gain_J1,
    'y_gain_J2': y_gain_J2,
    'fit_zernike': fit_zernike,
    'loss2': loss2,
    'huber_c': huber_c,
    'huber_positive_only': huber_positive_only,
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
    # CNN version
    'use_cnn': True,  # written by SGD_jax_CNN.py (SGD_batch.py resumes such a run with it)
    'optim': optim,
    'batches_per_pool': batches_per_pool,
    'lm_slots': lm_slots,
    'cnn_zernike_rounds': cnn_zernike_rounds,
    'cnn_zernike_steps': cnn_zernike_steps,
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
Nstart_test = jnp.array([3*3000. for k in range(NPSF)]).astype(jnp.float32)

Mtest = compute_M_jax(xp=x_start, yp=y_start, zp=z_exp, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                , zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                ,  second_plane=second_plane
              , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
htest = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=Nstart_test).astype(jnp.float32)
dim_simu = int(htest.shape[-1]//2)

# residual registration of the planes vs z (registration_z, see the parameter cell). The file gives, for the 6
# channels of the reconstruction, the offset relative to channel 1 (image (col, row), nm) vs the stage z relative to
# the focus of the middle plane (dz, nm), and the stage z of the focus of each plane (focus_dz). Conversion to the model:
# - emitter z: linear relation between the stage z of the focus of each plane (beads) and the z of the emitter at which
#   the same plane of the model is in focus (sharpest PSF of an isotropic emitter)
# - the planes are reversed (plane 0 of the model = plane 2 of the reconstruction, as the data patches); channel 0
#   (Mx) of a plane = its canal of the top row of the camera (canals 0-2), channel 1 (My) = bottom row (canals 3-5);
#   (x, y) of the fit = (col, -row) of the image. The two channels of a plane are corrected separately: they move
#   together with z but differ by ~10 nm, and with their mean only the eta bias went from 2.3 to 1.8 deg rms (simulation)
plane_shift_of_z = lambda z: None
if registration_z not in (None, 'None', ''):
    reg = np.load(registration_z)
    z_scan = jnp.linspace(0.05, 1.5, NPSF).astype(jnp.float32)
    M_scan = compute_M_jax(xp=x_start, yp=y_start, zp=z_scan, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                           Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x,
                           zernike_coefs_y=zernike_coefs_y, second_plane=second_plane, polar_projections=polar_projections,
                           lambd=lambd, f_tube=f_tube)
    h_scan = np.array(PSF_jax(rho=jnp.zeros(NPSF), eta=jnp.zeros(NPSF), delta=jnp.full(NPSF, 180.), M=M_scan,
                              N_photons=jnp.full(NPSF, 1e4)))[:, :, :, dim_simu-6:dim_simu+7, dim_simu-6:dim_simu+7]
    sharpness = h_scan.sum(2).max(axis=(-2, -1))/h_scan.sum(axis=(2, 3, 4)) # (z, plane of the model)
    z_focus_model = np.array(z_scan)[np.argmax(sharpness, axis=0)]           # um, planes of the model
    stage_focus = np.asarray(reg['focus_dz'], float)[::-1]                     # nm, planes of the model
    slope, intercept = np.polyfit(stage_focus, z_focus_model, 1)               # z_emitter = intercept + slope*dz
    z_grid = intercept + slope*np.asarray(reg['dz'], float)
    order = np.argsort(z_grid)
    offset = np.asarray(reg['offset'], float)                                  # (6 channels, nz, (col, row)) nm
    canal = [[2 - P + 3*c for c in (0, 1)] for P in range(3)]                     # (plane of the model, channel)
    reg_sx = np.array([[offset[canal[P][c], order, 0] for c in (0, 1)] for P in range(3)])/1000  # (3, 2, nz) um
    reg_sy = np.array([[-offset[canal[P][c], order, 1] for c in (0, 1)] for P in range(3)])/1000
    reg_z = jnp.array(z_grid[order]); reg_sx, reg_sy = jnp.array(reg_sx), jnp.array(reg_sy)
    def plane_shift_of_z(z):
        # lateral offset (um) of the image of each plane and channel at the z (um) of each emitter: (N_psf, 3, 2) x 2
        sx = jnp.stack([jnp.stack([jnp.interp(z, reg_z, reg_sx[k, c]) for c in (0, 1)], -1) for k in range(3)], 1)
        sy = jnp.stack([jnp.stack([jnp.interp(z, reg_z, reg_sy[k, c]) for c in (0, 1)], -1) for k in range(3)], 1)
        return sx, sy
    print('Residual registration of the planes vs z from '+str(registration_z)+': model planes in focus at z = '
          +str(np.round(z_focus_model, 3))+' um, z_emitter = '+f'{intercept:.3f} + {slope*1000:.3f}*dz_stage (um)')
    for P in range(3):
        for c in (0, 1):
            print(f'   plane {P} channel {c}: offset (x, y) nm at z = ' + ', '.join(f'{zz:.2f}: ({1000*np.interp(zz, z_grid[order], reg_sx[P, c]):+.0f}, '
                  f'{1000*np.interp(zz, z_grid[order], reg_sy[P, c]):+.0f})' for zz in (0.3, 0.45, 0.6, 0.75, 0.9)))


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
        gain, shift, level = noise_model(raw, dark_variance_adu*(sensitivity/(QE*EM))**2)
        background = np.mean(level)
        print('noise of frames '+str(last_frame_processed+1)+' to '+str(last_frame_processed+Nframe)+', var = gain*value + V, shift = V/gain')
        print('  gain  '+str(np.round(gain, 2))+'\n  shift '+str(np.round(shift, 2))+'\n  background '+str(np.round(level, 1)))
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
    result['Nstart'] = jnp.array([3*3000.*sensitivity/15.4 for i in range(NPSF)]).astype(jnp.float32) # 3000 per plane with sensitivity 15.4, x3 for the whole molecule. Other start: #jnp.array(jnp.sum(Nstart_by_plane, axis=1)).astype(jnp.float32)
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
buffer = (np.array([]), np.array([]), np.array([]), np.empty((0, 3, 3, 2))) # x, y, frame, noise (gain, shift, level) of each PSF
psf_buffer = np.empty((0, 3, 2, 13, 13))  # adjust shape to match your PSF dimensions
first_loop_of_the_launch = True
print('Starting at batch '+str(first_batch)+' after frame '+str(last_frame_processed)+', saving in '+str(save_folder))
# batches_per_pool batches are loaded together, each PSF starts from the CNN and is fitted until it converges (its GPU
# slot then takes the next PSF); the npz of each batch are saved as by SGD_jax.py, with the steps of each PSF and the
# durations of its pool
correction_dft = 0.9952298
# aberrations, same switches as the loop below: fit_zernike False: none (zern_x/zern_y set to 0 above); True: start
# at zern_x/zern_y, fixed if all the zern_speed2 are 0, else fitted. They are shared by the pool: with optim 'lm'
# they are fitted by rounds (LM fit of each PSF, then cnn_zernike_steps Adam steps on the aberrations with the PSF
# fixed, as the aberrations of SGD2: offset * zern_speed2, LR2, loss2), then a last LM fit
fit_zernike_cnn = bool(fit_zernike) and bool(np.any(np.asarray(zern_speed2_x) != 0) or np.any(np.asarray(zern_speed2_y) != 0))
if fit_zernike_cnn and optim != 'lm':
    print("WARNING optim 'adam': the aberrations are not fitted, they stay at zern_x/zern_y")
print('CNN loop: optimizer '+optim+', loss '+loss2+', aberrations '
      +('none' if not fit_zernike else 'fitted' if fit_zernike_cnn and optim == 'lm' else 'fixed at zern_x/zern_y'))
noise_px = lambda noise: (noise[:, 0, :, :, None, None], noise[:, 1, :, :, None, None]) # gain, shift of each pixel
def zernike_of(offset_x, offset_y):
    # aberrations (3, 15) of the model: start + fitted offset, as in loss_angle_with_M
    return jnp.reshape(zern_x + offset_x*zern_speed2_x, (3, 15)), jnp.reshape(zern_y + offset_y*zern_speed2_y, (3, 15))
if optim == 'lm':
    def model_lm(t7, bg, p):
        # expected image from x, y, z (um), N (whole molecule), rho, eta, delta (deg), the flat background and the
        # aberrations of the pool p, same model as SGD2: aberrations, registration of the planes vs z, x/y gain
        Mj = compute_M_jax(xp=t7[:, 0], yp=t7[:, 1], zp=t7[:, 2], d=d_[:t7.shape[0]], x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                           Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask,
                           zernike_base=zernike_base, zernike_coefs_x=p['zernx'], zernike_coefs_y=p['zerny'],
                           second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube,
                           plane_shift=plane_shift_of_z(t7[:, 2]))
        h = PSF_jax(rho=t7[:, 4], eta=t7[:, 5], delta=t7[:, 6], M=Mj, N_photons=t7[:, 3])[:, :, :, dim_simu-6:dim_simu+7, dim_simu-6:dim_simu+7]
        h = h * channel_gain()[None, :, :, None, None] # x/y gain correction of the channels, see y_gain_J1/J2
        return h + jnp.reshape(bg, (h.shape[0], 3, 2))[:, :, :, None, None]
    fit_pool = stream_fit.make_fit_lm(model_lm, lambda mu, data, noise: pixel_loss(mu, data, *noise_px(noise)),
                                      lambda mu, data, noise: pixel_weight(mu, data, *noise_px(noise)), min(int(lm_slots), NPSF))
    # gradient of loss2 vs the offsets of the aberrations for one batch of NPSF PSF, their parameters fixed
    zernike_grad = jax.jit(jax.grad(lambda off, prm, data, bg, noise: loss_angle_with_M(
        dict(prm, zern_x=off[0], zern_y=off[1]), delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y,
        zern_speed2_x, zern_speed2_y, data, bg, noise, dim_simu, d_)))
else:  # Adam with the loss of SGD2 (loss2), its speeds and LR2, background offsets with a speed of 0.1 photon/pixel
    speeds = dict(delta_speed=delta_speed, nphotons_speed2=nphotons_speed2, xy_speed2=xy_speed2, z_speed2=z_speed2)
    fit_pool = stream_fit.make_fit_adam(
        lambda p, data, bg, noise: loss_angle_with_M(p, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y, 0*zern_speed2_x, 0*zern_speed2_y, data, bg, noise, dim_simu, d_),
        speeds, float(LR2), 0.1, NPSF)

def cnn_z_equivalence(middle_plane_run, middle_plane_cnn, z_run=np.arange(0., 3.001, 0.02), z_cnn=np.arange(-1.5, 4.501, 0.01)):
    # the CNN was trained at another middle plane: for each z of z_run (middle plane of this run), the z' of z_cnn
    # whose PSF at the middle plane of the training matches best (least squares of the normalised 13 x 13 images,
    # 9 orientations, delta 30 deg, no aberration, no registration). Returns (z_run, z'), z' increasing
    if np.isclose(middle_plane_run, middle_plane_cnn):
        return z_run, z_run
    orientations = [(r, e) for r in (0., 60., 120.) for e in (20., 55., 90.)]
    @jax.jit
    def chunk_images(zz, mp):  # 50 depths, compiled once
        n = zz.shape[0]
        Mj = compute_M_jax(xp=jnp.zeros(n), yp=jnp.zeros(n), zp=zz, d=jnp.full(n, -mp), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                           Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask,
                           zernike_base=zernike_base, zernike_coefs_x=jnp.zeros((3, 15)), zernike_coefs_y=jnp.zeros((3, 15)),
                           second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
        h = jnp.stack([PSF_jax(rho=jnp.full(n, r), eta=jnp.full(n, e), delta=jnp.full(n, 30.), M=Mj, N_photons=jnp.ones(n))
                       [:, :, :, dim_simu-6:dim_simu+7, dim_simu-6:dim_simu+7] for r, e in orientations], 1)
        return h / jnp.sum(h, axis=(2, 3, 4, 5), keepdims=True)
    def images(z, mp):
        zz = np.concatenate([z, np.full(-len(z) % 50, z[-1])]).astype(np.float32)  # whole chunks of 50
        out = [np.asarray(chunk_images(jnp.asarray(zz[k:k+50]), jnp.float32(mp)), np.float64).reshape(50, -1) for k in range(0, len(zz), 50)]
        return np.concatenate(out)[:len(z)]
    run, ref = images(z_run, middle_plane_run), images(z_cnn, middle_plane_cnn)
    cost = (run**2).sum(1)[:, None] - 2*run @ ref.T + (ref**2).sum(1)[None]
    j = np.clip(cost.argmin(1), 1, len(z_cnn) - 2); i = np.arange(len(j))
    c0, c1, c2 = cost[i, j-1], cost[i, j], cost[i, j+1]
    z_eq = z_cnn[j] + (z_cnn[1] - z_cnn[0])*0.5*(c0 - c2)/np.maximum(c0 - 2*c1 + c2, 1e-30)  # parabola through the 3 best
    return z_run, np.maximum.accumulate(z_eq)
cnn_z_table = cnn_z_equivalence(float(middle_plane), TRAIN_MIDDLE_PLANE)
print(f'CNN trained at middle_plane {TRAIN_MIDDLE_PLANE}, run at {float(middle_plane):g}: z of the CNN - z = '
      + ', '.join(f'{1000*(np.interp(zz, *cnn_z_table) - zz):+.0f} nm at {zz:g} um' for zz in (0.25, 0.75, 1.25, 1.75))
      + '; start reliable for z in [%.2f, %.2f] um (z of the training %g-%g um)' % (*np.interp(TRAIN_Z_RANGE, cnn_z_table[1], cnn_z_table[0]), *TRAIN_Z_RANGE))

for group in range(first_batch, batch_nb, batches_per_pool):
    group_batches = list(range(group, min(group + batches_per_pool, batch_nb)))
    first_of_launch = first_loop_of_the_launch  # the first pool of a launch includes the JAX compilation
    first_loop_of_the_launch = False
    wall_start = time.time()
    t0 = time.perf_counter()
    loaded = []
    for batch in group_batches:
        current = {}
        load_batch(last_frame_processed, buffer, psf_buffer, NPSF, current, n_skip=n_psf_to_skip)
        n_psf_to_skip = 0
        buffer, psf_buffer, last_frame_processed = current['buffer'], current['psf_buffer'], current['last_frame_processed']
        loaded.append(current)
    t_load = time.perf_counter() - t0  # reading the frames of the whole pool
    n_pool = NPSF * len(loaded)
    cat = lambda key: jnp.concatenate([c[key] for c in loaded])

    # starting point of the fit: the CNN instead of the fixed starts and SGD1 of the loop below (no SGD1: the
    # background is fitted with the other parameters, one offset per plane)
    t0 = time.perf_counter()
    x0, y0, z0, N0, rho0, eta0, delta0 = cnn_first_guess(cat('noisy_psf'), z_table=cnn_z_table)  # um, um, um (middle plane of the run), photons of the whole molecule, deg, deg, deg
    x0.block_until_ready()
    t_cnn = time.perf_counter() - t0

    pool = dict(data=cat('noisy_psf'), noise=cat('noise'), background=cat('background_array').reshape(n_pool, 6),
                x0=x0, y0=y0, z0=z0, N0=N0, rho0=rho0, eta0=eta0, delta0=delta0)
    pool = {key: jnp.asarray(value, jnp.float32) for key, value in pool.items()}
    zern_offset = (jnp.zeros(3*15), jnp.zeros(3*15))
    t0 = time.perf_counter()
    if optim == 'lm':
        zern_optimizer = optax.adam(learning_rate=LR2)
        zern_state = zern_optimizer.init(zern_offset)
        for zernike_round in range(int(cnn_zernike_rounds) + 1 if fit_zernike_cnn else 1):
            if zernike_round > 0:
                # aberrations: Adam steps on the whole pool, the PSF fixed at the last LM fit, which is also the
                # start of the next LM fit
                prm = [dict(rho=jnp.asarray(res['rho'][sl]), eta=jnp.asarray(res['eta'][sl]), delta=jnp.asarray(res['delta'][sl])/delta_speed,
                            N_photons=jnp.asarray(res['N'][sl])/nphotons_speed2, x=jnp.asarray(res['x'][sl])/xy_speed2,
                            y=jnp.asarray(res['y'][sl])/xy_speed2, z=jnp.asarray(res['z'][sl])/z_speed2)
                       for sl in [slice(i*NPSF, (i+1)*NPSF) for i in range(len(loaded))]]
                bgs = [jnp.asarray(res['background'][i*NPSF:(i+1)*NPSF].reshape(-1)) for i in range(len(loaded))]
                for k in range(int(cnn_zernike_steps)):
                    grads = [zernike_grad(zern_offset, prm[i], loaded[i]['noisy_psf'], bgs[i], loaded[i]['noise']) for i in range(len(loaded))]
                    grad = jax.tree_util.tree_map(lambda *g: sum(g), *grads)
                    updates, zern_state = zern_optimizer.update(grad, zern_state)
                    zern_offset = optax.apply_updates(zern_offset, updates)
                pool.update({key+'0': jnp.asarray(res[key], jnp.float32) for key in ('x', 'y', 'z', 'N', 'rho', 'eta', 'delta')})
                pool['background'] = jnp.asarray(res['background'], jnp.float32)
            pool['zernx'], pool['zerny'] = zernike_of(*zern_offset)
            res = fit_pool(pool)
    else:
        res = fit_pool(pool)
    zernx_cnn, zerny_cnn = zernike_of(*zern_offset)
    t_fit = time.perf_counter() - t0  # fit of the whole pool (and of its aberrations)

    t0 = time.perf_counter()
    scores = []
    for i in range(len(loaded)):
        r = {key: value[i*NPSF:(i+1)*NPSF] for key, value in res.items()}
        scores.append(eval_batch(jnp.asarray(r['x']), jnp.asarray(r['y']), jnp.asarray(r['z']), zernx_cnn, zerny_cnn,
                                 jnp.asarray(r['rho'] % 360), jnp.asarray(r['eta'] % 180), jnp.asarray(r['delta']),
                                 jnp.asarray(r['N']), loaded[i]['noisy_psf'], jnp.asarray(r['background'].reshape(-1)),
                                 loaded[i]['noise'], dim_simu))
    jax.block_until_ready(scores)
    t_eval = time.perf_counter() - t0

    for i, batch in enumerate(group_batches):
        current = loaded[i]
        r = {key: value[i*NPSF:(i+1)*NPSF] for key, value in res.items()}
        score, deviance = scores[i]
        # same angles as the loop below: rho % 360, eta % 180, rho_offset, then fold_angles
        rho_found, eta_found = fold_angles(r['rho'] % 360 + rho_offset, r['eta'] % 180)
        x, y = np.array(current['x']), np.array(current['y'])
        x_ = (x/120).astype(int)*120 + 1000*r['x']/correction_dft
        y_ = (y/120).astype(int)*120 + 1000*r['y']/correction_dft
        np.savez_compressed(str(save_folder)+'/'+str(int(batch))+'.npz',
                            frame=current['frame'], x=np.array(x_), y=np.array(y_), z=1000*r['z'], N_photons=r['N'],
                            rho=rho_found, eta=eta_found, delta=r['delta'], score=np.array(score), deviance=np.array(deviance),
                            x_start=x, y_start=y, z_start=np.nan, rho_start=np.nan, delta_start=np.nan,
                            background_array_found=r['background'].reshape(-1),
                            zernx_found=np.array(zernx_cnn), zerny_found=np.array(zerny_cnn),
                            noise_gain=np.array(current['noise'][:, 0]), noise_shift=np.array(current['noise'][:, 1]),
                            steps=r['steps'], use_cnn=True, optim=optim,  # steps (Adam) or iterations (LM) of each PSF
                            group=group, n_pool=n_pool, first_of_launch=first_of_launch, wall_start=wall_start,
                            t_load=t_load, t_cnn=t_cnn, t_fit=t_fit, t_eval=t_eval)  # durations of the whole pool (s)
    print(f'pool {group}-{group_batches[-1]}: {n_pool} PSF, fit {t_fit:.0f} s, load {t_load:.0f} s, steps {res["steps"].mean():.1f}')

# %%
 