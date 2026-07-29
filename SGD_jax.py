# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:42:21 2026

@author: Amaury
amaury.autric@polytechnique.edu
"""

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
from concurrent.futures import ThreadPoolExecutor
import re
# %% PARAMETERS TO BE DEFINED

total_n_frame = 70000
QE = 0.92
EM = 200
sensitivity = 15.4

look_up_folder = '/mnt/d/Amaury/DATA'
save_folder = Path(filedialog.askdirectory(initialdir=look_up_folder, title="Select the directory containing the tiff files"))
path=(save_folder / "reconstruction")
save_folder=(save_folder / "NPZ")
# %% functions

def extract_frames(frame_0, N_frame):
    error_indices = []
    print('extracting frame '+str(frame_0)+' to '+str(frame_0+N_frame-1))

    def load_single(i):
        path_data = str(path) + '/' +str(frame_0+i) + '.tif'
        return i, extract_raw2(path_data)

    with ThreadPoolExecutor(max_workers=Nframe) as executor:
        results = list(executor.map(lambda i: load_single(i), range(N_frame)))

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

def limit(x, lim, slope, upper=True):
    if upper:
        return jnp.sum(jnp.exp((x-lim)*slope))
    else:
        return jnp.sum(jnp.exp(-1*(x-lim)*slope))
    
def loss_pos(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, d_):
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

def loss_angle_with_M(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_):
    # remove plot argument entirely
    dim_data = 6
    dim_simu = int(dim_simu)
    Mj = compute_M_jax(xp=params['x']*xy_speed2, yp=params['y']*xy_speed2, zp=params['z']*z_speed2, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                   Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, 
                   zernike_coefs_x=jnp.reshape(zernx, (3,15)), zernike_coefs_y=jnp.reshape(zerny, (3,15)),
                   second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)

    h = PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    loss = jnp.sum(jnp.add(h, -(data+sigma**2)*jnp.log(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]+sigma**2)))
    #loss = jnp.sum(jnp.pow(jnp.add(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), 2))
    delta_bound = limit(params['delta'], 180, 100, upper=True) + limit(params['delta'], 1, 100, upper=False)
    #rho_bound = limit(params['rho'], 0, 50, upper=False)
    return (loss + 1000.*(delta_bound)).astype(jnp.float32)

def plot_results(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_):
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
        if N_photons[nb]>6000:
            fig, ax = plt.subplots(3, 2)
            ax[0,0].imshow(data[nb,0,0], cmap='gray')
            ax[0,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(data[nb,1,0], cmap='gray')
            ax[1,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(data[nb,2,0], cmap='gray')
            ax[2,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(data[nb,0,1], cmap='gray')
            ax[0,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(data[nb,1,1], cmap='gray')
            ax[1,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(data[nb,2,1], cmap='gray')
            ax[2,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            plt.suptitle(f'Data - PSF {nb} | rho={float(rho[nb]):.3f} eta={float(eta[nb]):.3f} delta={float(delta[nb]):.2f} N={float(N_photons[nb]):.0f} z={float(z[nb]):.2f} bg={float(background_arr[nb].mean()):.2f}')
            plt.show()
        
            fig, ax = plt.subplots(3, 2)
            ax[0,0].imshow(h[nb,0,0], cmap='gray')
            ax[0,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(h[nb,1,0], cmap='gray')
            ax[1,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(h[nb,2,0], cmap='gray')
            ax[2,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(h[nb,0,1], cmap='gray')
            ax[0,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(h[nb,1,1], cmap='gray')
            ax[1,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(h[nb,2,1], cmap='gray')
            ax[2,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            plt.suptitle(f'Fit fiducial - PSF {nb} | rho={float(rho[nb]):.3f} eta={float(eta[nb]):.3f} delta={float(delta[nb]):.2f} N={float(N_photons[nb]):.0f} z={float(z[nb]):.2f} bg={float(background_arr[nb].mean()):.2f}')
            plt.show()
            
        else:
            maxi = max(np.max(data[nb].flatten()), np.max(h[nb].flatten()))
            mini = min(np.min(data[nb].flatten()), np.min(h[nb].flatten()))
            fig, ax = plt.subplots(3, 2)
            ax[0,0].imshow(data[nb,0,0], vmin=mini, vmax=maxi, cmap='gray')
            ax[0,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(data[nb,1,0], vmin=mini, vmax=maxi, cmap='gray')
            ax[1,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(data[nb,2,0], vmin=mini, vmax=maxi, cmap='gray')
            ax[2,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(data[nb,0,1], vmin=mini, vmax=maxi, cmap='gray')
            ax[0,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(data[nb,1,1], vmin=mini, vmax=maxi, cmap='gray')
            ax[1,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(data[nb,2,1], vmin=mini, vmax=maxi, cmap='gray')
            ax[2,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            plt.suptitle(f'Data - PSF {nb} | rho={float(rho[nb]):.3f} eta={float(eta[nb]):.3f} delta={float(delta[nb]):.2f} N={float(N_photons[nb]):.0f} z={float(z[nb]):.2f} bg={float(background_arr[nb].mean()):.2f}')
            plt.show()
            fig, ax = plt.subplots(3, 2)
            ax[0,0].imshow(background_arr[nb].mean()+h[nb,0,0], vmin=mini, vmax=maxi, cmap='gray')
            ax[0,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(background_arr[nb].mean()+h[nb,1,0], vmin=mini, vmax=maxi, cmap='gray')
            ax[1,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(background_arr[nb].mean()+h[nb,2,0], vmin=mini, vmax=maxi, cmap='gray')
            ax[2,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(background_arr[nb].mean()+h[nb,0,1], vmin=mini, vmax=maxi, cmap='gray')
            ax[0,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(background_arr[nb].mean()+h[nb,1,1], vmin=mini, vmax=maxi, cmap='gray')
            ax[1,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(background_arr[nb].mean()+h[nb,2,1], vmin=mini, vmax=maxi, cmap='gray')
            ax[2,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            plt.suptitle(f'Fit - PSF {nb} | rho={float(rho[nb]):.3f} eta={float(eta[nb]):.3f} delta={float(delta[nb]):.2f} N={float(N_photons[nb]):.0f} z={float(z[nb]):.2f} bg={float(background_arr[nb].mean()):.2f}')
            plt.show()
        

@functools.partial(jax.jit, static_argnames=['dim_simu'])
def score_eval(M_, rho, eta, delta, N_photons, data, background, sigma, dim_simu):
    dim_data = 6
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    score = jnp.sum(jnp.add(h, -(data+sigma**2)*jnp.log(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]+sigma**2)), axis=(1,2,3,4))
    return score

@functools.partial(jax.jit, static_argnames=['dim_simu'])
def eval_batch(x_found, y_found, z_found, zernx, zerny, rho_found, eta_found, delta_found, N_found2, noisy_psf, background, sigma, dim_simu):
    M = compute_M_jax(xp=x_found, yp=y_found, zp=z_found, d=d_, x=xx, y=yy, th1=th1, phi=phi, 
                      Ex0=Ex0, Ex1=Ex1, Ex2=Ex2, Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, 
                      zernike_base=zernike_base, zernike_coefs_x=zernx, zernike_coefs_y=zerny,
                      second_plane=second_plane, polar_projections=polar_projections, 
                      lambd=lambd, f_tube=f_tube)
    return score_eval(M, rho_found, eta_found, delta_found, N_found2, noisy_psf, background, sigma, dim_simu)
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
#%% defining useful variables

lambda_emission = jax.device_put(620) # nm
middle_plane = jax.device_put(1.)
d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])


# %% calibration data

def rot(angle):
    angle=angle*np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

J1 = [[ 0.7997112        ,             -0.15173765 + 1j*  0.3638948 ],[
      -0.14405449 + 1j*  0.34023893  ,   -0.80568093 + 1j*  -0.032433655 ]]
J2 = [[ 0.49279398               ,      -0.696921 + 1j*  0.2615071 ],[
      0.21217652 + 1j*  0.6954243   ,  0.32051226 + 1j*  0.42396948 ]]
rotation = 0
J_dichroic = np.array([rot(-rotation)@J1@rot(rotation), rot(-rotation)@J2@rot(rotation), rot(-rotation)@J1@rot(rotation)])


# %% SGD PARANETERS TO DEFINE
Nphotons_speed1 = jax.device_put(15000)
background_speed = jax.device_put(100)
LR1 = jax.device_put(0.03)
num_epochs_max1 = 80

num_epochs_max2 = 120
LR2 = jax.device_put(1.)
delta_speed = jax.device_put(1.)
nphotons_speed2 = jax.device_put(100)
xy_speed2 = jax.device_put(1/70)
z_speed2=jax.device_put(1/20)

# extraction parameters
Nframe= 20 # nb of frame per batch of extraction
last_frame_processed=10000 #starting point 
NPSF = 100 # nb of PSF per batch

n_photons_filtering = 100

# nb of batch of SGD
batch_nb = 30000

raw = np.zeros((Nframe,6,215,160))
buffer = (np.array([]), np.array([]), np.array([]))
psf_buffer = np.empty((0, 3, 2, 13, 13))  # adjust shape to match your PSF dimensions
#%%   #################### gradient descent ##################

# microscope parameters
d_ = jax.device_put(jnp.array([-float(d[1]) for k in range(NPSF)]))
second_plane = jax.device_put(jnp.array([d[1]-d[0], 0, d[1]-d[2]]))
polar_projections = jax.device_put(jnp.array([0, 45, 0]))

N=jax.device_put(jnp.array(80))
l_pixel=jax.device_put(jnp.array(16))
NA=jax.device_put(jnp.array(1.4))
mag=jax.device_put(jnp.array(100))
lambd=jax.device_put(jnp.array(lambda_emission))
f_tube=jax.device_put(jnp.array(200))
MAG=jax.device_put(jnp.array(200/150))

SAF = True

if SAF:                                             
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k_, f_o, costh2 = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic, SAF=SAF)
else:
    costh2=None
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k_, f_o = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic, SAF=SAF)

u, v, Npadding = padding_jax(r, r_cut, k_, f_o,  N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, 
           f_tube=f_tube, MAG=MAG)

phase_mask = jnp.stack([jnp.ones((N,N)), jnp.ones((N,N)), jnp.ones((N,N))])
zernike_base = generate_zernike_base_jax(r_cut=r_cut, N=N, zernike_order=4)
zernike_coefs_x = jnp.zeros((3,15)).astype(jnp.complex64)
zernike_coefs_y = jnp.zeros((3,15)).astype(jnp.complex64)
 
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
z_exp =  jax.device_put(jnp.array([float(d[1])*0.8 for k in range(NPSF)])).astype(jnp.float32)

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


def load_batch(last_frame_processed, buffer, psf_buffer, NPSF, result):
    bufferx, buffery, bufferindex = buffer
    x, y, index_frame = bufferx, buffery, bufferindex
    single_psf = psf_buffer    
    sigma = np.std(single_psf.flatten())
    background = np.mean(single_psf.flatten())
    while len(x)<NPSF:
        print(len(x))
        #t0 = time.time()
        raw, error_indices = extract_frames(last_frame_processed+1, Nframe)
        #print(f'extract_frames: {time.time()-t0:.2f}s')
        
        #t0 = time.time()
        x_, y_, index_frame_ = extract_positions(last_frame_processed+1, Nframe, error_indices)
        
        #print(f'extract_positions: {time.time()-t0:.2f}s')
        # converting to photon count
        raw = raw*sensitivity/(QE*EM)
        sigma = np.std(raw.flatten())
        background = np.mean(raw.flatten())
        L = raw.shape[2]*120
        W = raw.shape[3]*120
        # removing all the PSF where a parameter is evaluated to nan in Louise pipeline
        nb = len(x_)
        L = raw.shape[2]*120
        W = raw.shape[3]*120
        for k in range(nb-1, -1, -1):  # iterate backwards to safely delete
            if np.isnan(x_[k]) or np.isnan(y_[k]) or \
               (y_[k]<6*120) or (x_[k]<6*120) or (x_[k]>L-6*120) or (y_[k]>W-6*120):
                x_ = np.delete(x_, k, 0)
                y_ = np.delete(y_, k, 0)
                index_frame_ = np.delete(index_frame_, k, 0)
           
        index_frame_ = last_frame_processed+index_frame_+1
        
        # extracting the psf from the files
        #t0 = time.time()
        single_psf_ = extract_raw_xy(raw[0], x_[index_frame_==last_frame_processed+1], y_[index_frame_==last_frame_processed+1])
        #print(f'extract_raw_xy: {time.time()-t0:.2f}s')

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
    
    buffer = x[NPSF:], y[NPSF:], index_frame[NPSF:]
    psf_buffer = single_psf[NPSF:]
    noisy_psf = single_psf[:NPSF]
    x, y = x[:NPSF], y[:NPSF]
    index_frame = index_frame[:NPSF]

    Nstart_by_plane = np.sum(noisy_psf, axis=(2,3,4)) - background*len(noisy_psf[0,0].flatten())
    
    result['buffer'] = buffer        
    result['psf_buffer'] = psf_buffer 
    result['noisy_psf'] = jnp.array(noisy_psf)
    result['x'] = jnp.array(x)
    result['y'] = jnp.array(y)
    result['Nstart'] = jnp.array([5000. for i in range(NPSF)]).astype(jnp.float32)#jnp.array(jnp.sum(Nstart_by_plane, axis=1)).astype(jnp.float32)
    result['background_array'] = jnp.array(background*jnp.ones((NPSF,3,2))).astype(jnp.float32)
    result['sigma'] = jnp.array(sigma)
    result['frame'] = index_frame
    result['last_frame_processed'] = last_frame_processed

# functions for the SGD steps
@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step1(params, opt_state, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, d_):
    loss, grads = jax.value_and_grad(loss_pos)(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, d_)
    updates, opt_state = optimizer1.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step2(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_):
    loss, grads = jax.value_and_grad(loss_angle_with_M)(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_)
    updates, opt_state = optimizer2.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

optimizer1 = optax.adam(learning_rate=LR1)
optimizer2 = optax.adam(learning_rate=LR2)


################# main loop ########################
first_loop_of_the_launch = True
for batch in range(batch_nb):
   
    current = {}
    load_batch(last_frame_processed, buffer, psf_buffer, NPSF, current)
    buffer = current.get('buffer', buffer)
    psf_buffer = current.get('psf_buffer', psf_buffer)
    last_frame_processed = current.get('last_frame_processed', last_frame_processed)

    # build params from current batch
    noisy_psf = current['noisy_psf']  
    sigma = current['sigma']        
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
        params, opt_state, loss = step1(params, opt_state, Nphotons_speed1, background_speed, angle_rd2, angle_rd2, angle_rd1, noisy_psf, second_plane, sigma, dim_simu, d_)
        loss_.append(float(loss))
        z__.append(np.array(params['zp']))
        N__.append(np.array(params['N_photons'] * Nphotons_speed1))
        x__.append(np.array(params['xp']))
        bck.append(np.array(params['background'] * background_speed))
    
    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(loss_)
    ax[0,1].plot(z__)
    ax[0,2].plot(N__)
    ax[1,0].plot(x__)
    ax[1,1].plot(bck)
    plt.show()
    del(ax, loss_, z__, N__, x__, bck)

    x_found = params['xp']
    y_found = params['yp']
    z_found = params['zp']
    N_found = params['N_photons'] * Nphotons_speed1
    background_array_found = params['background'] * background_speed
    del(params, loss)
    

################################ second SGD on orientation #########################################################################################
    
    zern_x = jnp.zeros(3*15)
    zern_y = jnp.zeros(3*15)

    params = {
    'rho': rho_start,
    'eta': eta_start,
    'delta': delta_start/delta_speed,
    'N_photons': N_found/nphotons_speed2,
    'x': x_found/xy_speed2,
    'y': y_found/xy_speed2,
    'z': z_found/z_speed2
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
    
    for i in tqdm(range(num_epochs_max2)):
        params, opt_state, loss = step2(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y, noisy_psf, background_array_found, sigma, dim_simu, d_)
        loss_.append(float(loss))
        rho_.append(np.array(params['rho']))
        eta_.append(np.array(params['eta']))
        delta_.append(np.array(params['delta'] * delta_speed))
        x_.append(np.array(params['x'] * xy_speed2))
        z_.append(np.array(params['z'] * z_speed2))
        Np_.append(np.array(params['N_photons'] * nphotons_speed2))
            
    fig, ax = plt.subplots(4,2)
    ax[0,0].plot(loss_) 
    ax[0,1].plot(eta_)
    rho_ = np.array(rho_)
    eta_ = np.array(eta_)
    Np_ = np.array(Np_)
    ax[1,0].plot(rho_)
    delta_ = np.array(delta_)
    ax[1,1].plot(delta_)
    ax[2,0].plot(x_)
    ax[2,1].plot(z_)
    ax[3,0].plot(Np_)
    ax[3,1].hist((params['rho']%180)[Np_[-1]<10000])
    plt.show()
    mask_red = (rho_[-1, :] % 180 > 100) & (rho_[-1, :] % 180 < 130)
    mask_blue = ~mask_red
    '''
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
        plot_results(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y, noisy_psf, background_array_found, sigma, dim_simu, d_)
        first_loop_of_the_launch = False
    del(fig, ax, eta_, rho_, delta_, x_, z_)

    rho_found=params['rho']%360
    eta_found=params['eta']%180
    
    delta_found=params['delta']*delta_speed
    N_found2 = params['N_photons']*nphotons_speed2
    x_found = params['x']*xy_speed2
    y_found = params['y']*xy_speed2
    z_found = params['z']*z_speed2
    zernx = jnp.reshape(zern_x, (3,15))
    zerny = jnp.reshape(zern_y, (3,15))
    del(params, loss)
    
    score = eval_batch(x_found, y_found, z_found, zernx, zerny, rho_found, eta_found, delta_found, N_found2, noisy_psf, background_array_found, sigma, dim_simu)
    
    rho_found = np.array(rho_found)
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
                        delta=np.array(delta_found), score=np.array(score), x_start=np.array(x), 
                        y_start=np.array(y), z_start=np.nan,
                        rho_start=np.nan, delta_start=np.nan, 
                        background_array_found=np.array(background_array_found))
