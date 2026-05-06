# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:42:21 2026

@author: Amaury
amaury.autric@polytechnique.edu
"""

import sys
import os
import jax
import jax.numpy as jnp
# Add the parent directory to the Python path
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these'))
#os.chdir('/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these')
from simu_PSF_polarMFM_JAX import *
import matplotlib.pyplot as plt
from extract_experimental_psf import *
import gc
import optax
import copy
from tqdm import tqdm
import shutil
import functools
import threading
# %% PARAMETERS TO BE DEFINED

QE = 0.92
EM = 200
sensitivity = 15.4
Nframe= 10
total_n_frame = 15000
path_info = '/mnt/z/DATA/4polar_data_raw/2026_02_02_SLB_1um/SM_tres_haut/Calib_Polar_2026-02-02/images/RAW_DATA/image_Pos0.ome_results_fr1to15000_method=Propagation matrix_box-method=Fixed_box5.csv'
path_data_folder = '/mnt/z/DATA/4polar_data_raw/2026_02_02_SLB_1um/SM_tres_haut/Calib_Polar_2026-02-02/images/RAW_DATA/image_Pos0_reco_concat/'

N_batch = 20000
batch_offset = 0

raw = np.zeros((Nframe,6,214,129))

# %% functions
def extract_frames(frame_0, N_frame):
    error_indices = []
    print('extracting frame '+str(frame_0)+' to '+str(frame_0+N_frame-1))
    if total_n_frame>9999:
        nfill=5
    else:
        nfill=4
    for i in range(N_frame):
        #print(i)
        number = str(frame_0 + i).zfill(nfill)
        path_data = path_data_folder+'image_Pos0_'+number+'.tif'
        raw_ = extract_raw(path_data)
        if raw_ is None:
            error_indices.append(i)
            continue
        else:
            raw[i] = raw_
        del(raw_)
    return raw, error_indices

def extract_positions(frame_0, N_frame, error_indices):
    index_frame = []
    x, y, z, rho, delta = [], [], [], [], []
    ind = 0
    for i in range(N_frame):
        if i not in error_indices:
            x__, y__, z__, rho__, delta__ = position_from_data(data, frame_0+i)
            x = np.concatenate((x, x__))
            y = np.concatenate((y, y__))
            z = np.concatenate((z, z__))
            rho = np.concatenate((rho, rho__))
            delta = np.concatenate((delta, delta__))
            for k in range(len(x__)):
                index_frame.append(ind)
        ind+=1
    index_frame=np.array(index_frame)
    return x, y, z, rho, delta, index_frame

def limit(x, lim, slope, upper=True):
    if upper:
        return jnp.sum(jnp.exp((x-lim)*slope))
    else:
        return jnp.sum(jnp.exp(-1*(x-lim)*slope))
    
def loss_pos(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, d_, plot):
    Mj = compute_M_jax(xp=params['xp'], yp=params['yp'], zp=params['zp'], d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    dim_data = 6
    dim_simu = int(dim_simu)
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*Nphotons_speed1)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]

    loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(h+jnp.reshape(params['background']*background_speed, (h.shape[0],3,2))[:, :, :, None, None], -data), axis=(2,)), 2))

    x_bound = limit(params['xp'], 5*0.12, 100, upper=True) + limit(params['xp'], -5*0.12, 100, upper=False)
    y_bound = limit(params['yp'], 5*0.12, 100, upper=True) + limit(params['yp'], -5*0.12, 100, upper=False)
    z_bound = limit(params['zp'], 5., 100, upper=True) + limit(params['zp'], 0, 100, upper=False)
    return (loss +x_bound+y_bound+z_bound).astype(jnp.float32)

def loss_angle_with_M(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_, plot):
    dim_data = 6
    dim_simu = int(dim_simu)
    Mj = compute_M_jax(xp=params['x']*xy_speed2, yp=params['y']*xy_speed2, zp=params['z']*z_speed2, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=jnp.reshape(zernx, (3,15)), zernike_coefs_y=jnp.reshape(zerny, (3,15))
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    h = PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    
    loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), axis=(2,)), 2))
    delta_bound = limit(params['delta'], 180, 100, upper=True) + limit(params['delta'], 1, 100, upper=False)
    if plot:
        for nb in range(data.shape[0]):
            maxi = max(np.max(data[nb,:,:].flatten()), np.max(h[nb,:,:].flatten()))
            fig, ax = plt.subplots(3,2)
            ax[0,0].imshow(data[nb,0,0], vmin=0., vmax=maxi, cmap='gray')
            ax[0,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(data[nb,1,0], vmin=0., vmax=maxi, cmap='gray')
            ax[1,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(data[nb,2,0], vmin=0., vmax=maxi, cmap='gray')
            ax[2,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(data[nb,0,1], vmin=0., vmax=maxi, cmap='gray')
            ax[0,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(data[nb,1,1], vmin=0., vmax=maxi, cmap='gray')
            ax[1,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(data[nb,2,1], vmin=0., vmax=maxi, cmap='gray')
            ax[2,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            plt.show()
            del(fig, ax)
            fig, ax = plt.subplots(3,2)
            ax[0,0].imshow(h[nb,0,0], vmin=0., vmax=maxi, cmap='gray')
            ax[0,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(h[nb,1,0], vmin=0., vmax=maxi, cmap='gray')
            ax[1,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(h[nb,2,0], vmin=0., vmax=maxi, cmap='gray')
            ax[2,0].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(h[nb,0,1], vmin=0., vmax=maxi, cmap='gray')
            ax[0,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(h[nb,1,1], vmin=0., vmax=maxi, cmap='gray')
            ax[1,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(h[nb,2,1], vmin=0., vmax=maxi, cmap='gray')
            ax[2,1].scatter(x_fine[nb]/0.120+6, y_fine[nb]/0.120+6, s=10, c='r', marker='x')
            plt.show()
            del(fig, ax)
    return (loss + 1000.*(delta_bound)).astype(jnp.float32) 

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

data = pos_from_csv(path_info)

#%% extracting data

jax.clear_caches()
intermediate_folder = '/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these/working_folder_jax_sgd'
#shutil.rmtree(intermediate_folder)       # delete the folder
os.makedirs(intermediate_folder)
index_psf = 0

for batch_number in tqdm(range(N_batch)):
    # extracteing the raw 6-stack tiff files
    raw, error_indices = extract_frames((batch_number+batch_offset)*Nframe+1, Nframe)
    # extracting the position from Louise pipeline
    x, y, z, rho, delta, index_frame = extract_positions((batch_number+batch_offset)*Nframe+1, Nframe, error_indices)
    # converting to photon count
    raw = raw*sensitivity/(QE*EM)
    sigma = jnp.std(raw.flatten())
    background = jnp.mean(raw.flatten())
    L = raw.shape[2]*120
    W = raw.shape[3]*120
    # removing all the PSF where a parameter is evaluated to nan in Louise pipeline
    nb = len(x)
    L = raw.shape[2]*120
    W = raw.shape[3]*120
    for k, ele in enumerate(x):
        if np.isnan(x[nb-1-k]) or np.isnan(y[nb-1-k]) or np.isnan(z[nb-1-k]) or (y[nb-1-k]<6*120) or (x[nb-1-k]<6*120) or (x[nb-1-k]>L-6*120) or (y[nb-1-k]>W-6*120):
            x = np.delete(x,nb-1-k,0)
            y = np.delete(y,nb-1-k,0)
            z = np.delete(z,nb-1-k,0)
            rho = np.delete(rho,nb-1-k,0)
            delta = np.delete(delta,nb-1-k,0)
            index_frame = np.delete(index_frame,nb-1-k,0)
    # extracting the psf from the files
    
    single_psf = extract_raw_xy(raw[0], x[index_frame==0], y[index_frame==0])
    for i in range(1,Nframe):
        single_psf = np.concatenate((single_psf, extract_raw_xy(raw[i], x[index_frame==i], y[index_frame==i])))

    # dimenstion matching to have x in horizontal and y in vertical when considering what appears in a tiff file
    single_psf = single_psf[:,::-1,:,::-1,:]
    x, y = y, -x

    NPSF = len(x)
    print('NPSF = ', NPSF)
    
    for ii in range(NPSF):
        #print(ii)
        np.savez_compressed(intermediate_folder+'/'+str(index_psf+ii)+'.npz',
                            single_psf=single_psf[ii], x=x[ii], y=y[ii], sigma=sigma, background=background, frame=Nframe*(batch_number)+index_frame)
    index_psf+=NPSF

#%% defining useful variables

lambda_emission = jax.device_put(620) # nm
middle_plane = jax.device_put(1.2)
interplane = jax.device_put(0.385)
d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])

# %% calibration data
'''
J_dichroic = npjnparray([[0.7838338      ,               -0.25981125 + 1j*  -0.48329058],[
      -0.4230177 + 1j*  0.27765664  ,   -0.7788276 + 1j*  -0.28660256
]]) # this one is for the abstract
'''
J1 = jnp.array([[ 1.2604369        ,             -0.44922367 + 1j*  0.6776327 ],[
      -0.40610462 + 1j*  0.6554575  ,   -1.2775537 + 1j*  0.034650166 ]])
J2 = jnp.array([[ 0.57820666               ,      -1.2006966 + 1j*  0.67720294 ],[
      1.0710695 + 1j*  0.8422108   ,  0.2654659 + 1j*  0.58898413 ]])
J_dichroic = jnp.array([J1, J2, J1])


# %% SGD PARANETERS TO DEFINE
Nphotons_speed1 = jax.device_put(10000)
background_speed = jax.device_put(40)
LR1 = jax.device_put(0.03)
num_epochs_max1 = 100

num_epochs_max2 = 150
LR2 = jax.device_put(0.8)
delta_speed = jax.device_put(1.)
nphotons_speed2 = jax.device_put(100)
xy_speed2 = jax.device_put(1/50)
z_speed2=jax.device_put(1/30)

threading = False

save_folder = '/mnt/z/DATA/polMFM_experimental_processed/these_4polar_MFM/test_jax'
#%%   #################### gradient descent ##################
intermediate_folder = '/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these/working_folder_jax_sgd'
NPSF = 100

# microscope parameters
d_ = -float(d[1])#-jax.device_put(d[1])
second_plane = jax.device_put(jnp.array([d[1]-d[0], 0, d[1]-d[2]]))
polar_projections = jax.device_put(jnp.array([0, 45, 0]))

N=jax.device_put(jnp.array(80))
l_pixel=jax.device_put(jnp.array(16))
NA=jax.device_put(jnp.array(1.4))
mag=jax.device_put(jnp.array(100))
lambd=jax.device_put(jnp.array(lambda_emission))
f_tube=jax.device_put(jnp.array(200))
MAG=jax.device_put(jnp.array(200/150))

SAF = False

if SAF:
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k_, f_o, costh2 = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic)
else:
    costh2=None
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k_, f_o = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic)

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
z_exp =  jax.device_put(jnp.array([0.7 for k in range(NPSF)])) .astype(jnp.float32)

# gradient descent parameters
rho_start = jnp.array([45. for k in range(NPSF)]).astype(jnp.float32)
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

def load_batch(index_psf, NPSF, result):
    noisy_psf = np.zeros((NPSF, 3, 2, 13, 13))
    x = np.zeros(NPSF)
    y = np.zeros(NPSF)
    for ii in range(NPSF):
        data__ = np.load(intermediate_folder + '/' + str(index_psf + ii) + '.npz')
        noisy_psf[ii] = data__['single_psf']
        x[ii] = data__['x']
        y[ii] = data__['y']
        background = data__['background']
        sigma = data__['sigma']
        #frame = data__['frame']
    
    noisy_psf_jax = jnp.array(noisy_psf)
    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    Nstart_by_plane = np.sum(noisy_psf, axis=(2,3,4)) - background*len(noisy_psf[0,0].flatten())
    Nstart = jnp.array(jnp.sum(Nstart_by_plane, axis=1)).astype(jnp.float32)
    background_array = jnp.array(background*jnp.ones((NPSF,3,2))).astype(jnp.float32)

    result['noisy_psf'] = noisy_psf_jax
    result['x'] = x_jax
    result['y'] = y_jax
    result['Nstart'] = Nstart
    result['background_array'] = background_array
    result['sigma'] = jnp.array(sigma)

# functions for the SGD steps
@functools.partial(jax.jit, static_argnames=['dim_simu', 'd_', 'plot'])#, donate_argnums=(0, 1))
def step1(params, opt_state, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, d_, plot):
    loss, grads = jax.value_and_grad(loss_pos)(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, d_, plot)
    updates, opt_state = optimizer1.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@functools.partial(jax.jit, static_argnames=['dim_simu', 'd_', 'plot'])#, donate_argnums=(0, 1))
def step2(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_, plot):
    loss, grads = jax.value_and_grad(loss_angle_with_M)(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, d_, plot)
    updates, opt_state = optimizer2.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

optimizer1 = optax.adam(learning_rate=LR1)
optimizer2 = optax.adam(learning_rate=LR2)

# --- main loop ---
N_total = 1000000
batch_offset = 0
index_psf = 0
batches = range(0 + batch_offset*NPSF, N_total + batch_offset*NPSF, NPSF)

if threading:
    # preload first batch
    next_result = {}
    next_thread = threading.Thread(target=load_batch, args=(index_psf, NPSF, next_result))
    next_thread.start()
    index_psf += NPSF

for batch_start in batches:
    batch_end = min(batch_start + NPSF, N_total)
    print(f'Loading batch {batch_start} to {batch_end-1}')
    if threading:
        # wait for current batch to be ready
        next_thread.join()
        current = next_result
    
        # start loading next batch in background
        if batch_start + NPSF < N_total + batch_offset*NPSF:
            next_result = {}
            next_thread = threading.Thread(target=load_batch, args=(index_psf, NPSF, next_result))
            next_thread.start()
            index_psf += NPSF
    else:
        current = {}
        load_batch(index_psf, NPSF, current)
        index_psf += NPSF
    # build params from current batch
    noisy_psf = current['noisy_psf']  
    sigma = current['sigma']        
    x = current['x']  
    y = current['y']  
    #frame = current['frame']  
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
        params, opt_state, loss = step1(params, opt_state, Nphotons_speed1, background_speed, angle_rd2, angle_rd2, angle_rd1, noisy_psf, second_plane, sigma, dim_simu, d_, plot=False)
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
        params, opt_state, loss = step2(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y, noisy_psf, background_array_found, sigma, dim_simu, d_, plot=False)
        loss_.append(float(loss))
        eta_.append(np.array(params['eta']))
        rho_.append(np.array(params['rho']))
        delta_.append(np.array(params['delta'] * delta_speed))
        x_.append(np.array(params['x'] * xy_speed2))
        z_.append(np.array(params['z'] * z_speed2))
        Np_.append(np.array(params['N_photons'] * nphotons_speed2))
    fig, ax = plt.subplots(4,2)
    ax[0,0].plot(loss_) 
    ax[0,1].plot(eta_)
    ax[1,0].plot(rho_)
    ax[1,1].plot(delta_)
    ax[2,0].plot(x_)
    ax[2,1].plot(z_)
    ax[3,0].plot(Np_)
    plt.show()
    plt.plot(delta_)
    plt.show()
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
    mask = (rho_found>180)
    eta_found[mask] = (180-eta_found[mask])%180
    rho_found = rho_found%180
    
    x_ = (x/120).astype(int)*120 + 1000*x_found
    y_ = (y/120).astype(int)*120 + 1000*y_found
    np.savez_compressed(save_folder+'/'+str(int(batch_start))+'.npz', 
                        frame = np.nan, x=np.array(x_), 
                        y=np.array(y_), z=np.array(1000*z_found), N_photons=np.array(N_found2), 
                        rho=np.array(rho_found), eta=np.array(eta_found), 
                        delta=np.array(delta_found), score=np.array(score), x_start=np.array(x), 
                        y_start=np.array(y), z_start=np.nan,
                        rho_start=np.nan, delta_start=np.nan, 
                        background_array_found=np.array(background_array_found))
