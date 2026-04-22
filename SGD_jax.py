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
sys.path.append(os.path.abspath('..'))
from simu_PSF_polarMFM_JAX import *

device = torch.device('cuda')

# %% PARAMETERS TO BE DEFINED
lambda_emission = 638 # nm
middle_plane = 1.2
interplane = 0.385
QE = 0.92
EM = 200
sensitivity = 15.4
Nframe=100
total_n_frame = 7999
path_info = '\\\\NAS_LOCCO\\Amaury\\DATA\\4polar_data_raw\\2026_02_02_SLB_1um\\SM_tres_haut\\Calib_Polar_2026-02-02\\images\\RAW_DATA\\image_Pos0.ome_results_fr1to15000_method=Propagation matrix_box-method=Fixed_box5.csv'
path_data_folder = '\\\\NAS_LOCCO\\Amaury\\DATA\\4polar_data_raw\\2026_02_02_SLB_1um\\SM_tres_haut\\Calib_Polar_2026-02-02\\images\\RAW_DATA\\image_Pos0_reco_concat\\'

N_batch = 20000
batch_offset = 0

#%%
raw = jnp.zeros((Nframe,6,214,129))

d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])

# %% calibration data
'''mode = 'polar projections'
J_dichroic = np.array([[0.7838338      ,               -0.25981125 + 1j*  -0.48329058],[
      -0.4230177 + 1j*  0.27765664  ,   -0.7788276 + 1j*  -0.28660256
]]) # this one is for the abstract
'''
mode = 'Stokes'
J1 = np.array([[ 1.2604369        ,             -0.44922367 + 1j*  0.6776327 ],[
      -0.40610462 + 1j*  0.6554575  ,   -1.2775537 + 1j*  0.034650166 ]])
J2 = np.array([[ 0.57820666               ,      -1.2006966 + 1j*  0.67720294 ],[
      1.0710695 + 1j*  0.8422108   ,  0.2654659 + 1j*  0.58898413 ]])
J_dichroic = np.array([J1, J2, J1])

# %% functions
def extract_frames(frame_0, N_frame):
    error_indices = []
    print('extracting frame '+str(frame_0)+' to '+str(frame_0+N_frame-1))
    if total_n_frame>9999:
        nfill=5
    else:
        nfill=4
    for i in range(N_frame):
        number = str(frame_0 + i).zfill(nfill)
        #path_data = '/mnt/e/2026_01_19_SLB_1um/sample2/SLB1_10_NR40/SM/Calib_Polar_2026-01-19/images/RAW_DATA/image_Pos0_reco/image_Pos0_'+number+'.tif'
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
            x = jnp.concatenate((x, x__))
            y = jnp.concatenate((y, y__))
            z = jnp.concatenate((z, z__))
            rho = jnp.concatenate((rho, rho__))
            delta = jnp.concatenate((delta, delta__))
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
    
def loss_pos(xp, yp, zp, rho, eta, delta, N_photons, data, second_plane, background, sigma, dim_simu, plot):
    Mj = compute_M_jax(xp=xp, yp=yp, zp=zp, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, device=device, mode=mode)
    dim_data = 6
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]

    loss = torch.sum(torch.pow(torch.sum(torch.add(h+torch.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), dim=(2,)), 2))

    x_bound = limit(xp, 5*0.12, 100, upper=True) + limit(xp, -5*0.12, 100, upper=False)
    y_bound = limit(yp, 5*0.12, 100, upper=True) + limit(yp, -5*0.12, 100, upper=False)
    z_bound = limit(zp, 5., 100, upper=True) + limit(zp, 0, 100, upper=False)
    return (loss +x_bound+y_bound+z_bound).to(torch.float32)

def loss_angle_with_M(rho, eta, delta, N_photons, x_fine, y_fine, z_fine, zernx, zerny, data, background, sigma, dim_simu, plot):
    dim_data = 6
    Mj = compute_M_jax(xp=x_fine, yp=y_fine, zp=z_fine, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=jnp.reshape(zernx, (3,15)), zernike_coefs_y=jnp.reshape(zerny, (3,15))
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, device=device, mode=mode)
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    
    #loss = torch.sum(torch.add(h, -(data+sigma**2)*torch.log(h+torch.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]+sigma**2)))
    loss = jnp.sum(torch.pow(jnp.sum(jnp.add(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), dim=(2,)), 2))
    delta_bound = limit(delta, 180, 100, upper=True) + limit(delta, 1, 100, upper=False)
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
    return (loss + 1000.*(delta_bound)).to(torch.float32) #+ N_bound #+ 100000*torch.sum(h**2)

def score_eval(M_, rho, eta, delta, N_photons, data, background, sigma, dim_simu):
    dim_data = 6
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    score = jnp.sum(jnp.add(h, -(data+sigma**2)*jnp.log(h+background+sigma**2)), dim=(1,2,3,4))
    return score