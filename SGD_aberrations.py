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
sys.path.append(os.path.abspath('/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these'))
#os.chdir('/mnt/c/Users/Amaury/Documents/Github/4polarMFM_these')
from simu_PSF_polarMFM_JAX import *
import matplotlib.pyplot as plt
from extract_experimental_psf import *
import gc
import optax
import copy
from tqdm import tqdm
import functools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
import tifffile 
from scipy import ndimage
from tkinter import filedialog
from aberrations_beads import *
from matplotlib.widgets import LassoSelector

# %% PARAMETERS TO BE DEFINED

total_n_frame = 50000
QE = 0.92
EM = 10
sensitivity = 15.4


#%%
def limit(x, lim, slope, upper=True):
    if upper:
        return jnp.sum(jnp.exp((x-lim)*slope))
    else:
        return jnp.sum(jnp.exp(-1*(x-lim)*slope))

def loss_pos(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, dim_simu):
    Mj = compute_M_jax(xp=params['xp']*jnp.array([1 for i in range(101)]), yp=params['yp']*jnp.array([1 for i in range(101)]), zp=params['zp']*jnp.array([1 for i in range(101)])*0+0.5, d=params['d']-jnp.linspace(-1.5,1.5,101), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    dim_data = 10
    dim_simu = int(dim_simu)
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*jnp.array([1 for i in range(101)])*Nphotons_speed1)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]

    loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(h+jnp.reshape(params['background']*background_speed, (h.shape[0],3,2))[:, :, :, None, None], -data), axis=(2,)), 2))

    x_bound = limit(params['xp'], 5*0.12, 100, upper=True) + limit(params['xp'], -5*0.12, 100, upper=False)
    y_bound = limit(params['yp'], 5*0.12, 100, upper=True) + limit(params['yp'], -5*0.12, 100, upper=False)
    z_bound = limit(params['zp'], 2., 100, upper=True) + limit(params['zp'], 0, 100, upper=False)
    N_bound = limit(params['N_photons'], 0., 100, upper=False)
    return (loss +x_bound+y_bound+z_bound+N_bound).astype(jnp.float32)

def loss_angle_with_M(params, rho, eta, delta, nphotons_speed2, xy_speed2, z_speed2, data, background, dim_simu):
    # remove plot argument entirely
    dim_data = 10
    dim_simu = int(dim_simu)
    Mj = compute_M_jax(xp=params['x']*jnp.array([1 for i in range(101)])*xy_speed2, yp=params['y']*jnp.array([1 for i in range(101)])*xy_speed2, zp=params['z']*jnp.array([1 for i in range(101)])*z_speed2*0+0.5, d=params['d']-jnp.linspace(-1.5,1.5,101), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                   Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, 
                   zernike_coefs_x=jnp.reshape(params['zern_x'], (3,15)), zernike_coefs_y=jnp.reshape(params['zern_y'], (3,15)),
                   second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)

    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*jnp.array([1 for i in range(101)])*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    loss = jnp.sum(jnp.add(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -(data)*jnp.log(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None])))
    #loss = jnp.sum(jnp.pow(jnp.add(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), 2))
    #rho_bound = limit(params['rho'], 0, 50, upper=False)
    return loss.astype(jnp.float32)


#%% extracting positions/pre-loc

centers, parent = predetection_zstack_beads()

x, y = centers[:,0], centers[:,1]
x, y = y*120, -x*120
#%%
data = []
mean=[]
maxi=[]
mini=[]
for file in sorted(parent.iterdir(), key=lambda f: int(f.stem)):
    if False:
        im = extract_reconstructed(file)
        mean.append(np.mean(im))
        maxi.append(np.max(im))
        mini.append(np.min(im))
        fig, ax = plt.subplots()
        ax= plt.imshow(im[3],vmin=0,vmax=1000)
        cb = plt.colorbar(ax)
        plt.show()
    data.append(extract_reconstructed(file))
data = np.array(data).reshape(101,3,2,216,160)

#%%
psf3D = np.array([data[:,:,:, int(c[0])-10:int(c[0])+11,int(c[1])-10:int(c[1])+11] for c in centers])
psf3D = psf3D[:,:,::-1,:,::-1,:]*sensitivity/(QE*EM)

#%% defining useful variables

lambda_emission = jax.device_put(620) # nm
middle_plane = jax.device_put(1.)
interplane = jax.device_put(0.360)
d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])


# %% calibration data

def rot(angle):
    angle=angle*np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

J1 = [[ 0.7997112        ,             -0.15173765 + 1j*  0.3638948 ],[
      -0.14405449 + 1j*  0.34023893  ,   -0.80568093 + 1j*  -0.032433655 ]]
J2 = [[ 0.49279398               ,      -0.696921 + 1j*  0.2615071 ],[
      0.21217652 + 1j*  0.6954243   ,  0.32051226 + 1j*  0.42396948 ]]
rotation = 14
J_dichroic = np.array([rot(-rotation)@J1@rot(rotation), rot(-rotation)@J2@rot(rotation), rot(-rotation)@J1@rot(rotation)])


# %% SGD PARANETERS TO DEFINE
Nphotons_speed1 = jax.device_put(300000)
background_speed = jax.device_put(300)
LR1 = jax.device_put(0.01)
num_epochs_max1 = 150

num_epochs_max2 = 120
LR2 = jax.device_put(.02)
nphotons_speed2 = jax.device_put(10000)
xy_speed2 = jax.device_put(1/70)
z_speed2=jax.device_put(1/20)

look_up_folder = '/mnt/c/Users/Amaury/Desktop/DATA/'#'/mnt/z/DATA/4_polar_MFM_these/'
save_folder = filedialog.askdirectory(
    initialdir=look_up_folder)

# extraction parameters
NPSF = 101 # nb of PSF per batch

#%%   #################### gradient descent ##################

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
x_start = jax.device_put(0).astype(jnp.float32)
y_start = jax.device_put(0).astype(jnp.float32)
z_exp =  jax.device_put(0.5).astype(jnp.float32)

# gradient descent parameters
rho_start = jnp.array([45. for k in range(NPSF)]).astype(jnp.float32)
eta_start = jnp.array([90. for k in range(NPSF)]).astype(jnp.float32)
delta_start = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
Nstart_test = jnp.array(10000).astype(jnp.float32)
bck_test = jnp.array(500*jnp.ones((NPSF,3,2))).astype(jnp.float32)

Mtest = compute_M_jax(xp=x_start*jnp.array([1 for i in range(101)]), yp=y_start*jnp.array([1 for i in range(101)]), zp=z_exp*jnp.array([1 for i in range(101)]), d=d_-jnp.linspace(-1.5,1.5,101), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                , zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                ,  second_plane=second_plane
              , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
htest = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=Nstart_test*jnp.array([1 for i in range(101)])).astype(jnp.float32)

dim_simu = int(htest.shape[-1]//2)

# functions for the SGD steps
@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step1(params, opt_state, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, dim_simu):
    loss, grads = jax.value_and_grad(loss_pos)(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, dim_simu)
    updates, opt_state = optimizer1.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step2(params, opt_state, rho_start, eta_start, delta_start, nphotons_speed2, xy_speed2, z_speed2, data, background, dim_simu):
    loss, grads = jax.value_and_grad(loss_angle_with_M)(params, rho_start, eta_start, delta_start, nphotons_speed2, xy_speed2, z_speed2, data, background, dim_simu)
    updates, opt_state = optimizer2.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

optimizer1 = optax.adam(learning_rate=LR1)
optimizer2 = optax.adam(learning_rate=LR2)


################# main loop ########################
for bead in range(psf3D.shape[0]):

    angle_rd1 = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
    angle_rd2 = jnp.array([45. for k in range(NPSF)]).astype(jnp.float32)
    optimizer = optax.adam(learning_rate=LR1)
    params = {
        'xp': x_start,
        'yp': y_start,
        'zp': z_exp,
        'd': d_,
        'N_photons': Nstart_test / Nphotons_speed1,
        'background': bck_test.flatten() / background_speed
    }
    opt_state = optimizer.init(params)
    
    loss_ = []
    z__ = []
    N__ = []
    x__ =[]
    bck = []
    d__ = []
    
    for i in tqdm(range(num_epochs_max1)):
        params, opt_state, loss = step1(params, opt_state, Nphotons_speed1, background_speed, angle_rd2, angle_rd2, angle_rd1, psf3D[bead], second_plane, dim_simu)
        loss_.append(float(loss))
        z__.append(np.array(params['zp']))
        N__.append(np.array(params['N_photons'] * Nphotons_speed1))
        x__.append(np.array(params['xp']))
        bck.append(np.array(params['background'] * background_speed))
        d__.append(np.array(params['d']))
    
    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(loss_)
    ax[0,1].plot(z__)
    ax[0,2].plot(N__)
    ax[1,0].plot(x__)
    ax[1,1].plot(bck)
    ax[1,2].plot(d__)
    plt.show()
    del(ax, loss_, z__, N__, x__, bck)

    x_found = params['xp']
    y_found = params['yp']
    z_found = params['zp']
    N_found = params['N_photons'] * Nphotons_speed1
    background_array_found = params['background'] * background_speed
    d_found = params['d']
    del(params, loss)
    

################################ second SGD on orientation #########################################################################################
    
    zern_x = jnp.zeros(3*15)
    zern_y = jnp.zeros(3*15)

    params = {
    'zern_x': zern_x,
    'zern_y': zern_y,
    'N_photons': N_found/nphotons_speed2,
    'x': x_found/xy_speed2,
    'y': y_found/xy_speed2,
    'z': z_found/z_speed2,
    'd': d_found
    }
    optimizer = optax.adam(learning_rate=LR2)
    opt_state = optimizer.init(params)
    
    loss_ = []
    zern_x_ = []
    zern_y_ = []
    d__ = []
    x_ = []
    z_ = []
    Np_ = []
    
    for i in tqdm(range(num_epochs_max2)):
        params, opt_state, loss = step2(params, opt_state, rho_start, eta_start, delta_start, nphotons_speed2, xy_speed2, z_speed2, psf3D[bead], background_array_found, dim_simu)
        loss_.append(float(loss))
        zern_x_.append(np.array(params['zern_x'])*1000/(2*np.pi))
        zern_y_.append(np.array(params['zern_y'])*1000/(2*np.pi) )
        d__.append(np.array(params['d'] * delta_speed))
        x_.append(np.array(params['x'] * xy_speed2))
        z_.append(np.array(params['z'] * z_speed2))
        Np_.append(np.array(params['N_photons'] * nphotons_speed2))
            
    fig, ax = plt.subplots(4,2)
    ax[0,0].plot(loss_) 
    ax[0,1].plot(zern_x_)
    ax[1,0].plot(zern_y_)
    ax[1,1].plot(d__)
    ax[2,0].plot(x_)
    ax[2,1].plot(z_)
    ax[3,0].plot(Np_)
    plt.show()

    N_found2 = params['N_photons']*nphotons_speed2
    x_found = params['x']*xy_speed2
    y_found = params['y']*xy_speed2
    z_found = params['z']*z_speed2
    zernx_found = jnp.reshape(params['zern_x'], (3,15))
    zerny_found = jnp.reshape(params['zern_y'], (3,15))
    del(params, loss)
    
    score = eval_batch(x_found, y_found, z_found, zernx, zerny, rho_found, eta_found, delta_found, N_found2, noisy_psf, background_array_found, sigma, dim_simu)
    
    x_found = np.array(x_found)
    y_found = np.array(y_found)
    # the following manipulation is needed for consistency of the parametrization 
    # of the angles, because they are let free to be optimized without bound

    '''
    np.savez_compressed(save_folder+'/'+str(int(batch))+'.npz', 
                        frame = frame, x=np.array(x_), 
                        y=np.array(y_), z=np.array(1000*z_found), N_photons=np.array(N_found2), 
                        rho=np.array(rho_found), eta=np.array(eta_found), 
                        delta=np.array(delta_found), score=np.array(score), x_start=np.array(x), 
                        y_start=np.array(y), z_start=np.nan,
                        rho_start=np.nan, delta_start=np.nan, 
                        background_array_found=np.array(background_array_found))'''
