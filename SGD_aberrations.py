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
from matplotlib.colors import LogNorm, Normalize

n2 = 1.47 # beads embedded in glycerol

# %% PARAMETERS TO BE DEFINED

total_n_frame = 50000
QE = 0.92
EM = 25#10
sensitivity = 15.4
nstack = 101
signe_stack = -1
z_scale = 30/120

SAF=True
#%%
def limit(x, lim, slope, upper=True):
    if upper:
        return jnp.sum(jnp.exp((x-lim)*slope))
    else:
        return jnp.sum(jnp.exp(-1*(x-lim)*slope))
    
def loss_pos(params, Nphotons_speed1, z_speed, rad_speed, d_speed, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu):
    #jax.debug.print("second_plane: {}", second_plane)
    Mj = compute_M_jax(xp=params['xp']*jnp.array([1 for i in range(nstack)])
                       , yp=params['yp']*jnp.array([1 for i in range(nstack)])
                       , zp=params['zp']*z_speed*jnp.array([1 for i in range(nstack)])*0+(params['bead_radius']*rad_speed/1000)
                       , d=(params['d']*d_speed+signe_stack*jnp.linspace(-0.03*(nstack//2),0.03*(nstack//2),nstack)), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, SAF=SAF, cos_th2=cos_th2)
    #jax.debug.print("d_: {}", (params['d']*d_speed+signe_stack*jnp.linspace(-0.03*(nstack//2),0.03*(nstack//2),nstack)))
    dim_data = 10
    dim_simu = int(dim_simu)
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*jnp.array([1 for i in range(nstack)])*Nphotons_speed1)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    R = params['bead_radius'] * rad_speed / 120  # R in xy-pixel units
    H, W = h.shape[-2], h.shape[-1]
    
    yg = np.arange(H) - H // 2
    xg = np.arange(W) - W // 2
    yyy, xxx = np.meshgrid(yg, xg, indexing='ij')  # shape (H, W)
    
    # 2D disk kernel in xy only
    rr2 = jnp.asarray(xxx**2 + yyy**2)
    r = jnp.sqrt(rr2 + 1e-4)
    
    edge_width = 0.5
    sphere2d = jax.nn.sigmoid((R - r) / edge_width)  # shape (H, W)
    
    denom = jnp.clip(sphere2d.sum(), 1e-8, None)
    sphere2d = sphere2d / denom
    
    # broadcast to h shape (Z, dim1, dim2, H, W)
    kern = sphere2d[None, None, None, :, :]
    kern = jnp.broadcast_to(kern, h.shape)
    
    kern = jnp.fft.ifftshift(kern, axes=(3, 4))
    
    psf_conv = jnp.real(jnp.fft.ifftn(
        jnp.fft.fftn(h, axes=(3, 4)) * jnp.fft.fftn(kern, axes=(3, 4)),
        axes=(3, 4)
    ))
    #jax.debug.print("sigma: {}", sigma)
    res = psf_conv+jnp.reshape(params['background']*background_speed, (psf_conv.shape[0],3,2))[:, :, :, None, None]
    loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(res, -data), axis=(2,)), 2) ) #  /(jnp.sum(res, axis=(2,))+sigma**2)  + jnp.log(jnp.sum(res, axis=(2,))+sigma**2)
    
    renorm = jnp.sum(jnp.array([(jnp.max(jnp.sum(res[:,ii], axis=-3), axis=(1,2))-jnp.max(jnp.sum(data[:,ii], axis=-3), axis=(1,2)))**2 for ii in range(2)]))
    #jax.debug.print("dim: {}", jnp.sum(res[:,0], axis=-3).shape)
    x_bound = limit(params['xp'], 5*0.12, 100, upper=True) + limit(params['xp'], -5*0.12, 100, upper=False)
    y_bound = limit(params['yp'], 5*0.12, 100, upper=True) + limit(params['yp'], -5*0.12, 100, upper=False)
    z_bound = limit(params['zp'], 2., 100, upper=True) + limit(params['zp'], 0, 100, upper=False)
    N_bound = limit(params['N_photons'], 0., 100, upper=False)
    #jax.debug.print("loss: {}", loss)
    #jax.debug.print("renorm: {}", renorm)
    return (loss +renorm/100+x_bound+y_bound+z_bound+N_bound).astype(jnp.float32)
N=jax.device_put(jnp.array(80))
phase_mask = jnp.stack([jnp.ones((N,N)), jnp.ones((N,N)), jnp.ones((N,N))])
# phase_maskx=jnp.exp(1j*params['phase_maskx']), phase_masky=jnp.exp(1j*params['phase_masky'])
# phase_maskx=phase_mask, phase_masky=phase_mask,

def loss_angle_with_M(params, rho, eta, delta, nphotons_speed2, rad_speed2, d_speed2, xy_speed2, z_speed2, data, background, sigma, dim_simu):
    # remove plot argument entirely
    dim_data = 10
    dim_simu = int(dim_simu)
    Mj = compute_M_jax(xp=params['x']*jnp.array([1 for i in range(nstack)])*xy_speed2, yp=params['y']*jnp.array([1 for i in range(nstack)])*xy_speed2, zp=params['z']*jnp.array([1 for i in range(nstack)])*z_speed2*0+(params['bead_radius']*rad_speed2/1000), d=(params['d']*d_speed2+signe_stack*jnp.linspace(-0.03*(nstack//2),0.03*(nstack//2),nstack)), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                   Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, 
                   zernike_coefs_x=jnp.reshape(params['zern_x'], (3,15)), zernike_coefs_y=jnp.reshape(params['zern_y'], (3,15)),
                   second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, SAF=SAF, cos_th2=cos_th2)

    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*jnp.array([1 for i in range(nstack)])*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    R = params['bead_radius'] * rad_speed2 / 120  # R in xy-pixel units
    H, W = h.shape[-2], h.shape[-1]
    
    yg = np.arange(H) - H // 2
    xg = np.arange(W) - W // 2
    yyy, xxx = np.meshgrid(yg, xg, indexing='ij')  # shape (H, W)
    
    # 2D disk kernel in xy only
    rr2 = jnp.asarray(xxx**2 + yyy**2)
    r = jnp.sqrt(rr2 + 1e-4)
    
    edge_width = 0.5
    sphere2d = jax.nn.sigmoid((R - r) / edge_width)  # shape (H, W)
    
    denom = jnp.clip(sphere2d.sum(), 1e-8, None)
    sphere2d = sphere2d / denom
    
    # broadcast to h shape (Z, dim1, dim2, H, W)
    kern = sphere2d[None, None, None, :, :]
    kern = jnp.broadcast_to(kern, h.shape)
    
    kern = jnp.fft.ifftshift(kern, axes=(3, 4))
    
    psf_conv = jnp.real(jnp.fft.ifftn(
        jnp.fft.fftn(h, axes=(3, 4)) * jnp.fft.fftn(kern, axes=(3, 4)),
        axes=(3, 4)
    ))
    res = psf_conv+jnp.reshape(background, (psf_conv.shape[0],3,2))[:, :, :, None, None]
    #jax.debug.print("sigma: {}", sigma)
    renorm = jnp.sum(jnp.array([(jnp.max(jnp.sum(res[:,ii], axis=-3), axis=(1,2))-jnp.max(jnp.sum(data[:,ii], axis=-3), axis=(1,2)))**2 for ii in range(2)]))
    #loss = jnp.sum(jnp.pow(jnp.sum(jnp.add(psf_conv+jnp.reshape(background, (psf_conv.shape[0],3,2))[:, :, :, None, None], -data), axis=(2,)), 2))
    #loss = jnp.sum(jnp.pow(jnp.add(res, -data), 2))
    loss = jnp.sum(jnp.add(h, -(data+sigma**2)*jnp.log(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]+sigma**2)))
    #loss = jnp.sum(jnp.pow(jnp.add(h+jnp.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), 2))
    #rho_bound = limit(params['rho'], 0, 50, upper=False)
    return (loss+renorm/1000).astype(jnp.float32) 


#%% extracting positions/pre-loc
data_shape=[215,160]
centers, parent = predetection_zstack_beads(data_shape=data_shape)

x, y = centers[:,0], centers[:,1]
x, y = y*120, -x*120
#%% extracting data
data = []
mean=[]
maxi=[]
mini=[]
tif_files = [f for f in parent.iterdir() if f.suffix == '.tif']
for file in sorted(tif_files, key=lambda f: int(f.stem)):
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
data = np.array(data)
data = data.reshape(data.shape[0],3,2,data_shape[0],data_shape[1])
if data.shape[0] == 1010:
    # (10 repeats, 101 frames, 3, 2, 216, 160)
    data = data.reshape(10, 101, 3, 2, data_shape[0],data_shape[1])
    # Average over the 10 repeats
    data = data.mean(axis=0)
#%% reformating
psf3D = np.array([data[:,:,:, int(c[0])-10:int(c[0])+11,int(c[1])-10:int(c[1])+11] for c in centers])
psf3D = psf3D[:,::,::-1,:,::-1,:]*sensitivity/(QE*EM) #converting to zstack towrds up
psf3D = psf3D[:,:]

#%% defining useful variables

lambda_emission = jax.device_put(620) # nm
middle_plane = jax.device_put(0.12)
interplane = jax.device_put(0.33)
d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])


# %% calibration data

def rot(angle):
    angle=angle*np.pi/180
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

J_dichroic = np.array([np.eye(2), np.eye(2), np.eye(2)])


# %% SGD PARANETERS TO DEFINE
Nphotons_speed1 = jax.device_put(200000)
background_speed = jax.device_put(300)
d_speed = jax.device_put(3)
z_speed=jax.device_put(1)
rad_speed = jax.device_put(200)
LR1 = jax.device_put(0.006)
num_epochs_max1 = 600

num_epochs_max2 = 1000
LR2 = jax.device_put(1e-1)
nphotons_speed2 = jax.device_put(10e4)
rad_speed2 = jax.device_put(10e-6)
xy_speed2 = jax.device_put(10e-6)
z_speed2=jax.device_put(10e-6)
d_speed2 = jax.device_put(10e-6)

# extraction parameters
NPSF = nstack # nb of PSF per batch
sigma = np.std(psf3D.flatten())

#%%   #################### gradient descent ##################

# microscope parameters
d_ = -float(d[1])#-jax.device_put(d[1])
second_plane = jax.device_put(jnp.array([d[1]-d[0], 0, d[1]-d[2]])) #lower plane - middle - higher
polar_projections = jax.device_put(jnp.array([0, 45, 0]))

N=jax.device_put(jnp.array(80))
l_pixel=jax.device_put(jnp.array(16))
NA=jax.device_put(jnp.array(1.4))
mag=jax.device_put(jnp.array(100))
lambd=jax.device_put(jnp.array(lambda_emission))
f_tube=jax.device_put(jnp.array(200))
MAG=jax.device_put(jnp.array(200/150))

if SAF:                                             
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k_, f_o, cos_th2 = vectorial_BFP_perfect_focus_jax_unpolarized(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic, SAF=SAF)
else:
    costh2=None
    xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k_, f_o = vectorial_BFP_perfect_focus_jax_unpolarized(N, NA=NA, mag=mag, lambd_nm=lambd, f_tube_mm=f_tube, J_dichroic=J_dichroic)

u, v, Npadding = padding_jax(r, r_cut, k_, f_o,  N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, 
           f_tube=f_tube, MAG=MAG)

phase_mask = jnp.stack([jnp.ones((N,N)), jnp.ones((N,N)), jnp.ones((N,N))])
zernike_base = generate_zernike_base_jax(r_cut=r_cut, N=N, zernike_order=4, skip_indices = {0, 1, 2, 8,9,10,11,12,13,14}) #
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
phase_mask = pad_jax(phase_mask, Npadding).astype(jnp.float32)
zernike_base = pad_jax(zernike_base, Npadding).astype(jnp.complex64)
if SAF:
    cos_th2 = pad_jax(cos_th2, Npadding).astype(jnp.complex64)

# strating parameters (could be a first evaluation with coarse algo)
x_start = jax.device_put(0).astype(jnp.float32)
y_start = jax.device_put(0).astype(jnp.float32)
z_exp =  jax.device_put(0.5).astype(jnp.float32)

# gradient descent parameters
rho_start = jnp.array([45. for k in range(NPSF)]).astype(jnp.float32)
eta_start = jnp.array([90. for k in range(NPSF)]).astype(jnp.float32)
delta_start = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
Nstart_test = jnp.array(100000).astype(jnp.float32)
bck_test = jnp.array(300*jnp.ones((NPSF,3,2))).astype(jnp.float32)

Mtest = compute_M_jax(xp=x_start*jnp.array([1 for i in range(nstack)]), yp=y_start*jnp.array([1 for i in range(nstack)]), zp=z_exp*jnp.array([1 for i in range(nstack)]), d=d_+signe_stack*jnp.linspace(-0.03*(nstack//2)//2,0.03*(nstack//2),nstack), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                , zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                ,  second_plane=second_plane
              , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, SAF=SAF, cos_th2=cos_th2)
htest = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=Nstart_test*jnp.array([1 for i in range(nstack)])).astype(jnp.float32)

dim_simu = int(htest.shape[-1]//2)

# functions for the SGD steps
@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step1(params, opt_state, Nphotons_speed1, z_speed, rad_speed, d_speed, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu):
    loss, grads = jax.value_and_grad(loss_pos)(params, Nphotons_speed1, z_speed, rad_speed, d_speed, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu)
    updates, opt_state = optimizer1.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@functools.partial(jax.jit, static_argnames=['dim_simu'])#, donate_argnums=(0, 1))
def step2(params, opt_state, rho_start, eta_start, delta_start, nphotons_speed2, rad_speed2, d_speed2, xy_speed2, z_speed2, data, background, sigma, dim_simu):
    loss, grads = jax.value_and_grad(loss_angle_with_M)(params, rho_start, eta_start, delta_start, nphotons_speed2, rad_speed2, d_speed2, xy_speed2, z_speed2, data, background, sigma, dim_simu)
    updates, opt_state = optimizer2.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

optimizer1 = optax.adam(learning_rate=LR1)
optimizer2 = optax.adam(learning_rate=LR2)


################# main loop ########################
for bead in range(psf3D.shape[0]):

    angle_rd1 = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
    angle_rd2 = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
    optimizer = optax.adam(learning_rate=LR1)
    params = {
        'xp': x_start,
        'yp': y_start,
        'zp': z_exp/z_speed,
        'd': d_/d_speed,
        'N_photons': Nstart_test / Nphotons_speed1,
        'background': bck_test.flatten() / background_speed,
        'bead_radius': 100./rad_speed
    }
    opt_state = optimizer.init(params)
    
    loss_ = []
    z__ = []
    N__ = []
    x__ =[]
    bck = []
    d__ = []
    br = []
    
    for i in tqdm(range(num_epochs_max1)):
        params, opt_state, loss = step1(params, opt_state, Nphotons_speed1, z_speed, rad_speed, d_speed, background_speed, angle_rd2, angle_rd2, angle_rd1, psf3D[bead], second_plane, sigma, dim_simu)
        loss_.append(float(loss))
        z__.append(np.array(params['zp']*z_speed))
        N__.append(np.array(params['N_photons']) * Nphotons_speed1)
        x__.append(np.array(params['xp']))
        bck.append(np.array(params['background'] )* background_speed)
        d__.append(np.array(params['d'])*d_speed)
        br.append(np.array(params['bead_radius']*rad_speed))
    
    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(loss_)
    ax[0,1].plot(br)
    ax[0,2].plot(N__)
    ax[1,0].plot(x__)
    ax[1,1].plot(bck)
    ax[1,2].plot(d__)
    plt.show()
    del(ax, loss_, z__, N__, x__, bck, br)

    x_found0 = params['xp']
    y_found0 = params['yp']
    z_found0 = params['zp']*z_speed
    N_found0 = params['N_photons'] * Nphotons_speed1
    background_array_found = params['background'] * background_speed
    d_found0 = params['d']*d_speed
    bead_radius_found0 = params['bead_radius']*rad_speed
    del(params, loss)
    

################################ second SGD on orientation #########################################################################################
    
    zern_x = jnp.zeros(3*15)
    zern_y = jnp.zeros(3*15)

    params = {
    'zern_x': zern_x,
    'zern_y': zern_y,
    'N_photons': N_found0/nphotons_speed2,
    'x': x_found0/xy_speed2,
    'y': y_found0/xy_speed2,
    'z': z_found0/z_speed2,
    'd': d_found0/d_speed2,
    'bead_radius': bead_radius_found0/rad_speed2,
    'phase_maskx':phase_mask,
    'phase_masky':phase_mask
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
    br = []
    
    for i in tqdm(range(num_epochs_max2)):
        params, opt_state, loss = step2(params, opt_state, rho_start, eta_start, delta_start, nphotons_speed2, rad_speed2, d_speed2, xy_speed2, z_speed2, psf3D[bead], background_array_found, sigma, dim_simu)
        loss_.append(float(loss))
        zern_x_.append(np.array(params['zern_x'])*1000/(2*np.pi))
        zern_y_.append(np.array(params['zern_y'])*1000/(2*np.pi) )
        d__.append(np.array(params['d'])*d_speed2)
        x_.append(np.array(params['x'] )* xy_speed2)
        z_.append(np.array(params['z'] )* z_speed2)
        Np_.append(np.array(params['N_photons'] )* nphotons_speed2)
        br.append(np.array(params['bead_radius']*rad_speed2))
            
    fig, ax = plt.subplots(4,2)
    ax[0,0].plot(loss_) 
    ax[0,1].plot(zern_x_)
    ax[1,0].plot(zern_y_)
    ax[1,1].plot(d__)
    ax[2,0].plot(x_)
    ax[2,1].plot(br)
    ax[3,0].plot(Np_)
    plt.show()

    N_found2 = params['N_photons']*nphotons_speed2
    x_found = params['x']*xy_speed2
    y_found = params['y']*xy_speed2
    z_found = params['z']*z_speed2
    d_found = params['d']*d_speed2
    zernx_found = jnp.reshape(params['zern_x'], (3,15))
    zerny_found = jnp.reshape(params['zern_y'], (3,15))
    bead_radius_found = params['bead_radius']*rad_speed2
    phase_maskx_found = params['phase_maskx']
    phase_masky_found = params['phase_masky']
    del(params, loss)
    
    
    x_found = np.array(x_found)
    y_found = np.array(y_found)
    # the following manipulation is needed for consistency of the parametrization 
    # of the angles, because they are let free to be optimized without bound
    if True:
            
        Mtest1 = compute_M_jax(xp=x_found0*jnp.array([1 for i in range(nstack)]), yp=y_found0*jnp.array([1 for i in range(nstack)]), zp=z_found0*jnp.array([1 for i in range(nstack)])*0+0.1, d=d_found0+signe_stack*jnp.linspace(-0.03*(nstack//2),0.03*(nstack//2),nstack), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                        , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                        , zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                        ,  second_plane=second_plane
                      , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, SAF=SAF, cos_th2=cos_th2)
        htest1 = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest1, N_photons=N_found0*jnp.array([1 for i in range(nstack)])).astype(jnp.float32)
        R = bead_radius_found0 / 120
        H, W = htest1.shape[-2], htest1.shape[-1]
        
        yg = np.arange(H) - H // 2
        xg = np.arange(W) - W // 2
        yyy, xxx = np.meshgrid(yg, xg, indexing='ij')  # shape (H, W)
        
        # 2D disk kernel in xy only
        rr2 = jnp.asarray(xxx**2 + yyy**2)
        r = jnp.sqrt(rr2 + 1e-4)
        
        edge_width = 0.5
        sphere2d = jax.nn.sigmoid((R - r) / edge_width)  # shape (H, W)
        
        denom = jnp.clip(sphere2d.sum(), 1e-8, None)
        sphere2d = sphere2d / denom
        
        # broadcast to h shape (Z, dim1, dim2, H, W)
        kern = sphere2d[None, None, None, :, :]
        kern = jnp.broadcast_to(kern, htest1.shape)
        
        kern = jnp.fft.ifftshift(kern, axes=(3, 4))
        
        psf_conv1 = jnp.real(jnp.fft.ifftn(
            jnp.fft.fftn(htest1, axes=(3, 4)) * jnp.fft.fftn(kern, axes=(3, 4)),
            axes=(3, 4)
        ))
        #######################
        Mtest = compute_M_jax(xp=x_found*jnp.array([1 for i in range(nstack)]), yp=y_found*jnp.array([1 for i in range(nstack)]), zp=z_found*jnp.array([1 for i in range(nstack)])*0+0.1, d=d_found+signe_stack*jnp.linspace(-0.03*(nstack//2),0.03*(nstack//2),nstack), x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                        , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                        , zernike_coefs_x=zernx_found, zernike_coefs_y=zerny_found, phase_maskx=jnp.exp(1j*phase_maskx_found), phase_masky=jnp.exp(1j*phase_masky_found)
                        ,  second_plane=second_plane
                      , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, SAF=SAF, cos_th2=cos_th2)
        htest = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=N_found2*jnp.array([1 for i in range(nstack)])).astype(jnp.float32)
        R = bead_radius_found / 120
        H, W = htest.shape[-2], htest.shape[-1]
        
        yg = np.arange(H) - H // 2
        xg = np.arange(W) - W // 2
        yyy, xxx = np.meshgrid(yg, xg, indexing='ij')  # shape (H, W)
        
        # 2D disk kernel in xy only
        rr2 = jnp.asarray(xxx**2 + yyy**2)
        r = jnp.sqrt(rr2 + 1e-4)
        
        edge_width = 0.5
        sphere2d = jax.nn.sigmoid((R - r) / edge_width)  # shape (H, W)
        
        denom = jnp.clip(sphere2d.sum(), 1e-8, None)
        sphere2d = sphere2d / denom
        
        # broadcast to h shape (Z, dim1, dim2, H, W)
        kern = sphere2d[None, None, None, :, :]
        kern = jnp.broadcast_to(kern, htest.shape)
        
        kern = jnp.fft.ifftshift(kern, axes=(3, 4))
        
        psf_conv2 = jnp.real(jnp.fft.ifftn(
            jnp.fft.fftn(htest, axes=(3, 4)) * jnp.fft.fftn(kern, axes=(3, 4)),
            axes=(3, 4)
        ))
        key = jax.random.PRNGKey(0)
        #psf_conv2 = noise_jax(key, psf_conv2, QE=QE, EM=EM, b=200., sigma_b=10., sigma_r=150., bias=10.)
        PLAN=35
        for PLAN in [35,55]:
            sigma=0.5
            vmin=max(np.min(psf3D[bead,PLAN]),np.min(psf_conv1[PLAN]),np.min(psf_conv2[PLAN]))*1.1
            vmax=max(np.max(psf3D[bead,PLAN]),np.max(psf_conv1[PLAN]),np.max(psf_conv2[PLAN]))*1.1
            #norm=LogNorm(vmin=vmin, vmax=vmax)
            norm = Normalize(vmin=vmin, vmax=vmax)
            middle = psf_conv2.shape[-1]//2
            fig, ax = plt.subplots(3,2)
            ax[0,0].imshow(psf3D[bead,PLAN,0,0], norm=norm)
            ax[1,0].imshow(psf3D[bead,PLAN,1,0], norm=norm)
            ax[2,0].imshow(psf3D[bead,PLAN,2,0], norm=norm)
            ax[0,1].imshow(psf3D[bead,PLAN,0,1],norm=norm)
            ax[1,1].imshow(psf3D[bead,PLAN,1,1], norm=norm)
            ax[2,1].imshow(psf3D[bead,PLAN,2,1], norm=norm)
            plt.show()      
            fig, ax = plt.subplots(3,2)
            middle = htest1.shape[-1]//2

            ax[0,0].imshow(np.random.poisson(psf_conv1[PLAN,0,0,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[1,0].imshow(np.random.poisson(psf_conv1[PLAN,1,0,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[2,0].imshow(np.random.poisson(psf_conv1[PLAN,2,0,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[0,1].imshow(np.random.poisson(psf_conv1[PLAN,0,1,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[1,1].imshow(np.random.poisson(psf_conv1[PLAN,1,1,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[2,1].imshow(np.random.poisson(psf_conv1[PLAN,2,1,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            plt.show()
            fig, ax = plt.subplots(3,2)

            ax[0,0].imshow(np.random.poisson(psf_conv2[PLAN,0,0,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[1,0].imshow(np.random.poisson(psf_conv2[PLAN,1,0,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[2,0].imshow(np.random.poisson(psf_conv2[PLAN,2,0,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[0,1].imshow(np.random.poisson(psf_conv2[PLAN,0,1,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[1,1].imshow(np.random.poisson(psf_conv2[PLAN,1,1,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            ax[2,1].imshow(np.random.poisson(psf_conv2[PLAN,2,1,middle-10:middle+11,middle-10:middle+11])+np.random.normal(np.mean(background_array_found), sigma, psf3D[bead,PLAN,0,0].shape), norm=norm)
            plt.show()
        plt.plot(np.max(np.sum(psf3D[bead,:,0], axis=-3), axis=(1,2)), c='r')
        plt.plot(np.max(np.sum(psf3D[bead,:,1], axis=-3), axis=(1,2)), c='r')
        plt.plot(np.max(np.sum(psf3D[bead,:,2], axis=-3), axis=(1,2)), c='r')
        plt.plot(np.max(np.sum(psf_conv1[:,0], axis=-3), axis=(1,2)), c='b')
        plt.plot(np.max(np.sum(psf_conv1[:,1], axis=-3), axis=(1,2)), c='b')
        plt.plot(np.max(np.sum(psf_conv1[:,2], axis=-3), axis=(1,2)), c='b')
        plt.plot(np.max(np.sum(psf_conv2[:,0], axis=-3), axis=(1,2)), c='g')
        plt.plot(np.max(np.sum(psf_conv2[:,1], axis=-3), axis=(1,2)), c='g')
        plt.plot(np.max(np.sum(psf_conv2[:,2], axis=-3), axis=(1,2)), c='g')
        plt.show()
        
    fig, ax = plt.subplots(3,2)
    ax[0,0].imshow(np.real(phase_maskx_found[0]))
    ax[1,0].imshow(np.real(phase_maskx_found[1]))
    ax[2,0].imshow(np.real(phase_maskx_found[2]))
    ax[0,1].imshow(np.real(phase_masky_found[0]))
    ax[1,1].imshow(np.real(phase_masky_found[1]))
    ax[2,1].imshow(np.real(phase_masky_found[2]))
    plt.show()
    np.savez_compressed(parent / f"fit_{int(bead)}.npz", 
                        x_field = np.array(x),
                        y_field = np.array(y),
                        x=np.array(x_found), 
                        y=np.array(y_found), z=np.array(1000*z_found), 
                        N_photons=np.array(N_found2), 
                        d_found=np.array(d_found),
                        zernx_found=np.array(zernx_found),
                        zerny_found=np.array(zerny_found),
                        background_array_found=np.array(background_array_found))
