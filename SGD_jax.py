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
from simu_PSF_polarMFM_JAX import *
import matplotlib.pyplot as plt
from extract_experimental_psf import *
import gc
import optax
import copy
from tqdm import tqdm
import shutil
# %% PARAMETERS TO BE DEFINED
lambda_emission = jax.device_put(638) # nm
middle_plane = jax.device_put(1.2)
interplane = jax.device_put(0.385)
QE = jax.device_put(0.92)
EM = jax.device_put(200)
sensitivity = jax.device_put(15.4)
Nframe=jax.device_put(100)
total_n_frame = jax.device_put(7999)
path_info = '\\\\NAS_LOCCO\\Amaury\\DATA\\4polar_data_raw\\2026_02_02_SLB_1um\\SM_tres_haut\\Calib_Polar_2026-02-02\\images\\RAW_DATA\\image_Pos0.ome_results_fr1to15000_method=Propagation matrix_box-method=Fixed_box5.csv'
path_data_folder = '\\\\NAS_LOCCO\\Amaury\\DATA\\4polar_data_raw\\2026_02_02_SLB_1um\\SM_tres_haut\\Calib_Polar_2026-02-02\\images\\RAW_DATA\\image_Pos0_reco_concat\\'

N_batch = jax.device_put(20000)
batch_offset = jax.device_put(0)

#%%
raw = jnp.zeros((Nframe,6,214,129))

d = jnp.array([middle_plane-interplane, middle_plane, middle_plane+interplane])

# %% calibration data
'''mode = 'polar projections'
J_dichroic = npjnparray([[0.7838338      ,               -0.25981125 + 1j*  -0.48329058],[
      -0.4230177 + 1j*  0.27765664  ,   -0.7788276 + 1j*  -0.28660256
]]) # this one is for the abstract
'''
J1 = jnp.array([[ 1.2604369        ,             -0.44922367 + 1j*  0.6776327 ],[
      -0.40610462 + 1j*  0.6554575  ,   -1.2775537 + 1j*  0.034650166 ]])
J2 = jnp.array([[ 0.57820666               ,      -1.2006966 + 1j*  0.67720294 ],[
      1.0710695 + 1j*  0.8422108   ,  0.2654659 + 1j*  0.58898413 ]])
J_dichroic = jnp.array([J1, J2, J1])

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
    index_frame=jnp.array(index_frame)
    return x, y, z, rho, delta, index_frame

def limit(x, lim, slope, upper=True):
    if upper:
        return jnp.sum(jnp.exp((x-lim)*slope))
    else:
        return jnp.sum(jnp.exp(-1*(x-lim)*slope))
    
def loss_pos(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, plot):
    Mj = compute_M_jax(xp=params['xp'], yp=params['yp'], zp=params['zp'], d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    dim_data = 6
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=Mj, N_photons=params['N_photons']*Nphotons_speed1)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]

    loss = torch.sum(torch.pow(torch.sum(torch.add(h+torch.reshape(params['background']*background_speed, (h.shape[0],3,2))[:, :, :, None, None], -data), dim=(2,)), 2))

    x_bound = limit(xp, 5*0.12, 100, upper=True) + limit(xp, -5*0.12, 100, upper=False)
    y_bound = limit(yp, 5*0.12, 100, upper=True) + limit(yp, -5*0.12, 100, upper=False)
    z_bound = limit(zp, 5., 100, upper=True) + limit(zp, 0, 100, upper=False)
    return (loss +x_bound+y_bound+z_bound).astype(jnp.float32)

def loss_angle_with_M(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, plot):
    dim_data = 6
    Mj = compute_M_jax(xp=params['x_fine']*xy_speed2, yp=params['y_fine']*xy_speed2, zp=params['z_fine']*z_speed2, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=jnp.reshape(zernx, (3,15)), zernike_coefs_y=jnp.reshape(zerny, (3,15))
                    , second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    h = PSF_jax(rho=params['rho'], eta=params['eta'], delta=params['delta']*delta_speed, M=Mj, N_photons=params['N_photons']*nphotons_speed2)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    
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
    return (loss + 1000.*(delta_bound)).astype(jnp.float32) #+ N_bound #+ 100000*torch.sum(h**2)

def score_eval(M_, rho, eta, delta, N_photons, data, background, sigma, dim_simu):
    dim_data = 6
    h = PSF_jax(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    score = jnp.sum(jnp.add(h, -(data+sigma**2)*jnp.log(h+background+sigma**2)), dim=(1,2,3,4))
    return score

#%% extracting data

data = pos_from_csv(path_info)

# %% SGD PARANETERS TO DEFINE
Nphotons_speed1 = jax.device_put(10000)
background_speed = jax.device_put(40)
LR1 = jax.device_put(0.03)
num_epochs_max1 = jax.device_put(100)

num_epochs_max2 = jax.device_put(200)
LR2 = jax.device_put(0.7)
delta_speed = jax.device_put(1.)
nphotons_speed2 = jax.device_put(100)
xy_speed2 = jax.device_put(1/30)
z_speed2=jax.device_put(1)

save_folder = '\\\\NAS_LOCCO\\Amaury\\DATA\\polMFM_experimental_processed\\these_4polar_MFM\\test_jax'

#%% Gradient descent

jax.clear_caches()

intermediate_folder = 'working_folder_jax_sgd'

shutil.rmtree(intermediate_folder)       # delete the folder
os.makedirs(intermediate_folder)

for batch_number in range(N_batch):
    # extracteing the raw 6-stack tiff files
    raw, error_indices = jax.device_put(extract_frames((batch_number+batch_offset)*Nframe+1, Nframe))
    # extracting the position from Louise pipeline
    x, y, z, rho, delta, index_frame = jax.device_put(extract_positions((batch_number+batch_offset)*Nframe+1, Nframe, error_indices))
    # converting to photon count
    raw = raw*sensitivity/(QE*EM)
    # these quantites are used to evaluated the noise and inserted into the loss
    sigma = jnp.std(raw.flatten())
    background = jnp.mean(raw.flatten())
    L = raw.shape[2]*120
    W = raw.shape[3]*120
    # removing all the PSF where a parameter is evaluated to nan in Louise pipeline
    nb = len(x)
    L = raw.shape[2]*120
    W = raw.shape[3]*120
    for k, ele in enumerate(x):
        if jnp.isnan(x[nb-1-k]) or jnp.isnan(y[nb-1-k]) or jnp.isnan(z[nb-1-k]) or (y[nb-1-k]<6*120) or (x[nb-1-k]<6*120) or (x[nb-1-k]>L-6*120) or (y[nb-1-k]>W-6*120):
            x = jnp.delete(x,nb-1-k,0)
            y = jnp.delete(y,nb-1-k,0)
            z = jnp.delete(z,nb-1-k,0)
            rho = jnp.delete(rho,nb-1-k,0)
            delta = jnp.delete(delta,nb-1-k,0)
            index_frame = jnp.delete(index_frame,nb-1-k,0)
    # extracting the psf from the files
    single_psf = extract_raw_xy(raw[0], x[index_frame==0], y[index_frame==0])
    for i in range(1,Nframe):
        single_psf = jnp.concatenate((single_psf, extract_raw_xy(raw[i], x[index_frame==i], y[index_frame==i])))

    # dimenstion matching to have x in horizontal and y in vertical when considering what appears in a tiff file
    single_psf = single_psf[:,::-1,:,::-1,:]
    x, y = y, -x

    # nb of photons by plane roughly evaluated
    Nstart_by_plane = copy.deepcopy(jnp.sum(single_psf, axis=(2,3,4)) - background*len(single_psf[0,0].flatten()))

    # removing all the PSF where there are two emitters, either too bright or the middle plane less bright than the extremal ones
    nb = len(x)
    '''
    for k, ele in enumerate(x):
        if ((Nstart_by_plane[k,0]>Nstart_by_plane[k,1]) & (Nstart_by_plane[k,2]>Nstart_by_plane[k,1])) | (Nstart_by_plane[k,0]+Nstart_by_plane[k,1]+Nstart_by_plane[k,2]<2000):
            x = np.delete(x,nb-1-k,0)
            y = np.delete(y,nb-1-k,0)
            z = np.delete(z,nb-1-k,0)
            rho = np.delete(rho,nb-1-k,0)
            delta = np.delete(delta,nb-1-k,0)
            index_frame = np.delete(index_frame,nb-1-k,0)
            single_psf = np.delete(single_psf,nb-1-k,0)
    '''
    NPSF = len(x)
    print('NPSF = ', NPSF)
    # strating parameters (could be a first evaluation with coarse algo)
    x_start = jax.device_put(jnp.array([0. for k in range(len(x))]))
    y_start = jax.device_put(jnp.array([0. for k in range(len(x))]))
    z_exp =  jax.device_put(jnp.array([0.7 for k in range(len(x))])) 

    # microscope parameters
    d_ = -jax.device_put(jnp.array([d[1] for k in range(len(x))]))
    second_plane = jax.device_put(jnp.array([d[1]-d[0], 0, d[1]-d[2]]))
    polar_projections = jax.device_put(jnp.array([0, 45, 0]))

    if batch_number==0:
        N=jax.device_put(jnp.array(80))
        l_pixel=jax.device_put(jnp.array(16))
        NA=jax.device_put(jnp.array(1.4))
        mag=jax.device_put(jnp.array(100))
        lambd=jax.device_put(jnp.array(lambda_emission))
        f_tube=jax.device_put(jnp.array(200))
        MAG=jax.device_put(jnp.array(200/150))
        
        SAF = False
    
        if SAF:
            xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k_, f_o, costh2 = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, J_dichroic=J_dichroic)
        else:
            costh2=None
            xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k_, f_o = vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, J_dichroic=J_dichroic)
    
        u, v, Npadding = padding_jax(r, r_cut, k_, f_o,  N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, 
                  f_tube=f_tube, MAG=MAG)

        phase_mask = jnp.stack([jnp.ones((N,N)), jnp.ones((N,N)), jnp.ones((N,N))])
        zernike_base = generate_zernike_base_jax(r_cut=r_cut, N=N, zernike_order=4)
        zernike_coefs_x = jnp.zeros((3,15)).astype(jnp.complex64)
        zernike_coefs_y = jnp.zeros((3,15)).astype(jnp.complex64)
        '''zernike_coefs_x = torch.tensor([[ 0.        , -0.01509399,  0.08022121,  0.26224712, -0.39965165,
        -0.14400212,  0.06210446,  0.07706548, -0.03673849, -0.0059061 ,
         0.3918969 , -0.03448999, -0.2760792 ,  0.03974261, -0.07540362],
       [ 0.        ,  0.12417129,  0.0379636 ,  0.36084166,  0.18248007,
         0.16796161,  0.01906732,  0.03188295, -0.04461008,  0.00852952,
         0.2397723 ,  0.03531132,  0.03633038,  0.16416775,  0.19273765],
       [ 0.        , -0.07786292, -0.02687996,  0.02495545,  0.14203313,
        -0.14452088,  0.00205685,  0.05500394, -0.01802518, -0.00095783,
         0.04286083, -0.01074904,  0.1067887 ,  0.03914205,  0.01560213]], device=device)
        zernike_coefs_y = torch.tensor([[ 0.        ,  0.09011961, -0.01196684,  0.16113627,  0.31367758,
        -0.03343723, -0.00863215,  0.07828876, -0.04595673, -0.00473705,
         0.2601863 , -0.05901701,  0.160533  , -0.02936446,  0.01919452],
       [ 0.        ,  0.03916434, -0.03085124,  0.24977088,  0.13083968,
        -0.3738273 , -0.0195394 ,  0.01909356, -0.0176624 , -0.0158725 ,
         0.21947987, -0.12261663,  0.0596002 ,  0.11330672,  0.04495693],
       [ 0.        ,  0.02453548, -0.08325748,  0.15265286, -0.03383344,
         0.10807846, -0.02959344,  0.05958771, -0.01800941,  0.00523958,
         0.15786456,  0.10830489, -0.04955807,  0.08272237,  0.00375486]], device=device)
        '''
        
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

    # convert to tensor
    noisy_psf = jnp.array(jnp.array([single_psf[k] for k in range(len(x))]))

    # gradient descent parameters
    rho_start = jnp.array([45.for k in range(NPSF)]).astype(jnp.float32)
    eta_start = jnp.array([90. for k in range(NPSF)]).astype(jnp.float32)
    delta_start = jnp.array([80. for k in range(NPSF)]).astype(jnp.float32)
    Nstart = jnp.array(jnp.sum(Nstart_by_plane, axis=1)).astype(jnp.float32)
    
    Mtest = compute_M_jax(xp=x_start, yp=y_start, zp=z_exp, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                    , zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y
                    ,  second_plane=second_plane
                  , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    htest = PSF_jax(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=Nstart).astype(jnp.float32)
    
    dim_simu = int(htest.shape[-1]//2)

    background_array = jnp.tensor(background*jnp.ones((NPSF,3,2))).astype(jnp.float32)
    params = {
    'xp': x_start,
    'yp': y_start,
    'zp': z_exp,
    'N_photons': Nstart/Nphotons_speed1,
    'background': background_array.flatten()/background_speed
    }.to(jnp.float32)

    angle_rd1 = jnp.array([180. for k in range(NPSF)]).astype(jnp.float32)
    angle_rd2 = jnp.array([45. for k in range(NPSF)]).astype(jnp.float32)
    optimizer = optax.adam(learning_rate=LR1)
    opt_state = optimizer.init(params)
    
    loss_ = []
    z__ = []
    N__ = []
    x__ =[]
    bck = []
    @jax.jit
    def step(params, opt_state, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, plot):
        loss, grads = jax.value_and_grad(loss_pos)(params, Nphotons_speed1, background_speed, rho, eta, delta, data, second_plane, sigma, dim_simu, plot)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    for i in tqdm(range(num_epochs_max1)):
        params, opt_state, loss = step(params, opt_state, Nphotons_speed1, background_speed, angle_rd2, angle_rd2, angle_rd1, noisy_psf, second_plane, sigma, dim_simu, plot=False)
        loss_.append(loss)
        z__.append(params['zp'])
        N__.append((params['N_photons']*Nphotons_speed1))
        x__.append((params['xp']).cpu())
        bck.append(params['background']*background_speed)
    
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
    N_found = params['N_photons']*Nphotons_speed1
    background_array_found = params['background']*background_speed
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
    }.to(jnp.float32)
    optimizer = optax.adam(learning_rate=LR2)
    opt_state = optimizer.init(params)
    
    loss_ = []
    eta_ = []
    rho_ = []
    delta_ = []
    x_ = []
    z_ = []
    Np_ = []
    
    @jax.jit
    def step(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, plot):
        loss, grads = jax.value_and_grad(loss_angle_with_M)(params, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zernx, zerny, data, background, sigma, dim_simu, plot)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    for i in tqdm(range(num_epochs_max1)):
        params, opt_state, loss = step(params, opt_state, delta_speed, nphotons_speed2, xy_speed2, z_speed2, zern_x, zern_y, noisy_psf, background_array_found, sigma, dim_simu, plot=False)
        loss_.append(loss)
        eta_.append(params['eta'])
        rho_.append(params['rho'])
        delta_.append(params['delta']*delta_speed)
        x_.append(params['x']*xy_speed2)
        z_.append(params['z']*z_speed2)
        Np_.append(params['N_photons']*nphotons_speed2)
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

    mask = (rho_found>180)
    eta_found[mask] = (180-eta_found[mask])%180
    rho_found = rho_found%180
    
    delta_found=params['delta']*delta_speed
    N_found2 = params['N_photons']*nphotons_speed2
    x_found = params['x']*xy_speed2
    y_found = params['y']*xy_speed2
    z_found = params['z']*z_speed2
    zernx = jnp.reshape(zern_x, (3,15)).detach()
    zerny = jnp.reshape(zern_x, (3,15)).detach()
    del(params, loss)
    M = compute_M_jax(xp=x_found, yp=y_found, zp=z_found, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v, zernike_base=zernike_base
                    , zernike_coefs_x=zernx, zernike_coefs_y=zerny
                    ,  second_plane=second_plane
                  , polar_projections=polar_projections, lambd=lambd, f_tube=f_tube)
    score = score_eval(M.detach(), rho_found, eta_found, delta_found, N_found2, noisy_psf.cpu(), background, sigma, dim_simu)
    x_ = (x/120).astype(int)*120 + 1000*x_found
    y_ = (y/120).astype(int)*120 + 1000*y_found
    jnp.savez_compressed(save_folder+'\\'+str(int(batch_number)+1+batch_offset)+'.npz', 
                        frame = Nframe*(batch_number+batch_offset)+index_frame, x=x_, y=y_, 
                        z=1000*z_found, N_photons=N_found2, 
                        rho=rho_found, eta=eta_found, 
                        delta=delta_found, score=score, x_start=x, y_start=y, z_start=z,
                        rho_start=rho, delta_start=delta, background_array_found=background_array_found)
