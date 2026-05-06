# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:32:09 2026

@author: Amaury Autric
amaury.autric@polytechnique.edu
data processing in pyTorch for fitting the 3D position and 3D orientation in polMFM
"""
# %% Libraries
import sys
import os
import torch
# Add the parent directory to the Python path
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import copy
from simu_PSF_polarMFM import *
from extract_experimental_psf import *
from tqdm import tqdm
from torch.optim import SGD, Adam, AdamW

# %%

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:   
    device = torch.device('cpu')
    print("Using CPU")
    
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
batch_offset = 11

# %% 
raw = np.zeros((Nframe,6,214,129))

d = np.array([middle_plane-interplane, middle_plane, middle_plane+interplane])

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
    '''
    if upper:
       return torch.sum(torch.tensor(1/(1+torch.exp(-slope*(x-lim))), requires_grad=True, device=device))
    else:
        return torch.sum(torch.tensor(1/(1+torch.exp(slope*(x-lim))), requires_grad=True, device=device))
    '''
    if upper:
        return torch.sum(torch.exp((x-lim)*slope))
    else:
        return torch.sum(torch.exp(-1*(x-lim)*slope))
    
def loss_pos_torch(xp, yp, zp, rho, eta, delta, N_photons, data, second_plane, background, sigma, dim_simu, plot):
    M_ = compute_M(xp=xp, yp=yp, zp=zp, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, r=r, r_cut=r_cut, k=k_, f_o=f_o, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_x,
                        second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, 
                         device=device, mode=mode)
    dim_data = 6
    h = PSF(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]

    loss = torch.sum(torch.pow(torch.sum(torch.add(h+torch.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), dim=(2,)), 2))

    x_bound = limit(xp, 5*0.12, 100, upper=True) + limit(xp, -5*0.12, 100, upper=False)
    y_bound = limit(yp, 5*0.12, 100, upper=True) + limit(yp, -5*0.12, 100, upper=False)
    z_bound = limit(zp, 5., 100, upper=True) + limit(zp, 0, 100, upper=False)
    return (loss +x_bound+y_bound+z_bound).to(torch.float32)
#loss_pos = torch.compile(loss_pos_torch)
loss_pos = loss_pos_torch

def loss_angle_with_M_torch(rho, eta, delta, N_photons, x_fine, y_fine, z_fine, zernx, zerny, data, background, sigma, dim_simu, plot):
    dim_data = 6
    M_ = compute_M(xp=x_fine, yp=y_fine, zp=z_fine, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, r=r, r_cut=r_cut, k=k_, f_o=f_o, phase_masky=phase_mask, phase_maskx=phase_mask, zernike_base=zernike_base, zernike_coefs_x=torch.reshape(zernx, (3,15)), zernike_coefs_y=torch.reshape(zerny, (3,15)),
                        second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, device=device, mode=mode)
    h = PSF(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    
    #loss = torch.sum(torch.add(h, -(data+sigma**2)*torch.log(h+torch.reshape(background, (h.shape[0],3,2))[:, :, :, None, None]+sigma**2)))
    loss = torch.sum(torch.pow(torch.sum(torch.add(h+torch.reshape(background, (h.shape[0],3,2))[:, :, :, None, None], -data), dim=(2,)), 2))
    delta_bound = limit(delta, 180, 100, upper=True) + limit(delta, 1, 100, upper=False)
    if plot:
        for nb in range(data.shape[0]):
            maxi = max(np.max(data[nb,:,:].flatten().cpu().detach().numpy()), np.max(h[nb,:,:].flatten().cpu().detach().numpy()))
            fig, ax = plt.subplots(3,2)
            ax[0,0].imshow(data[nb,0,0].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[0,0].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(data[nb,1,0].cpu().detach().numpy() , vmin=0., vmax=maxi, cmap='gray')
            ax[1,0].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(data[nb,2,0].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[2,0].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(data[nb,0,1].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[0,1].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(data[nb,1,1].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[1,1].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(data[nb,2,1].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[2,1].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            plt.show()
            del(fig, ax)
            fig, ax = plt.subplots(3,2)
            ax[0,0].imshow(h[nb,0,0].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[0,0].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[1,0].imshow(h[nb,1,0].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[1,0].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[2,0].imshow(h[nb,2,0].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[2,0].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[0,1].imshow(h[nb,0,1].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[0,1].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[1,1].imshow(h[nb,1,1].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[1,1].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            ax[2,1].imshow(h[nb,2,1].cpu().detach().numpy(), vmin=0., vmax=maxi, cmap='gray')
            ax[2,1].scatter(x_fine[nb].cpu().detach().numpy()/0.120+6, y_fine[nb].cpu().detach().numpy()/0.120+6, s=10, c='r', marker='x')
            plt.show()
            del(fig, ax)
    return (loss + 1000.*(delta_bound)).to(torch.float32) #+ N_bound #+ 100000*torch.sum(h**2)
#loss_angle_with_M = torch.compile(loss_angle_with_M_torch)
loss_angle_with_M = loss_angle_with_M_torch

def score_eval(M_, rho, eta, delta, N_photons, data, background, sigma, dim_simu):
    dim_data = 6
    h = PSF(rho=rho, eta=eta, delta=delta, M=M_, N_photons=N_photons)[:,:,:,dim_simu-dim_data:dim_simu+dim_data+1,dim_simu-dim_data:dim_simu+dim_data+1]
    score = torch.sum(torch.add(h, -(data+sigma**2)*torch.log(h+background+sigma**2)), dim=(1,2,3,4))
    return score.numpy() 

# %% preloc extraction
data = pos_from_csv(path_info)

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

# %% SGD PARANETERS TO DEFINE
Nphotons_speed1 = 10000
background_speed = 40
LR1 = 0.03
num_epochs_max1 = 100

num_epochs_max2 = 200
LR2 = 0.7
delta_speed = 1.
nphotons_speed2 = 100
xy_speed2 = 1/30
z_speed2=1

save_folder = '\\\\NAS_LOCCO\\Amaury\\DATA\\polMFM_experimental_processed\\these_4polar_MFM\\sperm_cricket_11'
# %%gradient descent
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

for batch_number in range(N_batch):
    # extracteing the raw 6-stack tiff files
    raw, error_indices = extract_frames((batch_number+batch_offset)*Nframe+1, Nframe)
    # extracting the position from Louise pipeline
    x, y, z, rho, delta, index_frame = extract_positions((batch_number+batch_offset)*Nframe+1, Nframe, error_indices)
    # converting to photon count
    raw = raw*sensitivity/(QE*EM)
    # these quantites are used to evaluated the noise and inserted into the loss
    sigma = np.std(raw.flatten())
    background = np.mean(raw.flatten())
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

    # nb of photons by plane roughly evaluated
    Nstart_by_plane = copy.deepcopy(np.sum(single_psf, axis=(2,3,4)) - background*len(single_psf[0,0].flatten()))

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
    x_start = torch.tensor([0. for k in range(len(x))], requires_grad=False, device=device)
    y_start = torch.tensor([0. for k in range(len(x))], requires_grad=False, device=device)
    z_exp =  torch.tensor([0.7 for k in range(len(x))], requires_grad=False, device=device) 

    # microscope parameters
    d_ = -torch.tensor([d[1] for k in range(len(x))], requires_grad=False, device=device)
    second_plane = torch.tensor([d[1]-d[0], 0, d[1]-d[2]], device=device, requires_grad=False)
    polar_projections = np.array([0, 45, 0])

    if batch_number==0:
        N=torch.tensor(80, device=device, requires_grad=False)
        l_pixel=torch.tensor(16, device=device, requires_grad=False)
        NA=torch.tensor(1.4, device=device, requires_grad=False)
        mag=torch.tensor(100, device=device, requires_grad=False)
        lambd=torch.tensor(lambda_emission, device=device, requires_grad=False)
        f_tube=torch.tensor(200, device=device, requires_grad=False)
        MAG=torch.tensor(200/150, device=device, requires_grad=False)
        
        SAF = False
    
        if SAF:
            xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k_, f_o, costh2 = vectorial_BFP_perfect_focus(N, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF, device=device, J_dichroic=J_dichroic)
        else:
            costh2=None
            xx, yy, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k_, f_o = vectorial_BFP_perfect_focus(N, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF, device=device, J_dichroic=J_dichroic)
    
        u, v, Npadding = padding(r, r_cut, k_, f_o,  N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, 
                  f_tube=f_tube, MAG=MAG, device=device)

        phase_mask = torch.stack([torch.ones((N,N), device=device), torch.ones((N,N), device=device), torch.ones((N,N), device=device)])
        zernike_base = generate_zernike_base(r_cut=r_cut, N=N, zernike_order=4, device=device)
        zernike_coefs_x = torch.zeros((3,15), device=device).to(torch.complex64)
        zernike_coefs_y = torch.zeros((3,15), device=device).to(torch.complex64)
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
        
        xx = pad(xx, Npadding).to(torch.complex64).detach()
        yy = pad(yy, Npadding).to(torch.complex64).detach()
        th1 = pad(th1, Npadding).to(torch.complex64).detach()
        phi = pad(phi, Npadding).to(torch.complex64).detach()
        Ex0 = pad(Ex0, Npadding).to(torch.complex64).detach()
        Ex1 = pad(Ex1, Npadding).to(torch.complex64).detach()
        Ex2 = pad(Ex2, Npadding).to(torch.complex64).detach()
        Ey0 = pad(Ey0, Npadding).to(torch.complex64).detach()
        Ey1 = pad(Ey1, Npadding).to(torch.complex64).detach()
        Ey2 = pad(Ey2, Npadding).to(torch.complex64).detach()
        phase_mask = pad(phase_mask, Npadding).to(torch.complex64).detach()
        zernike_base = pad(zernike_base, Npadding).to(torch.complex64).detach()
        if SAF:
            costh2 = pad(costh2, Npadding).to(torch.complex64).detach()

    # convert to tensor
    noisy_psf = torch.tensor(np.array([single_psf[k] for k in range(len(x))]), device=device, dtype=torch.float32)

    # gradient descent parameters
    rho_start = torch.tensor([45.for k in range(NPSF)], device=device).to(torch.float32)
    eta_start = torch.tensor([90. for k in range(NPSF)], requires_grad=False, device=device).to(torch.float32)
    delta_start = torch.tensor([80. for k in range(NPSF)], device=device).to(torch.float32)
    Nstart = torch.tensor(np.sum(Nstart_by_plane, axis=1), device=device).to(torch.float32)
    
    Mtest = compute_M(xp=x_start, yp=y_start, zp=z_exp, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, r=r, r_cut=r_cut, k=k_, f_o=f_o, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_y,
                        second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, device=device, mode=mode)
    
    htest = PSF(rho=rho_start, eta=eta_start, delta=delta_start, M=Mtest, N_photons=Nstart).to(torch.float32)
    
    dim_simu = int(htest.shape[-1]//2)

    background_array = torch.tensor(background*np.ones((NPSF,3,2)) ,device=device).to(torch.float32)
    
    params = torch.cat((x_start, y_start, z_exp, Nstart/Nphotons_speed1, background_array.flatten()/background_speed)).to(torch.float32)
    params.requires_grad=True

    angle_rd1 = torch.tensor([180. for k in range(NPSF)], requires_grad=False, device=device).to(torch.float32)
    angle_rd2 = torch.tensor([45. for k in range(NPSF)], requires_grad=False, device=device).to(torch.float32)
    optimizer = torch.optim.Adam([params], lr=LR1)
    
    loss_ = []
    z__ = []
    N__ = []
    x__ =[]
    bck = []
    for i in tqdm(range(num_epochs_max1)):
        optimizer.zero_grad()  # Reset gradients
        loss = loss_pos(params[0:NPSF], params[NPSF:2*NPSF], params[2*NPSF:3*NPSF], 
                           angle_rd2, angle_rd2, angle_rd1, params[3*NPSF:4*NPSF]*Nphotons_speed1, 
                           noisy_psf, second_plane, params[4*NPSF:10*NPSF]*background_speed, sigma, dim_simu, plot=False)
        loss_.append(loss.cpu().detach().numpy())
        z__.append(params[2*NPSF:3*NPSF].cpu().detach().numpy())
        N__.append((params[3*NPSF:4*NPSF]*Nphotons_speed1).cpu().detach().numpy())
        x__.append((params[0*NPSF:1*NPSF]).cpu().detach().numpy())
        bck.append(params[4*NPSF:10*NPSF].cpu().detach().numpy()*background_speed)
        loss.backward()
        optimizer.step()
    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(loss_)
    ax[0,1].plot(z__)
    ax[0,2].plot(N__)
    ax[1,0].plot(x__)
    ax[1,1].plot(bck)
    plt.show()
    del(ax, loss_, z__, N__, x__, bck)

    x_found = params[0:NPSF].detach()
    y_found = params[NPSF:2*NPSF].detach()
    z_found = params[2*NPSF:3*NPSF].detach()
    N_found = params[3*NPSF:4*NPSF].detach()*Nphotons_speed1
    background_array_found = params[4*NPSF:10*NPSF].detach()*background_speed
    del(params, loss)
    

################################ second SGD on orientation #########################################################################################
    
    zern_x = torch.tensor(np.zeros(3*15), device=device)
    zern_y = torch.tensor(np.zeros(3*15), device=device)

    params = torch.cat((rho_start, eta_start, delta_start/delta_speed, N_found/nphotons_speed2, x_found/xy_speed2, y_found/xy_speed2, z_found/z_speed2))
    params.requires_grad=True

    # Use Stochastic Gradient Descent (SGD) to optimize params
    optimizer = torch.optim.Adam([params], lr=LR2)  # Learning rate = 0.01
    
    loss_ = []
    eta_ = []
    rho_ = []
    delta_ = []
    x_ = []
    z_ = []
    Np_ = []
    for i in tqdm(range(num_epochs_max2)):
        optimizer.zero_grad()  # Reset gradients
        loss = loss_angle_with_M(params[:NPSF], params[1*NPSF:2*NPSF], params[2*NPSF:3*NPSF]*delta_speed, params[3*NPSF:4*NPSF]*nphotons_speed2, params[4*NPSF:5*NPSF]*xy_speed2, params[5*NPSF:6*NPSF]*xy_speed2, params[6*NPSF:7*NPSF]*z_speed2, zern_x, zern_y, noisy_psf, background_array_found, sigma, dim_simu, plot=False)#(i==num_epochs_max-1))
        loss_.append(loss.cpu().detach().numpy())
        eta_.append(params[1*NPSF:2*NPSF].cpu().detach().numpy())
        rho_.append(params[0*NPSF:1*NPSF].cpu().detach().numpy())
        delta_.append(params[2*NPSF:3*NPSF].cpu().detach().numpy()*delta_speed)
        x_.append(params[4*NPSF:5*NPSF].cpu().detach().numpy()*xy_speed2)
        z_.append(params[6*NPSF:7*NPSF].cpu().detach().numpy()*z_speed2)
        Np_.append(params[3*NPSF:4*NPSF].cpu().detach().numpy()*nphotons_speed2)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
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

    rho_found=params[0:NPSF].detach()%360
    eta_found=params[1*NPSF:2*NPSF].detach()%180

    mask = (rho_found>180)
    eta_found[mask] = (180-eta_found[mask])%180
    rho_found = rho_found%180
    
    delta_found=params[2*NPSF:3*NPSF].detach()*delta_speed
    N_found2 = params[3*NPSF:4*NPSF].detach()*nphotons_speed2
    x_found = params[4*NPSF:5*NPSF].detach()*xy_speed2
    y_found = params[5*NPSF:6*NPSF].detach()*xy_speed2
    z_found = params[6*NPSF:7*NPSF].detach()*z_speed2
    zernx = torch.reshape(zern_x, (3,15)).detach()
    zerny = torch.reshape(zern_x, (3,15)).detach()
    del(params, loss)
    M = compute_M(xp=x_found, yp=y_found, zp=z_found, d=d_, x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2
                    , Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, r=r, r_cut=r_cut, k=k_, f_o=f_o, phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, zernike_coefs_y=zernike_coefs_x,
                        second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, device=device, mode=mode)
    score = score_eval(M.detach().cpu(), rho_found.cpu(), eta_found.cpu(), delta_found.cpu(), N_found2.cpu(), noisy_psf.cpu(), background_array_found.cpu(), sigma, dim_simu)
    x_ = (x/120).astype(int)*120 + 1000*x_found.cpu().detach().numpy()
    y_ = (y/120).astype(int)*120 + 1000*y_found.cpu().detach().numpy()
    np.savez_compressed(save_folder+'\\'+str(int(batch_number)+1+batch_offset)+'.npz', frame = Nframe*(batch_number+batch_offset)+index_frame, x=x_, y=y_, z=1000*z_found.cpu().detach().numpy(), N_photons=N_found2.cpu().detach().numpy(), rho=rho_found.cpu().detach().numpy(), eta=eta_found.cpu().detach().numpy(), delta=delta_found.cpu().detach().numpy(), score=score, x_start=x, y_start=y, z_start=z, rho_start=rho, delta_start=delta, background_array_found=background_array_found.cpu().detach().numpy())
