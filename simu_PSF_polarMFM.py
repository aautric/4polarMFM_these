# -*- coding: utf-8 -*-
"""
Simple microscope model
see PSF_demo.ipynb for explanations
amaury.autric@curie.fr
"""

import numpy as np
import torch
from numpy.random import normal, poisson
import matplotlib.pyplot as plt
from zernike import RZern

n1 = 1.52 # oil and sample indices
n2 = 1.33 # water index
h = 6.626*10**(-34) # Planck constant
c = 299792458 # speed of light

# everything in micrometers 
    
def vectorial_BFP_perfect_focus(N, NA=1.4, mag=100, lambd=617, f_tube=200, SAF=False, device='cpu', J_dichroic=None):
    #compute parameters
    lambd = 10**(-3)*lambd
    f_tube = f_tube*1000
    f_o = f_tube/mag
    k = 2*n1*np.pi/lambd # wave vector
    # the maximum spatial frequency could be limited by the aperture or the total reflection on the coverslip
    if isinstance(NA, torch.Tensor):
        if SAF:
            r_cut = NA.detach().cpu().numpy()/n1
            r_cut_saf = n2/n1
        else:
            r_cut = min(NA.detach().cpu().numpy()/n1, n2/n1) 
    else:
        if SAF:
            r_cut = NA/n1
            r_cut_saf = n2/n1
        else:
            r_cut = min(NA/n1, n2/n1) 
    #x, y, and r are dimensionless
    x, y = np.meshgrid(np.linspace(-r_cut,r_cut,N), np.linspace(-r_cut,r_cut,N))
    r = np.sqrt(x**2+y**2) # radial coordinate, dimensionless, proportional to spatial frequency
    phi = np.arctan2(y,x) # angle of radial coordinate in the BFP
    # see Louise's thesis for the descritption of these parameters (p 116)
    th1 = np.zeros((N,N))
    th2 = np.zeros((N,N))
    th1[r<r_cut] = np.arcsin(r[r<r_cut])
    if SAF:
        th2[r<r_cut_saf] = np.arcsin((n1/n2)*r[r<r_cut_saf])
    else:
        th2[r<r_cut] = np.arcsin((n1/n2)*r[r<r_cut])
    
    costh2 = np.cos(th2).astype(complex)
    if SAF: # Born & Wolf p50-51
        costh2[(r<r_cut)&(r>r_cut_saf)] = 1j*(np.sqrt((np.sin(th1[(r<r_cut)&(r>r_cut_saf)])*(n1/n2))**2-1))        
    else:
        costh2 = costh2.astype(float)
    #transmission coefficients
    Ts = (2*n2*costh2) / (n2*costh2 + n1*np.cos(th1))
    Tp = (2*n2*costh2) / (n2*np.cos(th1) + n1*costh2)
    
    # compute the fields
    Ex0 = ((n1/n2) * ((np.cos(th1)/costh2)*Ts*(np.sin(phi)**2) + Tp*(np.cos(phi)**2)*np.cos(th1)))/np.sqrt(np.cos(th1))
    Ex1 = (-((n1*np.sin(2*phi))/(2*n2))*((np.cos(th1)*Ts)/costh2 - Tp*np.cos(th1)))/np.sqrt(np.cos(th1))
    Ex2 = (-((n1/n2)**2)*(np.cos(th1)/costh2)*Tp*np.cos(phi)*np.sin(th1))/np.sqrt(np.cos(th1))
    Ex0[r>r_cut]=0.
    Ex1[r>r_cut]=0.
    Ex2[r>r_cut]=0.

    Ey0 = (-0.5*np.sin(2*phi)*(n1/n2)*((np.cos(th1)/costh2)*Ts - Tp*np.cos(th1)))/np.sqrt(np.cos(th1))
    Ey1 = ((n1/n2)*((np.cos(th1)/costh2)*Ts*(np.cos(phi)**2)+Tp*np.cos(th1)*(np.sin(phi)**2)))/np.sqrt(np.cos(th1))
    Ey2 = (-((n1/n2)**2)*(np.cos(th1)/costh2)*Tp*np.sin(phi)*np.sin(th1))/np.sqrt(np.cos(th1))
    Ey0[r>r_cut]=0.
    Ey1[r>r_cut]=0.
    Ey2[r>r_cut]=0.
    
    if J_dichroic is not None:
        if len(J_dichroic.shape)==3:
            Ex0_ = np.zeros((J_dichroic.shape[0], Ex0.shape[0], Ex0.shape[1])).astype(complex)
            Ex1_ = np.zeros((J_dichroic.shape[0], Ex0.shape[0], Ex0.shape[1])).astype(complex)
            Ex2_ = np.zeros((J_dichroic.shape[0], Ex0.shape[0], Ex0.shape[1])).astype(complex)
            Ey0_ = np.zeros((J_dichroic.shape[0], Ex0.shape[0], Ex0.shape[1])).astype(complex)
            Ey1_ = np.zeros((J_dichroic.shape[0], Ex0.shape[0], Ex0.shape[1])).astype(complex)
            Ey2_ = np.zeros((J_dichroic.shape[0], Ex0.shape[0], Ex0.shape[1])).astype(complex)
            for plane in range(J_dichroic.shape[0]):
                Ex0_[plane], Ey0_[plane] = J_dichroic[plane,0,0]*Ex0+J_dichroic[plane,0,1]*Ey0, J_dichroic[plane,1,0]*Ex0+J_dichroic[plane,1,1]*Ey0
                Ex1_[plane], Ey1_[plane] = J_dichroic[plane,0,0]*Ex1+J_dichroic[plane,0,1]*Ey1, J_dichroic[plane,1,0]*Ex1+J_dichroic[plane,1,1]*Ey1
                Ex2_[plane], Ey2_[plane] = J_dichroic[plane,0,0]*Ex2+J_dichroic[plane,0,1]*Ey2, J_dichroic[plane,1,0]*Ex2+J_dichroic[plane,1,1]*Ey2
                Ex0, Ex1, Ex2, Ey0, Ey1, Ey2 = Ex0_, Ex1_, Ex2_, Ey0_, Ey1_, Ey2_
        else:
            Ex0_, Ey0_ = J_dichroic[0,0]*Ex0+J_dichroic[0,1]*Ey0, J_dichroic[1,0]*Ex0+J_dichroic[1,1]*Ey0
            Ex1_, Ey1_ = J_dichroic[0,0]*Ex1+J_dichroic[0,1]*Ey1, J_dichroic[1,0]*Ex1+J_dichroic[1,1]*Ey1
            Ex2_, Ey2_ = J_dichroic[0,0]*Ex2+J_dichroic[0,1]*Ey2, J_dichroic[1,0]*Ex2+J_dichroic[1,1]*Ey2
    
    if isinstance(N, torch.Tensor):
        '''
        case of torch : has to return tensor objects.
        '''
        x = torch.tensor(x, device=device, requires_grad=False)
        y = torch.tensor(y, device=device, requires_grad=False)
        th1 = torch.tensor(th1, device=device, requires_grad=False)
        phi = torch.tensor(phi, device=device, requires_grad=False)
        Ex0 = torch.tensor(Ex0, device=device, requires_grad=False)
        Ex1 = torch.tensor(Ex1, device=device, requires_grad=False)
        Ex2 = torch.tensor(Ex2, device=device, requires_grad=False)
        Ey0 = torch.tensor(Ey0, device=device, requires_grad=False)
        Ey1 = torch.tensor(Ey1, device=device, requires_grad=False)
        Ey2 = torch.tensor(Ey2, device=device, requires_grad=False)
        r = torch.tensor(r, device=device, requires_grad=False)
        r_cut = torch.tensor(r_cut, device=device, requires_grad=False)
        costh2 = torch.tensor(costh2, device=device, requires_grad=False)
    
    if SAF:
        return x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k, f_o, costh2
    else:
        return x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k, f_o

'''   these three functions returns the phase shift in the BFP when changing the focus,and the position of the emitter '''

def psi_lat(x, y, theta, phi, lambd=0.617):
    ''' a translation in the real space is a multiplication by an exponential in the Fourier space'''
    if isinstance(x, torch.Tensor): 
        ''' torch case'''
        if len(x.shape)==0: 
            ''' case of a single PSF simu'''
            return torch.sin(theta)*(x*torch.cos(phi)+y*torch.sin(phi))*(2*torch.pi*n1)/(lambd)
        else: 
            ''' when simulating several PSF'''
            return (torch.reshape(torch.outer(x, (torch.sin(theta)*torch.cos(phi)).flatten()), torch.Size(x.shape + theta.shape))
        +torch.reshape(torch.outer(y, (torch.sin(theta)*torch.sin(phi)).flatten()), 
                       torch.Size(y.shape + theta.shape)))*(2*torch.pi*n1)/(lambd)
    else: 
        ''' numpy case'''
        return np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi))*(2*np.pi*n1)/(lambd)

def psi_z(theta, z, lambd=0.617, costh2=None):
    ''' version Yan et al . corrected '''
    if isinstance(z, torch.Tensor):
        if len(z.shape)==0:
            ''' if one PSF '''
            if costh2 is not None:
                return 2*torch.pi*n2*z*costh2/lambd
            else:
                return 2*torch.pi*n2*z*torch.sqrt(1-(n1*torch.sin(theta)/n2)**2)/lambd
        else:
            ''' if several PSF '''
            if costh2 is not None: 
                return (2*torch.pi*n2/lambd)*torch.einsum('n,uv->nuv', z, costh2)
            else:
                return torch.reshape(torch.outer(z, 2*torch.pi*n2*torch.sqrt(1-(n1*torch.sin(theta.flatten())/n2)**2)/lambd), torch.Size(z.shape + theta.shape))
    else:
        ''' numpy '''
        if costh2 is not None:
            return 2*np.pi*n2*z*costh2/lambd
        else:
            return 2*np.pi*n2*z*np.sqrt(1-(n1*np.sin(theta)/n2)**2)/lambd
    
def psi_f(theta, d, lambd=0.617):
    ''' version Yan et al. corrected'''
    if isinstance(d, torch.Tensor):
        n = n1 + (n2-n1)*(1+torch.sign(d))/2
        if len(d.shape)==0:
            ''' if one PSF '''
            return 2*torch.pi*n*torch.cos(theta)*d/lambd
        else:
            ''' if several PSF '''
            return torch.reshape(2*torch.pi*torch.outer(d*n, torch.sqrt(1-(torch.sin(theta.flatten()))**2))/lambd, torch.Size(d.shape + theta.shape))
    else:
        ''' numpy '''
        if d<0:
            return 2*np.pi*n1*np.cos(theta)*d/lambd
        else:
            return 2*np.pi*n2*np.cos(theta)*d/lambd

def generate_zernike_base(r_cut, N, zernike_order=4, device='cpu'):
    cart = RZern(zernike_order)        
    if (device!='cpu') & (isinstance(N, torch.Tensor)):
        ddx = np.linspace(-r_cut.cpu(), r_cut.cpu(), N.cpu())
        ddy = np.linspace(-r_cut.cpu(), r_cut.cpu(), N.cpu())
    else:        
        ddx = np.linspace(-r_cut, r_cut, N)
        ddy = np.linspace(-r_cut, r_cut, N)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    
    zernike_base = np.zeros(((zernike_order+1)*(zernike_order+2)//2, xv.shape[0], xv.shape[1]))
    zer = np.zeros(cart.nk)
    for index in range(1,cart.nk):
        zer[index]=1.
        zernike_base[index] = cart.eval_grid(zer, matrix=True)
        zernike_base[index][np.isnan(zernike_base[index])] = 0.
        if isinstance(r_cut, torch.Tensor):
            zernike_base[index][(xv**2+yv**2)>r_cut.cpu().numpy()**2]=0.
        else:
            zernike_base[index][(xv**2+yv**2)>r_cut**2]=0.
        zer[index]=0.
    if isinstance(r_cut, torch.Tensor):
        zernike_base = torch.tensor(zernike_base, device=device)
    return zernike_base
    

def BFP_phase(theta, phi, d, xp, yp, zp, r_cut, N=80, NA=1.4, mag=100, lambd=0.617, f_tube=200000, 
              SAF=False, costh2=None, zernike_base=None, zernike_coefs_x=np.zeros(15), zernike_coefs_y=np.zeros(15)):
    if isinstance(phi, torch.Tensor):
        print('Function only in numpy case')
        return None
    if (SAF==True) & (costh2 is None):
        raise ValueError("costh2 need to be given if SAF=True")
    if SAF:
        phase = psi_f(theta, d, lambd=lambd)+psi_z(theta, zp, lambd=lambd, costh2=costh2)+psi_lat(xp, yp, theta, phi, lambd=lambd)
    else:
        phase = psi_f(theta, d, lambd=lambd)+psi_z(theta, zp, lambd=lambd)+psi_lat(xp,yp,theta,phi, lambd=lambd)
    phase[np.isnan(phase)] = 0.
    if zernike_base is not None:
        zernike_phase_x = np.sum(zernike_coefs_x.reshape(-1, 1, 1)*zernike_base, axis=0)
        zernike_phase_y = np.sum(zernike_coefs_y.reshape(-1, 1, 1)*zernike_base, axis=0)
        total_phase_x = zernike_phase_x + phase 
        total_phase_y = zernike_phase_y + phase
        return (total_phase_x)%(2*np.pi) ,  (total_phase_y)%(2*np.pi) 
    else:
        if SAF:
            return (np.real(phase))%(2*np.pi)
        else:
            return (phase)%(2*np.pi)

def padding(r, r_cut, k, f_o,  N=80, l_pixel=16, NA=1.4, mag=100, lambd=617, 
              f_tube=200, MAG=200/150, device='cpu'):
    lambd = 10**(-3)*lambd # conversion nm to micrometers
    f_tube = f_tube*1000 # tube length focal. everything is in micrometers
    ''' Padding for matching the pixel size in real space '''
    Dx = 2*r_cut*f_o/N # discretization of BFP in length units
    ''' elements to add to make it match with pixel size '''
    if isinstance(r, torch.Tensor): 
        ''' torch version ''' 
        Npadding = ((2*torch.pi*MAG*f_tube)/(k*l_pixel*Dx)).to(torch.int32) - N
        if Npadding%2==1:
            Npadding=Npadding+1
        freq = (torch.fft.fftshift(torch.fft.fftfreq(N+Npadding, Dx, device=device))*2*torch.pi*f_tube/k)*MAG
        ### WARNING torch.meshgrid() returns exactly the opposite of np.meshgrid() !!!!
        v, u = torch.meshgrid(freq, freq)
    else: 
        ''' numpy version ''' 
        Npadding = int((2*np.pi*MAG*f_tube)/(k*l_pixel*Dx)) - N
        if Npadding%2==1:
            Npadding=Npadding+1
        freq = (np.fft.fftshift(np.fft.fftfreq(N+Npadding, Dx))*2*np.pi*f_tube/k)*MAG
        u, v = np.meshgrid(freq, freq)
    return u, v, Npadding

def pad(a, n):
    '''
    This functions add a padding of n at each side. Used for matching pixel size
    '''
    n0 = a.shape[1]
    type = a.dtype
    if isinstance(a, torch.Tensor):
        device = a.device
        if len(a.shape)==2:
            b = torch.zeros((n+n0, n+n0), dtype=type, device=device)
            b[:] = 0.
            b[n//2:-n//2,n//2:-n//2] = a
        else:
            b = torch.zeros((a.shape[0], n+n0, n+n0), dtype=type, device=device)
            b[:] = 0.
            b[:,n//2:-n//2,n//2:-n//2] = a
    else: 
        if len(a.shape)==2:
            b = np.zeros((n+n0, n+n0)).astype(type)
            b[:] = 0.
            b[n//2:-n//2,n//2:-n//2] = a
        if len(a.shape)==3:
            b = np.zeros((a.shape[0], n+n0, n+n0)).astype(type)
            b[:] = 0.
            b[:,n//2:-n//2,n//2:-n//2] = a
    return b
        
def compute_M(xp, yp, zp, d, x, y, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2, r, r_cut, k, f_o, phase_maskx, 
              phase_masky, zernike_base=None, zernike_coefs_x=None, zernike_coefs_y=None, 
              second_plane=None, polar_projections=None, lambd=617, 
              f_tube=200, device='cpu', BFP_version=False, 
              SAF=False, costh2=None, mode='polar projections'):
    ''' This function takes all the geometrical parameters, dipole position and focal plane
    It also takes the microcope-dependant fields Ex0 ...
    If xp, yp, ... are tensors of length N, then N psf are computed
    if you put also second_planes>0, then two planes are simulated for the N psf (the second is projected onto 45, 135)
    It could be computed using GPU ('cuda')
    It returns u, v, meshgrids describing the image plane coordinates
    and the M matrix
    This matrix is of shape
        (2, 3, 3, N_pix, N_pix) if a single PSF is simulated (the 2 is because of the two polarizations projections)
        (N, 2, 3, 3,, N_pix, N_pix) if N PSF are simulated
        if second_plane is not None it is of form (N, K, 2, 3, 3,, N_pix, N_pix), with K being the indices for planes shifted 
        (the shift is in second_plane (negative convention again)), and projected on polarizations that are in 
        polar_projections (+90)
    '''
    lambd = 10**(-3)*lambd # conversion nm to micrometers
    f_tube = f_tube*1000 # tube length focal. everything is in micrometers
    
    if (Ex0.shape[-1]!=phase_maskx.shape[-1]):
        raise ValueError("Shape mismatch")
    if (SAF==True) & (costh2 is None):
        raise ValueError("costh2 need to be given if SAF=True")
        
    if isinstance(xp, torch.Tensor): 
        ''' torch version ''' 
        if second_plane!=None: 
            ''' several plane case ''' 
            if type(second_plane)==float or type(second_plane)==int: 
                ''' just in case we give second_plane under the wrong form '''
                second_plane = [second_plane]
                polar_projections = [polar_projections]
            if SAF:
                phase = torch.stack([torch.exp((1j*(psi_f(th1, d+second_plane[ind], lambd=lambd)+psi_z(th1, zp, lambd=lambd, costh2=costh2)+psi_lat(xp,yp,th1,phi, lambd=lambd))).to(torch.complex64)) for ind in range(len(second_plane))])
            else:#
                phase = torch.stack([torch.exp((1j*(psi_f(th1, d+second_plane[ind], lambd=lambd)+psi_z(th1, zp, lambd=lambd)+psi_lat(xp,yp,th1,phi, lambd=lambd))).to(torch.complex64)) for ind in range(len(second_plane))])
        else:
            ''' single plane '''     
            if SAF:
                phase = torch.exp(1j*(psi_f(th1, d, lambd=lambd)+psi_z(th1, zp, lambd=lambd, costh2=costh2)+psi_lat(xp,yp,th1,phi, lambd=lambd)))
            else:
                phase = torch.exp(1j*(psi_f(th1, d, lambd=lambd)+psi_z(th1, zp, lambd=lambd)+psi_lat(xp,yp,th1,phi, lambd=lambd)))
    else:
        ''' numpy '''
        if SAF:
            phase = np.exp(1j*(psi_f(th1, d, lambd=lambd)+psi_z(th1, zp, lambd=lambd, costh2=costh2)+psi_lat(xp,yp,th1,phi, lambd=lambd)))
        else:
            phase = np.exp(1j*(psi_f(th1, d, lambd=lambd)+psi_z(th1, zp, lambd=lambd)+psi_lat(xp,yp,th1,phi, lambd=lambd)))
        phase[np.isnan(phase)] = 0.
        
    ##############################################################################
    '''aberrations computations'''
    if zernike_base is not None:
        if isinstance(phi, torch.Tensor):
            '''torch case'''
            if second_plane==None:
                zernike_mask_x = torch.exp(1j * torch.sum(zernike_coefs_x.view(-1, 1, 1)*zernike_base, dim=0))
                zernike_mask_y = torch.exp(1j * torch.sum(zernike_coefs_y.view(-1, 1, 1)*zernike_base, dim=0))
            else:
                ''' in case there are several planes, the parameters are lilstede in the plaene order 
                (the same as in several_planes, which give the distance from plane to a fictive nominal plane, 
                 and polar_proj)'''
                zernike_mask_x = torch.exp((1j * torch.sum(torch.einsum('ab, bcd->abcd', zernike_coefs_x, zernike_base), dim=1).to(torch.complex64)))
                zernike_mask_y = torch.exp((1j * torch.sum(torch.einsum('ab, bcd->abcd', zernike_coefs_y, zernike_base), dim=1).to(torch.complex64)))
        else:
            '''numpy case'''
            zernike_mask_x = np.exp(1j * np.sum(zernike_coefs_x.reshape(-1, 1, 1)*zernike_base, axis=0))
            zernike_mask_y = np.exp(1j * np.sum(zernike_coefs_y.reshape(-1, 1, 1)*zernike_base, axis=0))
    else:
            if isinstance(phi, torch.Tensor):
                '''torch case'''
                if second_plane==None:
                    zernike_mask_x = torch.ones(Ex0.shape[-1], Ex0.shape[-1]).to(device)
                    zernike_mask_y = torch.ones(Ex0.shape[-1], Ex0.shape[-1]).to(device)
                else:
                    ''' in case there are several planes, the parameters are lilstede in the plaene order 
                    (the same as in several_planes, which give the distance from plane to a fictive nominal plane, 
                     and polar_proj)'''
                    zernike_mask_x = torch.ones(len(second_plane), Ex0.shape[-1], Ex0.shape[-1]).to(torch.complex64).to(device)
                    zernike_mask_y = torch.ones(len(second_plane), Ex0.shape[-1], Ex0.shape[-1]).to(torch.complex64).to(device)
            else:
                '''numpy case'''
                zernike_mask_x = np.ones(Ex0.shape[-1], Ex0.shape[-1])
                zernike_mask_y = np.ones(Ex0.shape[-1], Ex0.shape[-1])
    ##############################################################################
    '''computing the M matrix'''

    if isinstance(xp, torch.Tensor): 
        ''' torch version ''' 
        if len(phase.shape)==2: 
            ''' 1 PSF 
            compute the field with the phase masks that takes defocus and position in account
            and compute the field in image plane with fft '''
            if second_plane!=None:
                raise ValueError("Cannot do multi-plane in the case of single PSF (not implemented)")
            E00 = torch.fft.fftshift(torch.fft.fft2(phase_maskx*zernike_mask_x*Ex0*phase))
            E01 = torch.fft.fftshift(torch.fft.fft2(phase_maskx*zernike_mask_x*Ex1*phase))
            E02 = torch.fft.fftshift(torch.fft.fft2(phase_maskx*zernike_mask_x*Ex2*phase))
            
            E10 = torch.fft.fftshift(torch.fft.fft2(phase_masky*zernike_mask_y*Ey0*phase))
            E11 = torch.fft.fftshift(torch.fft.fft2(phase_masky*zernike_mask_y*Ey1*phase))
            E12 = torch.fft.fftshift(torch.fft.fft2(phase_masky*zernike_mask_y*Ey2*phase))

            if BFP_version:
                 '''version that is used for plotting the BFP intensity (no fft)'''
                 Ex_bfp = torch.stack([Ex0, Ex1, Ex2], dim=0)
                 Ey_bfp = torch.stack([Ey0, Ey1, Ey2], dim=0)
                 
                 Mx = torch.einsum('abc, ubc -> aubc', torch.conj(Ex_bfp), Ex_bfp) 
                 My = torch.einsum('abc, ubc -> aubc', torch.conj(Ey_bfp), Ey_bfp)
            else:
                 '''version that is used for plotting the image plane intensity (with fft)'''
                 Ex_im = torch.stack([E00, E01, E02], dim=0)
                 Ey_im = torch.stack([E10, E11, E12], dim=0)
                 
                 Mx = torch.einsum('abc, ubc -> aubc', torch.conj(Ex_im), Ex_im) 
                 My = torch.einsum('abc, ubc -> aubc', torch.conj(Ey_im), Ey_im)
            M = torch.stack([Mx, My], dim=0).to(torch.complex64)
             
        else: 
            ''' several PSF '''     
            if second_plane is not None: 
                ''' several planes '''
                ''' rotated polarizations as a function of the plane '''
                polar_rad = (polar_projections=='rad')
                polar_projections[polar_rad] = 0. # will be managed just after
                
                if mode=='polar projections': # the simple case consists in projecting on different angles for each plane
                # the update mode consists in calibrating each plane with a Stokes matrix, Then it is taken in account in the computation of Ex0 ...
                    polar_projections = torch.tensor(polar_projections.astype(float), device=device)
                    ex0 = torch.stack([torch.cos((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ex0 + torch.sin((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ey0 for ind in range(len(polar_projections))])
                    ex1 = torch.stack([torch.cos((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ex1 + torch.sin((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ey1 for ind in range(len(polar_projections))])
                    ex2 = torch.stack([torch.cos((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ex2 + torch.sin((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ey2 for ind in range(len(polar_projections))])
                    
                    ey0 = torch.stack([-torch.sin((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ex0 + torch.cos((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ey0 for ind in range(len(polar_projections))])
                    ey1 = torch.stack([-torch.sin((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ex1 + torch.cos((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ey1 for ind in range(len(polar_projections))])
                    ey2 = torch.stack([-torch.sin((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ex2 + torch.cos((polar_projections[ind]).to(torch.complex64)*torch.pi/180)*Ey2 for ind in range(len(polar_projections))])
                        
                '''Managing the case of radial/azimuthal polar'''
                if polar_rad.any(): 
                    for pl in range(len(second_plane)):
                        if polar_rad[pl]:
                            ex0[pl], ey0[pl] = ex0[pl]*torch.cos(phi) +  ey0[pl]*torch.sin(phi), -ex0[pl]*torch.sin(phi) +  ey0[pl]*torch.cos(phi)
                            ex1[pl], ey1[pl] = ex1[pl]*torch.cos(phi) +  ey1[pl]*torch.sin(phi), -ex1[pl]*torch.sin(phi) +  ey1[pl]*torch.cos(phi)
                            ex2[pl], ey2[pl] = ex2[pl]*torch.cos(phi) +  ey2[pl]*torch.sin(phi), -ex2[pl]*torch.sin(phi) +  ey2[pl]*torch.cos(phi)
                
                if BFP_version:
                     '''version that is used for plotting the BFP intensity (no fft)
                     version multiplane multi PSF
                     each Exx is shape Kplanes, u, v
                     adding the dim of NPSF artificially'''
                     E00 = torch.stack([phase_maskx[ind]*zernike_mask_x[ind]*phase[ind]*ex0[ind,:,:] for ind in range(len(polar_projections))]).unsqueeze(1).expand(len(second_plane), len(xp), phi.shape[-1], phi.shape[-2])
                     E01 = torch.stack([phase_maskx[ind]*zernike_mask_x[ind]*phase[ind]*ex1[ind,:,:] for ind in range(len(polar_projections))]).unsqueeze(1).expand(len(second_plane), len(xp), phi.shape[-1], phi.shape[-2])
                     E02 = torch.stack([phase_maskx[ind]*zernike_mask_x[ind]*phase[ind]*ex2[ind,:,:] for ind in range(len(polar_projections))]).unsqueeze(1).expand(len(second_plane), len(xp), phi.shape[-1], phi.shape[-2])
                     
                     E10 = torch.stack([phase_masky[ind]*zernike_mask_y[ind]*phase[ind]*ey0[ind,:,:] for ind in range(len(polar_projections))]).unsqueeze(1).expand(len(second_plane), len(xp), phi.shape[-1], phi.shape[-2])
                     E11 = torch.stack([phase_masky[ind]*zernike_mask_y[ind]*phase[ind]*ey1[ind,:,:] for ind in range(len(polar_projections))]).unsqueeze(1).expand(len(second_plane), len(xp), phi.shape[-1], phi.shape[-2])
                     E12 = torch.stack([phase_masky[ind]*zernike_mask_y[ind]*phase[ind]*ey2[ind,:,:] for ind in range(len(polar_projections))]).unsqueeze(1).expand(len(second_plane), len(xp), phi.shape[-1], phi.shape[-2])
                     
                else:
                    '''version that is used for plotting the image plane intensity (fft)
                    version multiplane multi PSF
                    each Exx is shape Kplanes, NPSF, u, v'''
                    if mode=='polar projections':
                        E00 = torch.stack([torch.fft.fftshift(torch.fft.fft2(phase_maskx[ind]*zernike_mask_x[ind]*phase[ind]*ex0[ind,:,:], dim=(-2, -1)), dim=(-2, -1)) for ind in range(len(polar_projections))])
                        E01 = torch.stack([torch.fft.fftshift(torch.fft.fft2(phase_maskx[ind]*zernike_mask_x[ind]*phase[ind]*ex1[ind,:,:], dim=(-2, -1)), dim=(-2, -1)) for ind in range(len(polar_projections))])
                        E02 = torch.stack([torch.fft.fftshift(torch.fft.fft2(phase_maskx[ind]*zernike_mask_x[ind]*phase[ind]*ex2[ind,:,:], dim=(-2, -1)), dim=(-2, -1)) for ind in range(len(polar_projections))])
                        
                        E10 = torch.stack([torch.fft.fftshift(torch.fft.fft2(phase_masky[ind]*zernike_mask_y[ind]*phase[ind]*ey0[ind,:,:], dim=(-2, -1)), dim=(-2, -1)) for ind in range(len(polar_projections))])
                        E11 = torch.stack([torch.fft.fftshift(torch.fft.fft2(phase_masky[ind]*zernike_mask_y[ind]*phase[ind]*ey1[ind,:,:], dim=(-2, -1)), dim=(-2, -1)) for ind in range(len(polar_projections))])
                        E12 = torch.stack([torch.fft.fftshift(torch.fft.fft2(phase_masky[ind]*zernike_mask_y[ind]*phase[ind]*ey2[ind,:,:], dim=(-2, -1)), dim=(-2, -1)) for ind in range(len(polar_projections))])
                    elif mode=='Stokes':
                        E00 = torch.fft.fftshift(torch.fft.fft2(phase_maskx[:,None,:,:]*zernike_mask_x[:,None,:,:]*phase*Ex0[:,None,:,:]), dim=(-2, -1))
                        E01 = torch.fft.fftshift(torch.fft.fft2(phase_maskx[:,None,:,:]*zernike_mask_x[:,None,:,:]*phase*Ex1[:,None,:,:]), dim=(-2, -1))
                        E02 = torch.fft.fftshift(torch.fft.fft2(phase_maskx[:,None,:,:]*zernike_mask_x[:,None,:,:]*phase*Ex2[:,None,:,:]), dim=(-2, -1))
                        E10 = torch.fft.fftshift(torch.fft.fft2(phase_masky[:,None,:,:]*zernike_mask_y[:,None,:,:]*phase*Ey0[:,None,:,:]), dim=(-2, -1))
                        E11 = torch.fft.fftshift(torch.fft.fft2(phase_masky[:,None,:,:]*zernike_mask_y[:,None,:,:]*phase*Ey1[:,None,:,:]), dim=(-2, -1))
                        E12 = torch.fft.fftshift(torch.fft.fft2(phase_masky[:,None,:,:]*zernike_mask_y[:,None,:,:]*phase*Ey2[:,None,:,:]), dim=(-2, -1))
                ''' M is the matrix of the basis functions 
                shape polar, 3, 3, Kplanes, NPSF, u, v'''
                M = torch.permute(torch.stack([
                    torch.stack([torch.stack([E00*torch.conj(E00), E00*torch.conj(E01), E00*torch.conj(E02)]), 
                                  torch.stack([E01*torch.conj(E00), E01*torch.conj(E01), E01*torch.conj(E02)]), 
                                  torch.stack([E02*torch.conj(E00), E02*torch.conj(E01), E02*torch.conj(E02)])]), 
                    torch.stack([torch.stack([E10*torch.conj(E10), E10*torch.conj(E11), E10*torch.conj(E12)]), 
                                  torch.stack([E11*torch.conj(E10), E11*torch.conj(E11), E11*torch.conj(E12)]),
                                   torch.stack([E12*torch.conj(E10), E12*torch.conj(E11), E12*torch.conj(E12)])])
                    ]), (4, 3, 0, 1, 2, 5, 6)).to(torch.complex64)
            
            else: 
                '''single plane sevral PSF'''
                if BFP_version: 
                    '''version that is used for plotting the BFP intensity (no fft)'''
                    Ex_bfp = torch.stack([Ex0*phase*phase_maskx*zernike_mask_x, Ex1*phase*phase_maskx*zernike_mask_x, Ex2*phase*phase_maskx*zernike_mask_x], dim=0)
                    Ey_bfp = torch.stack([Ey0*phase_masky*zernike_mask_y*phase, Ey1*phase_masky*zernike_mask_y*phase, Ey2*phase_masky*zernike_mask_y*phase], dim=0)
                    
                    Mx = torch.einsum('abc, ubc -> aubc', torch.conj(Ex_bfp), Ex_bfp) 
                    My = torch.einsum('abc, ubc -> aubc', torch.conj(Ey_bfp), Ey_bfp)
                    M = torch.stack([torch.stack([Mx, My], dim=0) for ind in range(len(d))])
                else: 
                    '''version that is used for plotting the image plane intensity (with fft)'''
                    E00 = torch.fft.fftshift(torch.fft.fft2(phase_maskx*zernike_mask_x*phase*Ex0[None,:,:], dim=(-2, -1)), dim=(-2, -1))
                    E01 = torch.fft.fftshift(torch.fft.fft2(phase_maskx*zernike_mask_x*phase*Ex1[None,:,:], dim=(-2, -1)), dim=(-2, -1))
                    E02 = torch.fft.fftshift(torch.fft.fft2(phase_maskx*zernike_mask_x*phase*Ex2[None,:,:], dim=(-2, -1)), dim=(-2, -1))
                    
                    E10 = torch.fft.fftshift(torch.fft.fft2(phase_masky*zernike_mask_y*phase*Ey0[None,:,:], dim=(-2, -1)), dim=(-2, -1))
                    E11 = torch.fft.fftshift(torch.fft.fft2(phase_masky*zernike_mask_y*phase*Ey1[None,:,:], dim=(-2, -1)), dim=(-2, -1))
                    E12 = torch.fft.fftshift(torch.fft.fft2(phase_masky*zernike_mask_y*phase*Ey2[None,:,:], dim=(-2, -1)), dim=(-2, -1))
                    M = torch.permute(torch.stack([
                        torch.stack([torch.stack([E00*torch.conj(E00), E00*torch.conj(E01), E00*torch.conj(E02)]), 
                                      torch.stack([E01*torch.conj(E00), E01*torch.conj(E01), E01*torch.conj(E02)]), 
                                      torch.stack([E02*torch.conj(E00), E02*torch.conj(E01), E02*torch.conj(E02)])]), 
                        torch.stack([torch.stack([E10*torch.conj(E10), E10*torch.conj(E11), E10*torch.conj(E12)]), 
                                      torch.stack([E11*torch.conj(E10), E11*torch.conj(E11), E11*torch.conj(E12)]),
                                       torch.stack([E12*torch.conj(E10), E12*torch.conj(E11), E12*torch.conj(E12)])])
                        ]), (3, 0, 1, 2, 4, 5)).to(torch.complex64)
    else:
        '''numpy version'''
        ''' u v coordinates in image plane'''     
        
        E00 = np.fft.fftshift(np.fft.fft2(Ex0*phase*phase_maskx*zernike_mask_x))
        E01 = np.fft.fftshift(np.fft.fft2(Ex1*phase*phase_maskx*zernike_mask_x))
        E02 = np.fft.fftshift(np.fft.fft2(Ex2*phase*phase_maskx*zernike_mask_x))
        
        E10 = np.fft.fftshift(np.fft.fft2(Ey0*phase*phase_masky*zernike_mask_y))
        E11 = np.fft.fftshift(np.fft.fft2(Ey1*phase*phase_masky*zernike_mask_y))
        E12 = np.fft.fftshift(np.fft.fft2(Ey2*phase*phase_masky*zernike_mask_y))
        
        if BFP_version:
            '''version that is used for plotting the BFP intensity (no fft)'''
            Ex_bfp = np.array([Ex0*phase*phase_maskx*zernike_mask_x, Ex1*phase*phase_maskx*zernike_mask_x, Ex2*phase*phase_maskx*zernike_mask_x])
            Ey_bfp = np.array([Ey0*phase*phase_masky*zernike_mask_y, Ey1*phase*phase_masky*zernike_mask_y, Ey2*phase*phase_masky*zernike_mask_y])
            Mx = np.einsum('abc, ubc -> aubc', np.conj(Ex_bfp), Ex_bfp) 
            My = np.einsum('abc, ubc -> aubc', np.conj(Ey_bfp), Ey_bfp)
        else:
            '''version that is used for plotting the image plane intensity (with fft)'''
            Ex_im = np.array([E00, E01, E02])
            Ey_im = np.array([E10, E11, E12])
            
            Mx = np.einsum('abc, ubc -> aubc', np.conj(Ex_im), Ex_im) 
            My = np.einsum('abc, ubc -> aubc', np.conj(Ey_im), Ey_im)
        M = np.array([Mx, My])
    #print(Npadding) 
    return M

def PSF(rho, eta, delta, M, N_photons=1000, device='cpu'):
    '''
    This function somputes the PSF from the orientation parameters and the calibration matrix M
    that contains templates. It could be computed using GPU ('cuda')
    If a single psf is computed the results has shape (2, Nx, Ny)
    the 2 is for the two polarization projections
    If N psf are computed (M has to be computed accordingly and the angles are tensors of length N), the the shape is (N, 2, Nx, Ny)
    If it is bi-plane (this comes from the parameters in the computation of M), the the shape is (N, 2_plane, 2_polar, Nx, Ny)
    '''
    if isinstance(eta, torch.Tensor):
        rho_ = rho * torch.pi / 180
        eta_ = eta * torch.pi / 180
        delta_ = delta * torch.pi / 180
    else:
        rho = rho * np.pi / 180
        eta = eta * np.pi / 180
        delta = delta * np.pi / 180

    N_photons = N_photons 
    
    if isinstance(eta, torch.Tensor):
        if len(eta.shape)==0:
            ''' rotation matrix if only 1 PSF is simulated'''
            R = torch.stack([
                torch.stack([(torch.sin(rho_)**2)*(1 - torch.cos(eta_)) + torch.cos(eta_), 
                             torch.sin(rho_) * torch.cos(rho_) * (torch.cos(eta_) - 1), 
                             torch.cos(rho_) * torch.sin(eta_)]),
                torch.stack([torch.sin(rho_) * torch.cos(rho_) * (torch.cos(eta_) - 1), 
                             (torch.cos(rho_)**2)*(1 - torch.cos(eta_)) + torch.cos(eta_), 
                             torch.sin(rho_) * torch.sin(eta_)]),
                torch.stack([-torch.cos(rho_) * torch.sin(eta_), 
                             -torch.sin(rho_) * torch.sin(eta_), 
                             torch.cos(eta_)])
            ]).to(torch.complex64)
            ''' eigenvalues for computing the linear combination of basis functions with wobbling '''
            lam = torch.stack([
                (1 - torch.cos(delta_ / 2)) * (torch.cos(delta_ / 2) + 2) / 6,
                (1 - torch.cos(delta_ / 2)) * (torch.cos(delta_ / 2) + 2) / 6, 
                ((torch.cos(delta_ / 2)**3 - 1) / (torch.cos(delta_ / 2) - 1)) / 3]).to(torch.complex64)
        else:
            ''' rotation matrix if several PSF is simulated'''
            R = torch.permute(torch.stack([
                torch.stack([(torch.sin(rho_)**2)*(1 - torch.cos(eta_)) + torch.cos(eta_), 
                             torch.sin(rho_) * torch.cos(rho_) * (torch.cos(eta_) - 1), 
                             torch.cos(rho_) * torch.sin(eta_)]),
                torch.stack([torch.sin(rho_) * torch.cos(rho_) * (torch.cos(eta_) - 1), 
                             (torch.cos(rho_)**2)*(1 - torch.cos(eta_)) + torch.cos(eta_), 
                             torch.sin(rho_) * torch.sin(eta_)]),
                torch.stack([-torch.cos(rho_) * torch.sin(eta_), 
                             -torch.sin(rho_) * torch.sin(eta_), 
                             torch.cos(eta_)])
            ]).to(torch.complex64), (2,0,1))
            ''' eigenvalues for computing the linear combination of basis functions with wobbling '''
            lam = torch.permute(torch.stack([
                (1 - torch.cos(delta_ / 2)) * (torch.cos(delta_ / 2) + 2) / 6,
                (1 - torch.cos(delta_ / 2)) * (torch.cos(delta_ / 2) + 2) / 6, 
                ((torch.cos(delta_ / 2)**3 - 1) / (torch.cos(delta_ / 2) - 1)) / 3]).to(torch.complex64), (1,0))
    else:
        ''' rotation matrix in numpy'''
        R = np.array([
            [(np.sin(rho)**2)*(1 - np.cos(eta)) + np.cos(eta), 
             np.sin(rho) * np.cos(rho) * (np.cos(eta) - 1), 
             np.cos(rho) * np.sin(eta)],
            [np.sin(rho) * np.cos(rho) * (np.cos(eta) - 1), 
             (np.cos(rho)**2)*(1 - np.cos(eta)) + np.cos(eta), 
             np.sin(rho) * np.sin(eta)],
            [-np.cos(rho) * np.sin(eta), 
             -np.sin(rho) * np.sin(eta), 
             np.cos(eta)]
        ])
        ''' eigenvalues for computing the linear combination of basis functions with wobbling '''
        lam = np.array([
            (1 - np.cos(delta / 2)) * (np.cos(delta / 2) + 2) / 6, 
            (1 - np.cos(delta / 2)) * (np.cos(delta / 2) + 2) / 6, 
            ((np.cos(delta / 2)**3 - 1) / (np.cos(delta / 2) - 1)) / 3
        ])
    
    ''' Linear combination to compute the PSF '''
    if isinstance(eta, torch.Tensor):
        if len(eta.shape)==0:
            ''' 1 PSF in torch '''
            psfx = torch.real(torch.einsum('a, auv -> uv', lam, torch.movedim(torch.diagonal(torch.einsum('ab, bcuv -> acuv', R.T, torch.einsum('abuv, bc -> acuv', M[0], R)), dim1=0, dim2=1), -1, 0)))
            psfy = torch.real(torch.einsum('a, auv -> uv', lam, torch.movedim(torch.diagonal(torch.einsum('ab, bcuv -> acuv', R.T, torch.einsum('abuv, bc -> acuv', M[1], R)), dim1=0, dim2=1), -1, 0)))
            psf = torch.stack([psfx, psfy], dim=0)
            
            psf = torch.clamp(torch.real(psf), min=0.)
            ''' Normalize the PSF tensor '''
            norm = torch.sum(psf)
            psf = (psf * (N_photons / norm)).to(torch.float32)
        else:
            if M.ndim==7:
                ''' several psf and several planes '''
                psf = torch.sum(torch.diagonal(torch.einsum('uab, uvwbdxy -> uvwadxy', torch.transpose(R, 1, 2), torch.einsum('uvwabxy,ubd->uvwadxy', M, R*lam.unsqueeze(1))), dim1=3, dim2=4), dim=5)
                psf = torch.clamp(torch.real(psf), min=0.)
                ''' Normalize the PSF tensor '''
                norm = torch.sum(psf, dim=list(range(1, psf.ndim)))
                if N_photons.ndim==0:
                    ''' if Psf have same number of photons '''
                    psf = torch.einsum('abcde, a -> abcde', psf*N_photons, 1/norm).to(torch.float32)
                else:
                    ''' if Psf have different number of photons '''
                    psf = torch.einsum('abcde, a -> abcde', torch.einsum('abcde, a -> abcde', psf, N_photons), 1/norm).to(torch.float32)
            else:
                ''' several psf but one plane '''
                psf = torch.sum(torch.diagonal(torch.einsum('uab, uvbdxy -> uvadxy', torch.transpose(R, 1, 2), torch.einsum('uvabxy,ubd->uvadxy', M, R*lam.unsqueeze(1))), dim1=2, dim2=3), dim=4)
              
                psf = torch.clamp(torch.real(psf), min=0.)
                ''' Normalize the PSF tensor '''
                norm = torch.sum(psf, dim=list(range(1, psf.ndim)))
                if N_photons.ndim==0:
                    psf = psf*N_photons/norm.view(-1, 1, 1, 1).to(torch.float32)
                else:
                    psf = torch.einsum('abcd, a -> abcd', psf, N_photons)/norm.view(-1, 1, 1, 1).to(torch.float32)
    else:
        psfx = (np.real(np.einsum('a, auv -> uv', lam, np.moveaxis(np.diagonal(np.einsum('ab, bcuv -> acuv', R.T, np.einsum('abuv, bc -> acuv', M[0], R)), axis1=0, axis2=1), -1, 0))))
        psfy = np.real(np.einsum('a, auv -> uv', lam, np.moveaxis(np.diagonal(np.einsum('ab, bcuv -> acuv', R.T, np.einsum('abuv, bc -> acuv', M[1], R)), axis1=0, axis2=1), -1, 0)))
        psf = np.array([psfx, psfy])
        psf[psf<0] = 0.
        ''' Normalize the PSF tensor '''
        norm = np.sum(psf)
        psf = psf * (N_photons / norm)
    return psf

def BFP_intensity(rho, eta, delta, M, N_photons=1000, device='cpu'):
    '''
    This is exactly like PSF (same linear combination), but the matrix M is computed without fft2
    This function is just to make clearer code when used but the way to use it is just to feed PSF with a different M matrix 
    to compute this special M matrix, use BFP_version = True
    '''
    if isinstance(eta, torch.Tensor):
        return PSF(rho, eta, delta, M.to(torch.complex64), N_photons=N_photons, device=device)
    else:
        return PSF(rho, eta, delta, M, N_photons=N_photons, device=device)

def noise(PSF, QE, EM, b, sigma_b, sigma_r, bias):
    '''
    given a computed PSF, adds noise according to a mixed Poisson Gaussian noise
    shot noise is taken in account with Poisson distribution and backgound by Gaussian noise with b mean and standard deviation sigma_b
    the camera has a conversion factor QE*EM
    Afterwards, we add a gaussin read noise (sigma_r)
    See "A convex 3D deconvolution algorithm for low photon count fluorescence imaging", Ikoma et al. (2018)
    the value is in number of photons
    '''
    if isinstance(PSF, torch.Tensor):
        device = PSF.device
        background = normal(0, sigma_b, PSF.shape)+b
        background[background<0] = 0.
        read = normal(0, sigma_r, PSF.shape)
        poiss = torch.poisson(PSF+torch.tensor(background, device=device))
        gamma_dist = torch.distributions.Gamma(concentration=poiss.clamp(min=1e-6)*QE, rate=EM)
        excess = gamma_dist.sample()
        noisy =  poiss + torch.tensor(read, device=device)*(1/(QE*EM)) + bias*(1/(QE*EM)) + excess*(1/(QE*EM))
        return noisy
    else:
        background = normal(b, sigma_b, PSF.shape)
        background[background<0] = 0.
        noisy = poisson(PSF+background) + normal(0, sigma_r, PSF.shape) + bias
        return noisy
