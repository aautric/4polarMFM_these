#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:48:18 2026

@author: amaury

Copy of fisher_4pMFM.py (kept for MFM_fisher.ipynb) used by MFM_fisher_claude.py. 2026-10-28: d gamma/d delta fixed.
"""

from simu_PSF_polarMFM import *

def fisher(M_matrix, RHO, ETA, DELTA, Np, BACKGROUND, device='cuda', gain=1., dark_var=0.):
    # noise of each pixel: variance = gain x (psf + BACKGROUND) + dark_var (photons; gain = 1, dark_var = 0: Poisson only,
    # gain = 1 + 1/QE: Poisson + EM excess, gain and dark_var of the noise model of SGD_jax.py for real data)
    omega = 2*torch.pi*(1-torch.cos(DELTA/2))
    gamma = 1 - (3*omega)/(4*torch.pi) + omega**2/(8*torch.pi**2)
    
    dxx_eta = 2*gamma*(torch.cos(RHO)**2)*torch.sin(ETA)*torch.cos(ETA)
    dyy_eta = 2*gamma*(torch.sin(RHO)**2)*torch.sin(ETA)*torch.cos(ETA)
    dzz_eta = -2*gamma*torch.sin(ETA)*torch.cos(ETA)
    dxy_eta = 2*gamma*(torch.sin(RHO)*torch.cos(RHO))*torch.sin(ETA)*torch.cos(ETA)
    dxz_eta = gamma*torch.cos(RHO)*(torch.cos(ETA)**2-torch.sin(ETA)**2)
    dyz_eta = gamma*torch.sin(RHO)*(torch.cos(ETA)**2-torch.sin(ETA)**2)

    dxx_rho = -2*gamma*(torch.sin(ETA)**2)*torch.sin(RHO)*torch.cos(RHO)
    dyy_rho = 2*gamma*(torch.sin(ETA)**2)*torch.sin(RHO)*torch.cos(RHO)
    dzz_rho = 0*ETA
    dxy_rho = gamma*(torch.sin(ETA)**2)*(torch.cos(RHO)**2-torch.sin(RHO)**2)
    dxz_rho = -gamma*torch.cos(ETA)*torch.sin(ETA)*torch.sin(RHO)
    dyz_rho = gamma*torch.cos(ETA)*torch.sin(ETA)*torch.cos(RHO)

    dxx_gamma = (torch.sin(ETA)*torch.cos(RHO))**2 -1/3
    dyy_gamma = (torch.sin(ETA)*torch.sin(RHO))**2 -1/3
    dzz_gamma = (torch.cos(ETA))**2 -1/3
    dxy_gamma = (torch.sin(ETA)**2)*torch.cos(RHO)*torch.sin(RHO)
    dxz_gamma = torch.sin(ETA)*torch.cos(ETA)*torch.cos(RHO)
    dyz_gamma = torch.sin(ETA)*torch.cos(ETA)*torch.sin(RHO)

    Bxx = torch.real(M_matrix[:,:,:,0,0])
    Byy = torch.real(M_matrix[:,:,:,1,1])
    Bzz = torch.real(M_matrix[:,:,:,2,2])
    Bxy = torch.real(M_matrix[:,:,:,0,1]+M_matrix[:,:,:,1,0])
    Bxz = torch.real(M_matrix[:,:,:,0,2]+M_matrix[:,:,:,2,0])
    Byz = torch.real(M_matrix[:,:,:,1,2]+M_matrix[:,:,:,2,1])
    # whole image (all planes, polarisations, pixels) of each basis term, for the derivative of the total S = norm
    Sxx, Syy, Szz, Sxy, Sxz, Syz = [torch.sum(b, dim=(1,2,3,4)) for b in (Bxx, Byy, Bzz, Bxy, Bxz, Byz)]
    dgamma_ddelta = ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(torch.pi*torch.sin(DELTA/2))
    dS = torch.stack([Sxx*dxx_rho + Syy*dyy_rho + Szz*dzz_rho + Sxy*dxy_rho + Sxz*dxz_rho + Syz*dyz_rho,
                      Sxx*dxx_eta + Syy*dyy_eta + Szz*dzz_eta + Sxy*dxy_eta + Sxz*dxz_eta + Syz*dyz_eta,
                      dgamma_ddelta*(Sxx*dxx_gamma + Syy*dyy_gamma + Szz*dzz_gamma + Sxy*dxy_gamma + Sxz*dxz_gamma + Syz*dyz_gamma)])
    psf, norm = PSF(rho=RHO*180/torch.pi, eta=ETA*180/torch.pi, delta=DELTA*180/torch.pi, M=M_matrix, N_photons=Np, device=device, return_norm=True)
    #plt.imshow(psf[0,0,0].cpu().detach()+BACKGROUND.cpu().detach())
    #plt.show()
    #psf = psf.detach()
    #norm=norm.detach()
    Bxx, Byy, Bzz, Bxy, Bxz, Byz = torch.einsum('apbcd,a->apbcd', Bxx, (Np/norm)), torch.einsum('apbcd,a->apbcd', Byy, (Np/norm)), torch.einsum('apbcd,a->apbcd', Bzz, (Np/norm)), torch.einsum('apbcd,a->apbcd', Bxy, (Np/norm)), torch.einsum('apbcd,a->apbcd', Bxz, (Np/norm)), torch.einsum('apbcd,a->apbcd', Byz, (Np/norm))
    
    NN = psf.shape[3]
    CR = torch.stack([torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxx_rho) + 
                      torch.einsum('apbcd,a->apbcd', Byy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyy_rho) + 
                      torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dzz_rho) + 
                      torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxy_rho) + 
                      torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxz_rho) + 
                      torch.einsum('apbcd,a->apbcd', Byz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyz_rho), 
                      torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxx_eta) + 
                      torch.einsum('apbcd,a->apbcd', Byy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyy_eta) + 
                      torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dzz_eta) + 
                      torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxy_eta) + 
                      torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxz_eta) + 
                      torch.einsum('apbcd,a->apbcd', Byz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyz_eta), 
                      # d gamma/d delta = (d gamma/d Omega)(d Omega/d delta), Omega = 2 pi (1 - cos(delta/2)) -> d Omega/d delta = pi sin(delta/2)
                      # (was 2 pi sin(delta), the derivative for a half-angle delta: too large by 4 cos(delta/2))
                      torch.einsum('a,apbcd->apbcd', ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(torch.pi*torch.sin(DELTA/2)), 
                                   torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxx_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Byy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyy_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dzz_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxy_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxz_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Byz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyz_gamma))])
    # PSF() normalises to N photons, s = N I/S with S = norm depending on the orientation:
    # d s/d theta = (N/S) d I/d theta - (s/S) d S/d theta (CR above is the first term)
    CR = CR - torch.einsum('apbcd,ua->uapbcd', psf[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dS/norm)
    to_return = torch.sum(torch.einsum('uvapbcd, apbcd -> uvapbcd', torch.einsum('uapbcd, vapbcd -> uvapbcd', CR, CR), 1/(gain*(psf[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7]+BACKGROUND)+dark_var)), dim=(3,4,5,6))
    return to_return #torch.sqrt(to_return.real**2+to_return.imag**2+1e-12)  

def fisher_ratiometric(M_matrix, RHO, ETA, DELTA, Np, BACKGROUND, device='cuda', window=11, fixed_eta=False, gain=1., dark_var=0.):
    # Fisher matrix of the intensities of the 6 channels (3 planes x 2 polarisations), each summed over a square window
    # window: size (px) of the window centred on each channel, None = whole image (all the photons of the molecule)
    # fixed_eta: eta known (ratiometric method: rho and delta are estimated), Fisher matrix of (rho, delta) only (2 x 2).
    #   With the whole image, eta and delta cannot be separated (singular 3 x 3 matrix): use window=None with
    #   fixed_eta=True for a pure polarisation-ratio bound
    # gain, dark_var: noise of each pixel, variance = gain x (psf + BACKGROUND) + dark_var, as in fisher
    omega = 2*torch.pi*(1-torch.cos(DELTA/2))
    gamma = 1 - (3*omega)/(4*torch.pi) + omega**2/(8*torch.pi**2)
    
    dxx_eta = 2*gamma*(torch.cos(RHO)**2)*torch.sin(ETA)*torch.cos(ETA)
    dyy_eta = 2*gamma*(torch.sin(RHO)**2)*torch.sin(ETA)*torch.cos(ETA)
    dzz_eta = -2*gamma*torch.sin(ETA)*torch.cos(ETA)
    dxy_eta = 2*gamma*(torch.sin(RHO)*torch.cos(RHO))*torch.sin(ETA)*torch.cos(ETA)
    dxz_eta = gamma*torch.cos(RHO)*(torch.cos(ETA)**2-torch.sin(ETA)**2)
    dyz_eta = gamma*torch.sin(RHO)*(torch.cos(ETA)**2-torch.sin(ETA)**2)

    dxx_rho = -2*gamma*(torch.sin(ETA)**2)*torch.sin(RHO)*torch.cos(RHO)
    dyy_rho = 2*gamma*(torch.sin(ETA)**2)*torch.sin(RHO)*torch.cos(RHO)
    dzz_rho = 0*ETA
    dxy_rho = gamma*(torch.sin(ETA)**2)*(torch.cos(RHO)**2-torch.sin(RHO)**2)
    dxz_rho = -gamma*torch.cos(ETA)*torch.sin(ETA)*torch.sin(RHO)
    dyz_rho = gamma*torch.cos(ETA)*torch.sin(ETA)*torch.cos(RHO)

    dxx_gamma = (torch.sin(ETA)*torch.cos(RHO))**2 -1/3
    dyy_gamma = (torch.sin(ETA)*torch.sin(RHO))**2 -1/3
    dzz_gamma = (torch.cos(ETA))**2 -1/3
    dxy_gamma = (torch.sin(ETA)**2)*torch.cos(RHO)*torch.sin(RHO)
    dxz_gamma = torch.sin(ETA)*torch.cos(ETA)*torch.cos(RHO)
    dyz_gamma = torch.sin(ETA)*torch.cos(ETA)*torch.sin(RHO)

    # intensity of each channel summed over the SAME window for the derivative and the variance (was: whole image for
    # the derivative, 6 x 6 px for the variance)
    NN = M_matrix.shape[-1]
    w = slice(None) if window is None else slice(NN//2-window//2, NN//2-window//2+window)
    B_full = [torch.real(M_matrix[:,:,:,0,0]), torch.real(M_matrix[:,:,:,1,1]), torch.real(M_matrix[:,:,:,2,2]),
              torch.real(M_matrix[:,:,:,0,1]+M_matrix[:,:,:,1,0]), torch.real(M_matrix[:,:,:,0,2]+M_matrix[:,:,:,2,0]),
              torch.real(M_matrix[:,:,:,1,2]+M_matrix[:,:,:,2,1])]
    Bxx, Byy, Bzz, Bxy, Bxz, Byz = [torch.sum(b[:,:,:,w,w], dim=(-2,-1)) for b in B_full]
    # whole image of each basis term, for the derivative of the total S = norm (normalisation of PSF())
    Sxx, Syy, Szz, Sxy, Sxz, Syz = [torch.sum(b, dim=(1,2,3,4)) for b in B_full]
    dgamma_ddelta = ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(torch.pi*torch.sin(DELTA/2))
    dS = torch.stack([Sxx*dxx_rho + Syy*dyy_rho + Szz*dzz_rho + Sxy*dxy_rho + Sxz*dxz_rho + Syz*dyz_rho,
                      Sxx*dxx_eta + Syy*dyy_eta + Szz*dzz_eta + Sxy*dxy_eta + Sxz*dxz_eta + Syz*dyz_eta,
                      dgamma_ddelta*(Sxx*dxx_gamma + Syy*dyy_gamma + Szz*dzz_gamma + Sxy*dxy_gamma + Sxz*dxz_gamma + Syz*dyz_gamma)])
    psf, norm = PSF(rho=RHO*180/torch.pi, eta=ETA*180/torch.pi, delta=DELTA*180/torch.pi, M=M_matrix, N_photons=Np, device=device, return_norm=True)
    signal = torch.sum(psf[:,:,:,w,w], dim=(-2,-1)) # photons of the molecule in the window of each channel
    n_pix = psf[0,0,0,w,w].numel()
    psf = gain*torch.sum(((psf[:,:,:,w,w]+BACKGROUND)), dim=(-2,-1)) + n_pix*dark_var # variance of the window sum
    
    #psf = psf.detach()
    #norm=norm.detach()
    Bxx, Byy, Bzz, Bxy, Bxz, Byz = torch.einsum('apb,a->apb', Bxx, (Np/norm)), torch.einsum('apb,a->apb', Byy, (Np/norm)), torch.einsum('apb,a->apb', Bzz, (Np/norm)), torch.einsum('apb,a->apb', Bxy, (Np/norm)), torch.einsum('apb,a->apb', Bxz, (Np/norm)), torch.einsum('apb,a->apb', Byz, (Np/norm))
    
    CR = torch.stack([torch.einsum('apb,a->apb', Bxx[:,:,:], dxx_rho) + 
                      torch.einsum('apb,a->apb', Byy[:,:,:], dyy_rho) + 
                      torch.einsum('apb,a->apb', Bzz[:,:,:], dzz_rho) + 
                      torch.einsum('apb,a->apb', Bxy[:,:,:], dxy_rho) + 
                      torch.einsum('apb,a->apb', Bxz[:,:,:], dxz_rho) + 
                      torch.einsum('apb,a->apb', Byz[:,:,:], dyz_rho), 
                      torch.einsum('apb,a->apb', Bxx[:,:,:], dxx_eta) + 
                      torch.einsum('apb,a->apb', Byy[:,:,:], dyy_eta) + 
                      torch.einsum('apb,a->apb', Bzz[:,:,:], dzz_eta) + 
                      torch.einsum('apb,a->apb', Bxy[:,:,:], dxy_eta) + 
                      torch.einsum('apb,a->apb', Bxz[:,:,:], dxz_eta) + 
                      torch.einsum('apb,a->apb', Byz[:,:,:], dyz_eta), 
                      # d Omega/d delta = pi sin(delta/2), as in fisher
                      torch.einsum('a,apb->apb', ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(torch.pi*torch.sin(DELTA/2)), 
                                   torch.einsum('apb,a->apb', Bxx[:,:,:], dxx_gamma) + 
                                   torch.einsum('apb,a->apb', Byy[:,:,:], dyy_gamma) + 
                                   torch.einsum('apb,a->apb', Bzz[:,:,:], dzz_gamma) + 
                                   torch.einsum('apb,a->apb', Bxy[:,:,:], dxy_gamma) + 
                                   torch.einsum('apb,a->apb', Bxz[:,:,:], dxz_gamma) + 
                                   torch.einsum('apb,a->apb', Byz[:,:,:], dyz_gamma))])
    # normalisation of PSF(): d s/d theta = (N/S) d I/d theta - (s/S) d S/d theta
    CR = CR - torch.einsum('apb,ua->uapb', signal, dS/norm)
    if fixed_eta:
        CR = CR[[0, 2]] # rho, delta
    to_return = torch.sum(torch.einsum('uvapb, apb -> uvapb', torch.einsum('uapb, vapb -> uvapb', CR, CR), 1/psf), dim=(3,4))
    return to_return #torch.sqrt(to_return.real**2+to_return.imag**2+1e-12)  