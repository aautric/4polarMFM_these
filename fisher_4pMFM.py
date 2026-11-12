#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:48:18 2026

@author: amaury

Used by MFM_fisher.ipynb and MFM_fisher_claude.py. 2026-10-28 (Claude): d gamma/d delta fixed (pi sin(delta/2)),
derivative of the normalisation of PSF() added, noise gain / dark_var, window and fixed_eta options of
fisher_ratiometric (2026-11-10: merged here from fisher_4pMFM_claude.py, the defaults keep the old calls valid).
"""

from simu_PSF_polarMFM import *

def fisher(M_matrix, RHO, ETA, DELTA, Np, BACKGROUND, device='cuda', gain=1., dark_var=0., window=13):
    # noise of each pixel: variance = gain x (psf + BACKGROUND) + dark_var (photons; gain = 1, dark_var = 0: Poisson only,
    # gain = 1 + 1/QE: Poisson + EM excess, gain and dark_var of the noise model of SGD_jax.py for real data)
    # Purpose: Fisher information matrix of (rho, eta, delta) for each PSF, shape (3, 3, NPSF), angles in RADIANS
    # (RHO, ETA, DELTA in rad; CRB = sqrt(diag(F^-1)) in rad). x, y, z, N and the background are treated as KNOWN
    # (not in the matrix): the bound is optimistic compared with a fit of all the parameters.
    # F_uv = sum over the pixels of ds/du ds/dv / var, var = gain (s + b) + dark_var: exact for Poisson (gain 1,
    # dark_var 0); otherwise the Gaussian approximation without the term of the variance's own derivative.
    # Model (same as PSF() of simu_PSF_polarMFM.py): unnormalised image I = sum_ij Re(M_ij) m_ij with the second moment
    # m = (1 - gamma)/3 Id + gamma u u^T, u = (sin eta cos rho, sin eta sin rho, cos eta), delta = FULL cone angle,
    # gamma = cos(delta/2)(1 + cos(delta/2))/2 (= 1 - 3 Omega/(4 pi) + Omega^2/(8 pi^2), Omega = 2 pi (1 - cos(delta/2)),
    # identical to the eigenvalues lam of PSF()); s = N I/S, S = sum of I over the whole computed image.
    # Pixels: window x window centred on the centre pixel NN//2 of each channel; default 13 = the 13 x 13 patch of the
    # fit (2026-11-12, review item 16: was NN//2-7 ... NN//2+6 = 14 x 14, CRB 1-4 % too low; window=14 gives the old
    # pixels back). Checked 2026-11-12 against finite differences of PSF() (6 PSF, eta 10-160, background 0 and 25):
    # Fisher matrix within 2e-4, CRB identical to 0.01 deg.
    omega = 2*torch.pi*(1-torch.cos(DELTA/2))
    gamma = 1 - (3*omega)/(4*torch.pi) + omega**2/(8*torch.pi**2)
    
    # d m_ij/d eta and d m_ij/d rho = gamma d(u_i u_j) (the constant (1 - gamma)/3 Id does not depend on the angles)
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

    # d m_ij/d gamma = u_i u_j - delta_ij/3 (then x d gamma/d delta)
    dxx_gamma = (torch.sin(ETA)*torch.cos(RHO))**2 -1/3
    dyy_gamma = (torch.sin(ETA)*torch.sin(RHO))**2 -1/3
    dzz_gamma = (torch.cos(ETA))**2 -1/3
    dxy_gamma = (torch.sin(ETA)**2)*torch.cos(RHO)*torch.sin(RHO)
    dxz_gamma = torch.sin(ETA)*torch.cos(ETA)*torch.cos(RHO)
    dyz_gamma = torch.sin(ETA)*torch.cos(ETA)*torch.sin(RHO)

    # basis images of each term m_ij (off-diagonal terms appear twice in sum_ij: M_ij + M_ji)
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
    w = slice(NN//2 - window//2, NN//2 - window//2 + window) # 13: NN//2-6 ... NN//2+6 (centred); 14: NN//2-7 ... NN//2+6
    CR = torch.stack([torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,w,w], dxx_rho) + 
                      torch.einsum('apbcd,a->apbcd', Byy[:,:,:,w,w], dyy_rho) + 
                      torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,w,w], dzz_rho) + 
                      torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,w,w], dxy_rho) + 
                      torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,w,w], dxz_rho) + 
                      torch.einsum('apbcd,a->apbcd', Byz[:,:,:,w,w], dyz_rho), 
                      torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,w,w], dxx_eta) + 
                      torch.einsum('apbcd,a->apbcd', Byy[:,:,:,w,w], dyy_eta) + 
                      torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,w,w], dzz_eta) + 
                      torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,w,w], dxy_eta) + 
                      torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,w,w], dxz_eta) + 
                      torch.einsum('apbcd,a->apbcd', Byz[:,:,:,w,w], dyz_eta), 
                      # d gamma/d delta = (d gamma/d Omega)(d Omega/d delta), Omega = 2 pi (1 - cos(delta/2)) -> d Omega/d delta = pi sin(delta/2)
                      # (was 2 pi sin(delta), the derivative for a half-angle delta: too large by 4 cos(delta/2))
                      torch.einsum('a,apbcd->apbcd', ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(torch.pi*torch.sin(DELTA/2)), 
                                   torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,w,w], dxx_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Byy[:,:,:,w,w], dyy_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,w,w], dzz_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,w,w], dxy_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,w,w], dxz_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Byz[:,:,:,w,w], dyz_gamma))])
    # PSF() normalises to N photons, s = N I/S with S = norm depending on the orientation:
    # d s/d theta = (N/S) d I/d theta - (s/S) d S/d theta (CR above is the first term)
    CR = CR - torch.einsum('apbcd,ua->uapbcd', psf[:,:,:,w,w], dS/norm)
    # F_uv = sum over planes, polarisations and pixels of the window of CR_u CR_v / var
    to_return = torch.sum(torch.einsum('uvapbcd, apbcd -> uvapbcd', torch.einsum('uapbcd, vapbcd -> uvapbcd', CR, CR), 1/(gain*(psf[:,:,:,w,w]+BACKGROUND)+dark_var)), dim=(3,4,5,6))
    return to_return #torch.sqrt(to_return.real**2+to_return.imag**2+1e-12)  

def fisher_ratiometric(M_matrix, RHO, ETA, DELTA, Np, BACKGROUND, device='cuda', window=11, fixed_eta=False, gain=1., dark_var=0.):
    # Fisher matrix of the intensities of the 6 channels (3 planes x 2 polarisations), each summed over a square window
    # window: size (px) of the window centred on each channel, None = whole image (all the photons of the molecule)
    # fixed_eta: eta known (ratiometric method: rho and delta are estimated), Fisher matrix of (rho, delta) only (2 x 2).
    #   With the whole image, eta and delta cannot be separated (singular 3 x 3 matrix): use window=None with
    #   fixed_eta=True for a pure polarisation-ratio bound
    # gain, dark_var: noise of each pixel, variance = gain x (psf + BACKGROUND) + dark_var, as in fisher
    # Purpose: bound of a RATIOMETRIC method (what can be learnt from the 6 channel intensities only, no image shape),
    # compared with fisher() (whole image) in MFM_fisher_claude.py; not used for the CRB of the simulations.
    # Same model and derivatives as fisher(); each channel is one Poisson count = sum over its window, variance
    # gain x (signal + n_pix b) + n_pix dark_var. Checked 2026-11-12 against finite differences of PSF() (window 11
    # and None, b 0 and 25, fixed_eta False/True): within 6e-4.
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