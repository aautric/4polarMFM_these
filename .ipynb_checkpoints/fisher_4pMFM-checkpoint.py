#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:48:18 2026

@author: amaury
"""

from simu_PSF_polarMFM import *

def fisher(M_matrix, RHO, ETA, DELTA, Np, BACKGROUND, device='cuda'):
    omega = 2*torch.pi*(1-torch.cos(DELTA))
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
                      torch.einsum('a,apbcd->apbcd', ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(2*torch.pi*torch.sin(DELTA)), 
                                   torch.einsum('apbcd,a->apbcd', Bxx[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxx_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Byy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyy_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bzz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dzz_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bxy[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxy_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Bxz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dxz_gamma) + 
                                   torch.einsum('apbcd,a->apbcd', Byz[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7], dyz_gamma))])
    to_return = torch.sum(torch.einsum('uvapbcd, apbcd -> uvapbcd', torch.einsum('uapbcd, vapbcd -> uvapbcd', CR, CR), 1/((psf[:,:,:,NN//2-7:NN//2+7,NN//2-7:NN//2+7]+BACKGROUND))), dim=(3,4,5,6))
    return to_return #torch.sqrt(to_return.real**2+to_return.imag**2+1e-12)  

def fisher_ratiometric(M_matrix, RHO, ETA, DELTA, Np, BACKGROUND, device='cuda'):
    omega = 2*torch.pi*(1-torch.cos(DELTA))
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

    Bxx = torch.real(torch.sum(M_matrix[:,:,:,0,0], dim=(-2,-1)))
    Byy = torch.real(torch.sum(M_matrix[:,:,:,1,1], dim=(-2,-1)))
    Bzz = torch.real(torch.sum(M_matrix[:,:,:,2,2], dim=(-2,-1)))
    Bxy = torch.real(torch.sum(M_matrix[:,:,:,0,1]+M_matrix[:,:,:,1,0], dim=(-2,-1)))
    Bxz = torch.real(torch.sum(M_matrix[:,:,:,0,2]+M_matrix[:,:,:,2,0], dim=(-2,-1)))
    Byz = torch.real(torch.sum(M_matrix[:,:,:,1,2]+M_matrix[:,:,:,2,1], dim=(-2,-1)))
    psf, norm = PSF(rho=RHO*180/torch.pi, eta=ETA*180/torch.pi, delta=DELTA*180/torch.pi, M=M_matrix, N_photons=Np, device=device, return_norm=True)
    NN = psf.shape[3]
    psf = torch.sum(((psf[:,:,:,NN//2-3:NN//2+3,NN//2-3:NN//2+3]+BACKGROUND)), dim=(-2,-1))
    
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
                      torch.einsum('a,apb->apb', ((omega/(4*torch.pi**2))-3/(4*torch.pi))*(2*torch.pi*torch.sin(DELTA)), 
                                   torch.einsum('apb,a->apb', Bxx[:,:,:], dxx_gamma) + 
                                   torch.einsum('apb,a->apb', Byy[:,:,:], dyy_gamma) + 
                                   torch.einsum('apb,a->apb', Bzz[:,:,:], dzz_gamma) + 
                                   torch.einsum('apb,a->apb', Bxy[:,:,:], dxy_gamma) + 
                                   torch.einsum('apb,a->apb', Bxz[:,:,:], dxz_gamma) + 
                                   torch.einsum('apb,a->apb', Byz[:,:,:], dyz_gamma))])
    to_return = torch.sum(torch.einsum('uvapb, apb -> uvapb', torch.einsum('uapb, vapb -> uvapb', CR, CR), 1/psf), dim=(3,4))
    return to_return #torch.sqrt(to_return.real**2+to_return.imag**2+1e-12)  