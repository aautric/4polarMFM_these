# -*- coding: utf-8 -*-
"""
Simple microscope model
see PSF_demo.ipynb for explanations
amaury.autric@curie.fr
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import random
from zernike import RZern
import numpy as np
import matplotlib.pyplot as plt

n1 = 1.52 # oil and sample indices
n2 = 1.33 # water index
h = 6.626*10**(-34) # Planck constant
c = 299792458 # speed of light

# everything in micrometers 
    
def vectorial_BFP_perfect_focus_jax(
    N: int,
    NA: float = 1.4,
    mag: float = 100.0,
    lambd_nm: float = 617.0,
    f_tube_mm: float = 200.0,
    J_dichroic=jnp.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]),
    SAF=False,
    dtype=jnp.float32,
):
    """
    JAX version of vectorial_BFP_perfect_focus.
    Not jitted, this function defines all the sizes of folowing objects and is thus very important
    - N : grid size (static, int)
    - NA, mag, lambd_nm (nm), f_tube_mm (mm), refractive indices n1,n2
    - returns: x, y, th1, phi, (Ex0,Ex1,Ex2), (Ey0,Ey1,Ey2), r, r_cut, k, f_o
    """
    print("Compiling vectorial_BFP_perfect_focus_jax")
    # convert units
    lambd = jnp.array(1e-3 * lambd_nm, dtype=dtype)      # nm -> um
    f_tube = jnp.array(1000.0 * f_tube_mm, dtype=dtype)  # mm -> um
    f_o = f_tube / mag
    k = 2.0 * n1 * jnp.pi / lambd

    # spatial cutoff
    if SAF:
        r_cut_SAF = n2/n1
        r_cut = NA/n1
    else:
        r_cut = jnp.minimum(NA / n1, n2 / n1)

    # meshgrid
    coords = jnp.linspace(-r_cut, r_cut, N, dtype=dtype)
    x, y = jnp.meshgrid(coords, coords, indexing="xy")
    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)

    # angles
    th1 = jnp.where(r < r_cut, jnp.arcsin(r), 0.0)
    if SAF:
        th2 = jnp.where(r < r_cut_SAF, jnp.arcsin((n1 / n2) * r), 0.0)
    else:
        th2 = jnp.where(r < r_cut, jnp.arcsin((n1 / n2) * r), 0.0)

    # transmission coefficients
    cos_th1 = jnp.cos(th1)
    cos_th2 = jnp.cos(th2)
    
    if SAF:
        mask_saf = (r < r_cut) & (r > r_cut_SAF)
        new_vals = 1j * jnp.sqrt((jnp.sin(th1) * (n1 / n2))**2 - 1)
        cos_th2 = jnp.where(mask_saf, new_vals, cos_th2)    
    Ts = jnp.where(r < r_cut, (2.0 * n2 * cos_th2) / (n2 * cos_th2 + n1 * cos_th1), 0.0)
    Tp = jnp.where(r < r_cut, (2.0 * n2 * cos_th2) / (n2 * cos_th1 + n1 * cos_th2), 0.0)

    sqrt_cos_th1 = jnp.sqrt(jnp.clip(cos_th1, min=1e-12))  # stable sqrt

    # components
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    sin2phi = jnp.sin(2.0 * phi)

    # Ex
    Ex0 = ((n1 / n2) * ((cos_th1 / cos_th2) * Ts * (sin_phi**2) + Tp * (cos_phi**2) * cos_th1) / sqrt_cos_th1)
    Ex1 = (-((n1 * sin2phi) / (2.0 * n2)) * ((cos_th1 * Ts) / cos_th2 - Tp * cos_th1) / sqrt_cos_th1)
    Ex2 = (-((n1 / n2) ** 2) * (cos_th1 / cos_th2) * Tp * cos_phi * jnp.sin(th1) / sqrt_cos_th1)

    # Ey
    Ey0 = (-0.5 * sin2phi * (n1 / n2) * ((cos_th1 / cos_th2) * Ts - Tp * cos_th1) / sqrt_cos_th1)
    Ey1 = ((n1 / n2) * ((cos_th1 / cos_th2) * Ts * (cos_phi**2) + Tp * cos_th1 * (sin_phi**2)) / sqrt_cos_th1)
    Ey2 = (-((n1 / n2) ** 2) * (cos_th1 / cos_th2) * Tp * sin_phi * jnp.sin(th1) / sqrt_cos_th1)

    # mask outside r_cut
    mask = (r < r_cut).astype(dtype)
    Ex0 *= mask
    Ex1 *= mask
    Ex2 *= mask
    Ey0 *= mask
    Ey1 *= mask
    Ey2 *= mask
    
    # J_dichroic: (planes, 2, 2), E: (H, W)
    # Stack E fields into (2, H, W) then apply Jones matrix
    E0 = jnp.stack([Ex0, Ey0])  # (2, H, W)
    E1 = jnp.stack([Ex1, Ey1])
    E2 = jnp.stack([Ex2, Ey2])
    
    # einsum: for each plane, apply 2x2 Jones matrix to 2xHxW field
    E0_ = jnp.einsum('pij,jhw->pihw', J_dichroic, E0)  # (planes, 2, H, W)
    E1_ = jnp.einsum('pij,jhw->pihw', J_dichroic, E1)
    E2_ = jnp.einsum('pij,jhw->pihw', J_dichroic, E2)
    
    # unpack
    Ex0_, Ey0_ = E0_[:,0], E0_[:,1]
    Ex1_, Ey1_ = E1_[:,0], E1_[:,1]
    Ex2_, Ey2_ = E2_[:,0], E2_[:,1]
    
    if SAF:
        return x, y, th1, phi, (Ex0_, Ex1_, Ex2_), (Ey0_, Ey1_, Ey2_), r, r_cut, r_cut_SAF, k, f_o, cos_th2
    else:
        return x, y, th1, phi, (Ex0_, Ex1_, Ex2_), (Ey0_, Ey1_, Ey2_), r, r_cut, k, f_o

def psi_lat_jax(x, y, theta, phi, lambd=0.617):
    """
    JAX version of psi_lat.
    Works for:
        - scalar x,y
        - vectors x,y
        - 2D grids x,y
        - theta, phi arrays (broadcasting)
    """
    print("Compiling psi_lat_jax")
    phase = jnp.einsum('bc,abc->abc', jnp.sin(theta), (jnp.einsum('a,bc->abc', x, jnp.cos(phi)) +jnp.einsum('a,bc->abc', y, jnp.sin(phi))))
    return phase * (2*jnp.pi*n1) / lambd

def psi_z_jax(theta, z, lambd=0.617, cos_th2=None):
    print("Compiling psi_z_jax")
    if cos_th2 is None:
        sqrt_term = jnp.sqrt(1 - (n1 * jnp.sin(theta) / n2)**2)
    else:
        sqrt_term = cos_th2
    #jax.debug.print("z nan: {}", jnp.isnan(z).any())
    factor = 2 * jnp.pi * n2 / lambd
    out = jnp.einsum('a, bc -> abc', z, factor*sqrt_term)
    return out

def psi_f_jax(theta, d, lambd=0.617):
    """
    JAX version of psi_f (Yan et al. corrected)
    Computes the phase term along the focal region for vectorial PSF simulations.
    """
    print("Compiling psi_f_jax")
    # Refractive index depending on sign of d
    n = n1 + (n2 - n1) * (1 + jnp.sign(d)) / 2
    return jnp.einsum('a, bc -> abc', d*n , 2 * jnp.pi * jnp.cos(theta) / lambd)

psi_f_jit = jax.jit(psi_f_jax)
psi_z_jit = jax.jit(psi_z_jax)
psi_lat_jit = jax.jit(psi_lat_jax)

def generate_zernike_base_jax(r_cut, N, zernike_order=4, skip_indices = {0, 1, 2, 3}):
    print("Compiling generate_zernike_base_jax")
    cart = RZern(zernike_order)       
    ddx = np.linspace(-r_cut, r_cut, N)
    ddy = np.linspace(-r_cut, r_cut, N)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    
    zernike_base = np.zeros(((zernike_order+1)*(zernike_order+2)//2, xv.shape[0], xv.shape[1]))
    zer = np.zeros(cart.nk)
    for index in range(cart.nk):
        if index not in skip_indices:
            zer[index]=1.
            zernike_base[index] = cart.eval_grid(zer, matrix=True)
            zernike_base[index][np.isnan(zernike_base[index])] = 0.
            zernike_base[index][(xv**2+yv**2)>r_cut**2]=0.
            zer[index]=0.
    return jnp.array(zernike_base)

def padding_jax(r, r_cut, k, f_o,  N=80, l_pixel=16, NA=1.4, mag=100, lambd=617, 
              f_tube=200, MAG=200/150, device='cpu'):
    print("Compiling padding_jax")
    lambd = 10**(-3)*lambd # conversion nm to micrometers
    f_tube = f_tube*1000 # tube length focal. everything is in micrometers
    # --- Padding and frequency grid ---
    Dx = 2 * r_cut * f_o / N
    Npadding = int((2 * jnp.pi * MAG * f_tube) / (k * l_pixel * Dx)) - N
    Npadding += Npadding % 2
    freq = jnp.fft.fftshift(jnp.fft.fftfreq(N + Npadding, Dx)) * 2 * jnp.pi * f_tube / k * MAG
    u, v = jnp.meshgrid(freq, freq)
    return u, v, Npadding

#@partial(jax.jit, static_argnums=(1,))
def pad_jax(a, n):
    """Traceable JAX padding."""
    print("Compiling pad_jax")
    if a.ndim == 2:
        padded = jnp.zeros((a.shape[0] + n, a.shape[1] + n), dtype=a.dtype)
        padded = padded.at[n//2:-n//2, n//2:-n//2].set(a)
    elif a.ndim == 3:
        padded = jnp.zeros((a.shape[0], a.shape[1] + n, a.shape[2] + n), dtype=a.dtype)
        padded = padded.at[:, n//2:-n//2, n//2:-n//2].set(a)
    elif a.ndim == 4:
        padded = jnp.zeros((a.shape[0], a.shape[1], a.shape[2] + n, a.shape[3] + n), dtype=a.dtype)
        padded = padded.at[:,:, n//2:-n//2, n//2:-n//2].set(a)
    else:
        raise ValueError("Unsupported number of dimensions for padding.")
    return padded

# Npadding is used as a global variable

def compute_M_jax(xp, yp, zp, d, x, y, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2, u, v,
                  zernike_base=None, zernike_coefs_x=None, zernike_coefs_y=None, 
                  phase_maskx=None, phase_masky=None, lambd=617, f_tube=200,
              second_plane=jnp.array([0.35, 0, -0.35]), polar_projections=jnp.array([0., 45., 0.]), BFP_version=False, 
              SAF=False, cos_th2=None):
    """
    JAX version of compute_M for multiple PSFs and multiple planes, fully JIT-compatible.
    Returns u, v coordinates and M matrix of shape (N_psf, K_plane, 2, 3, 3, N_pix, N_pix)
    """
    print("Compiling compute_M_jax")
    #jax.debug.print("z: {}", zp)    
    #jax.debug.print("d: {}", jnp.mean(d))
    lambd = 10**(-3)*lambd # conversion nm to micrometers
    f_tube = f_tube*1000 # tube length focal. everything is in micrometers
    # --- Phase for all planes (vectorized with vmap) ---
    def phase_per_plane(dp):
        return jnp.exp(1j * (
            psi_f_jit(th1, d + dp, lambd) +
            psi_z_jit(th1, zp, lambd, cos_th2) +
            psi_lat_jit(xp, yp, th1, phi, lambd)
        ))#
    
    # dim 3, N, Npix, Npix
    phase = jax.vmap(phase_per_plane)(jnp.array(second_plane))  # 3 planes

    # --- Zernike masks --- dim 15, Npix, Npix
    '''aberrations computations'''
    if zernike_base is not None:
        zernike_mask_x = jnp.exp(1j * jnp.einsum('pa,auv->puv', zernike_coefs_x, zernike_base))
        zernike_mask_y = jnp.exp(1j * jnp.einsum('pa,auv->puv', zernike_coefs_y, zernike_base))
    else:
        zernike_mask_x = jnp.ones((Ex0.shape[-1], Ex0.shape[-1]))
        zernike_mask_y = jnp.ones((Ex0.shape[-1], Ex0.shape[-1]))
    
    # --- total phase mask --- dim N, 3, uv for each polar
    total_phase_x = jnp.einsum('pnuv, puv -> npuv', phase, zernike_mask_x)
    total_phase_y = jnp.einsum('pnuv, puv -> npuv', phase, zernike_mask_y)
    if phase_maskx is not None:
        total_phase_x = jnp.einsum('npuv, puv -> npuv', total_phase_x, phase_maskx)
        total_phase_y = jnp.einsum('npuv, puv -> npuv', total_phase_y, phase_masky)
    '''  version before the Stokes matrix version
    # --- Polarization rotations (numeric, JIT-friendly) ---

    def rotate_fields(Ex_list, Ey_list, proj): # for 1 given projection angle
        ex_rot = jnp.stack([jnp.cos(proj*jnp.pi/180) * Ex_list[j] +
                            jnp.sin(proj*jnp.pi/180) * Ey_list[j] for j in range(3)])
        ey_rot = jnp.stack([-jnp.sin(proj*jnp.pi/180) * Ex_list[j] +
                             jnp.cos(proj*jnp.pi/180) * Ey_list[j] for j in range(3)])
        return ex_rot, ey_rot # 3, uv
    #print(total_phase_x.shape)
    # Apply rotation for all planes
    def rotated_plane(plane_idx):
        return rotate_fields([Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], polar_projections[plane_idx])

    ex_stack, ey_stack = jax.vmap(rotated_plane)(jnp.arange(len(second_plane))) # dim p, 3, Npix, Npix
    '''
    ex_stack = jnp.array([Ex0, Ex1, Ex2])
    ey_stack = jnp.array([Ey0, Ey1, Ey2])
    E_ex = jnp.fft.fftshift(jnp.fft.fft2(jnp.einsum('apuv, npuv -> napuv', ex_stack, total_phase_x)), axes=(-2,-1))
    E_ey = jnp.fft.fftshift(jnp.fft.fft2(jnp.einsum('apuv, npuv -> napuv', ey_stack, total_phase_y)), axes=(-2,-1))
    
    # --- Compute M matrices ---
    def compute_plane_M(Eplane): 
        E0, E1, E2 = Eplane  # shape: 3, p, uv
        M_plane = jnp.stack([
            jnp.stack([E0*jnp.conj(E0), E0*jnp.conj(E1), E0*jnp.conj(E2)]),
            jnp.stack([E1*jnp.conj(E0), E1*jnp.conj(E1), E1*jnp.conj(E2)]),
            jnp.stack([E2*jnp.conj(E0), E2*jnp.conj(E1), E2*jnp.conj(E2)])
        ])  # shape: (3,3, p, uv)
        return M_plane

    #dim n, 3, 3, p, uv
    Mx_list = jax.vmap(compute_plane_M)(E_ex)
    My_list = jax.vmap(compute_plane_M)(E_ey) # parallelized over n
    #dim 2, n, 3, 3, p, uv to n, p, 2, 3, 3, uv 
    M = jnp.transpose(jnp.stack([Mx_list, My_list]), (1,4,0,2,3,5,6))
    return M

compute_M_jax_jit = jax.jit(compute_M_jax)

def PSF_jax(rho, eta, delta, M, N_photons=1000):
    """
    Compute PSF for multiple PSFs and multiple planes in JAX.

    Parameters
    ----------
    rho, eta, delta : arrays of shape (N_psf,)
        Orientation parameters in degrees.
    M : array, shape (N_psf, K_plane, 2, 3, 3, Nx, Ny)
        Calibration matrices (basis functions).
    N_photons : int or array
        Total photons per PSF.
        
    Returns
    -------
    psf : array, shape (N_psf, K_plane, 2, Nx, Ny)
        PSF for each PSF and plane, for two polarization projections.
    """
    print(f"Compiling PSF_jax - shapes: rho={rho.shape}, eta={eta.shape}, delta={delta.shape}, M={M.shape}, N_photons={type(N_photons)}")
    # Convert angles to radians
    rho = jnp.deg2rad(rho)
    eta = jnp.deg2rad(eta)
    delta = jnp.deg2rad(delta)

    # Rotation matrices for multiple PSFs
    sin_rho = jnp.sin(rho)
    cos_rho = jnp.cos(rho)
    cos_eta = jnp.cos(eta)
    sin_eta = jnp.sin(eta)

    # shape: (N_psf, 3, 3)
    R = jnp.stack([
        jnp.stack([sin_rho**2 * (1 - cos_eta) + cos_eta,
                   sin_rho * cos_rho * (cos_eta - 1),
                   cos_rho * sin_eta], axis=-1),
        jnp.stack([sin_rho * cos_rho * (cos_eta - 1),
                   cos_rho**2 * (1 - cos_eta) + cos_eta,
                   sin_rho * sin_eta], axis=-1),
        jnp.stack([-cos_rho * sin_eta,
                   -sin_rho * sin_eta,
                   cos_eta], axis=-1)
    ], axis=1)

    # Eigenvalues for wobbling
    lam = jnp.stack([
        (1 - jnp.cos(delta / 2)) * (jnp.cos(delta / 2) + 2) / 6,
        (1 - jnp.cos(delta / 2)) * (jnp.cos(delta / 2) + 2) / 6,
        ((jnp.cos(delta / 2)**3 - 1) / (jnp.cos(delta / 2) - 1)) / 3
    ], axis=-1)  # shape (N_psf, 3)

    # Expand lam for broadcasting over planes
    lam_exp = lam[:, None, :]  # shape: (N_psf, 1, 3)

    # Compute PSF for each polarization
    def psf_projection(Mpol, Rpol, lam_exppol):
        # Mpol: (N_psf, K_plane, 3, 3, Nx, Ny)
        # einsum: rotate basis functions and compute weighted sum
        rotated = jnp.einsum('da,acpquv->dcpquv', Rpol.transpose(1,0), jnp.einsum('pqabuv,bc->acpquv', Mpol, Rpol * lam_exppol))
        diag = jnp.diagonal(rotated, axis1=0, axis2=1)  # take diagonal over rotated 3x3
        psf = jnp.sum(diag, axis=-1)  # sum over eigenvalues
        return jnp.clip(jnp.real(psf), min=0.0)

    psf = jax.vmap(psf_projection)(M, R, lam_exp)

    # Normalize each PSF
    norm = jnp.sum(psf, axis=(2, 3, 4), keepdims=True)
    psf = psf * N_photons[:,None,None,None,None] / norm

    return psf

def noise_jax(key, PSF, QE=1.0, EM=1.0, b=0.0, sigma_b=0.0, sigma_r=0.0, bias=0.0):
    """
    Adds mixed Poisson-Gaussian noise to a PSF using JAX.
    
    key      : jax.random.PRNGKey
    PSF      : array, shape (N_psf, n_plane, n_pol, Nx, Ny)
    QE       : quantum efficiency
    EM       : electron-multiplying gain
    b        : background mean
    sigma_b  : background std
    sigma_r  : read noise std
    bias     : camera bias (offset)
    """
    key_poiss, key_b, key_r, key_gamma = random.split(key, 4)

    # Gaussian background (added only inside Poisson)
    background = b + sigma_b * random.normal(key_b, PSF.shape)
    background = jnp.clip(background, min=0.0)
    
    # Poisson shot noise on PSF + background
    lam = jnp.clip(PSF + background, min=1e-6)
    poiss = random.poisson(key_poiss, lam)
    
    # Gaussian read noise with bias as mean
    read = bias + sigma_r * random.normal(key_r, PSF.shape)
    
    # Gamma excess noise from EM gain
    #gamma_conc = jnp.clip(poiss * QE, min=1e-6)
    #excess = random.gamma(key_gamma, gamma_conc) / (QE )
    
    # Combine — same as PyTorch: read + excess only
    noisy = read + poiss
        
    return noisy

