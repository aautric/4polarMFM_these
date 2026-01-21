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

n1 = 1.52 # oil and sample indices
n2 = 1.33 # water index
h = 6.626*10**(-34) # Planck constant
c = 299792458 # speed of light

# everything in micrometers 
    
@partial(jax.jit, static_argnums=(0,))
def vectorial_BFP_perfect_focus_jax(
    N: int,
    NA: float = 1.4,
    mag: float = 100.0,
    lambd_nm: float = 617.0,
    f_tube_mm: float = 200.0,
    n1: float = 1.33,
    n2: float = 1.0,
    dtype=jnp.float32,
):
    """
    JAX version of vectorial_BFP_perfect_focus.
    - N : grid size (static, int)
    - NA, mag, lambd_nm (nm), f_tube_mm (mm), refractive indices n1,n2
    - returns: x, y, th1, phi, (Ex0,Ex1,Ex2), (Ey0,Ey1,Ey2), r, r_cut, k, f_o
    """
    # convert units
    lambd = jnp.array(1e-3 * lambd_nm, dtype=dtype)      # nm -> um
    f_tube = jnp.array(1000.0 * f_tube_mm, dtype=dtype)  # mm -> um
    f_o = f_tube / mag
    k = 2.0 * n1 * jnp.pi / lambd

    # spatial cutoff
    r_cut = jnp.minimum(NA / n1, n2 / n1)

    # meshgrid
    coords = jnp.linspace(-r_cut, r_cut, N, dtype=dtype)
    x, y = jnp.meshgrid(coords, coords, indexing="xy")
    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)

    # angles
    th1 = jnp.where(r < r_cut, jnp.arcsin(r), 0.0)
    th2 = jnp.where(r < r_cut, jnp.arcsin((n1 / n2) * r), 0.0)

    # transmission coefficients
    cos_th1 = jnp.cos(th1)
    cos_th2 = jnp.cos(th2)
    Ts = jnp.where(r < r_cut, (2.0 * n2 * cos_th2) / (n2 * cos_th2 + n1 * cos_th1), 0.0)
    Tp = jnp.where(r < r_cut, (2.0 * n2 * cos_th2) / (n2 * cos_th1 + n1 * cos_th2), 0.0)

    sqrt_cos_th1 = jnp.sqrt(jnp.clip(cos_th1, a_min=1e-12))  # stable sqrt

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

    return (
        x, y, th1, phi,
        (Ex0, Ex1, Ex2),
        (Ey0, Ey1, Ey2),
        r, r_cut, k, f_o
    )

def psi_lat_jax(x, y, theta, phi, n1=1.518, lambd=0.617):
    """
    JAX version of psi_lat.
    Works for:
        - scalar x,y
        - vectors x,y
        - 2D grids x,y
        - theta, phi arrays (broadcasting)
    """

    phase = jnp.einsum('bc,abc->abc', jnp.sin(theta), (jnp.einsum('a,bc->abc', x, jnp.cos(phi)) +jnp.einsum('a,bc->abc', y, jnp.sin(phi))))
    return phase * (2*jnp.pi*n1) / lambd

def psi_z_jax(theta, z, NA=1.4, mag=100, lambd=0.617, f_tube=200_000, n1=1.33, n2=1.515):
    """
    JAX version of psi_z (Yan et al. corrected)
    Computes the phase term along z for vectorial PSF simulations.
    """

    sqrt_term = jnp.sqrt(1 - (n1 * jnp.sin(theta) / n2)**2)
    factor = 2 * jnp.pi * n2 / lambd
    print(sqrt_term.shape)
    # If z is scalar, broadcast automatically
    
    return jnp.einsum('a, bc -> abc', z, factor*sqrt_term)

def psi_f_jax(theta, d, NA=1.4, mag=100, lambd=0.617, f_tube=200_000, n1=1.33, n2=1.515):
    """
    JAX version of psi_f (Yan et al. corrected)
    Computes the phase term along the focal region for vectorial PSF simulations.
    """

    # Refractive index depending on sign of d
    n = n1 + (n2 - n1) * (1 + jnp.sign(d)) / 2

    # Scalar case
    if d.ndim == 0:
        return jnp.einsum('a, bc -> abc', d, 2 * jnp.pi * n * jnp.cos(theta) / lambd)
    else:
        # Multiple PSFs: outer product of d*n and cos(theta)
        # Use jnp.cos(theta) for general theta array
        cos_theta = jnp.sqrt(1 - jnp.sin(theta.flatten())**2)
        return jnp.outer(d * n, cos_theta) * 2 * jnp.pi / lambd

psi_f_jit = jax.jit(psi_f_jax)
psi_z_jit = jax.jit(psi_z_jax)
psi_lat_jit = jax.jit(psi_lat_jax)

def generate_zernike_base_jax(r_cut, N, zernike_order=4):
    """
    Generate a Zernike polynomial basis on a square grid for JAX.
    
    Parameters
    ----------
    r_cut : float
        Maximum radial coordinate (aperture radius).
    N : int
        Grid size (N x N).
    zernike_order : int
        Maximum Zernike order.
    
    Returns
    -------
    zernike_base : array
        Array of shape (num_modes, N, N) containing Zernike modes.
    """
    from zernike import RZern  # Replace with your Zernike class compatible with numpy/jax

    cart = RZern(zernike_order)
    ddx = jnp.linspace(-r_cut, r_cut, N)
    ddy = jnp.linspace(-r_cut, r_cut, N)
    xv, yv = jnp.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)  # your function to setup the grid

    num_modes = (zernike_order + 1) * (zernike_order + 2) // 2
    zernike_base = jnp.zeros((num_modes, N, N))
    
    # Loop over Zernike modes
    for index in range(1, cart.nk):
        zer = jnp.zeros(cart.nk)
        zer = zer.at[index].set(1.0)
        mode = cart.eval_grid(zer, matrix=True)  # returns a (N,N) array
        mode = jnp.nan_to_num(mode, nan=0.0)
        mask = (xv**2 + yv**2) <= r_cut**2
        mode = mode * mask
        zernike_base = zernike_base.at[index].set(mode)
    
    return zernike_base

@jax.jit
def compute_M_jax(xp, yp, zp, d, th1, phi, 
                  Ex0, Ex1, Ex2, Ey0, Ey1, Ey2, 
                  r_cut, k, f_o, phase_maskx, phase_masky, 
                  zernike_base, zernike_coefs_x, zernike_coefs_y,
                  second_plane, polar_projections,
                  N=80, l_pixel=16, MAG=200/150):
    """
    JAX version of compute_M for multiple PSFs and multiple planes, fully JIT-compatible.
    Returns u, v coordinates and M matrix of shape (N_psf, K_plane, 2, 3, 3, N_pix, N_pix)
    """
    lambd = 1e-3 * 617
    f_tube = f_o * 1000  # assuming f_o passed as tube/mag

    # --- Phase for all planes (vectorized with vmap) ---
    def phase_per_plane(dp):
        return jnp.exp(1j * (
            psi_f_jit(th1, d + dp) +
            psi_z_jit(th1, zp) +
            psi_lat_jit(xp, yp, th1, phi)
        ))
    
    phase = jax.vmap(phase_per_plane)(jnp.array(second_plane))  # (K_plane, ...)

    # --- Zernike masks ---
    zernike_mask_x = jnp.exp(1j * jnp.tensordot(zernike_coefs_x, zernike_base, axes=1))
    zernike_mask_y = jnp.exp(1j * jnp.tensordot(zernike_coefs_y, zernike_base, axes=1))

    # --- Padding and frequency grid ---
    Dx = 2 * r_cut * f_o / N
    Npadding = int((2 * jnp.pi * MAG * f_tube) / (k * l_pixel * Dx)) - N
    Npadding += Npadding % 2
    freq = jnp.fft.fftshift(jnp.fft.fftfreq(N + Npadding, Dx)) * 2 * jnp.pi * f_tube / k * MAG
    v, u = jnp.meshgrid(freq, freq)

    # --- Polarization rotations (numeric, JIT-friendly) ---
    polar = jnp.array([0.0, 0.0, 0.0])  # adjust if needed
    polar_proj_vals = jnp.array(polar_projections, dtype=float)

    def rotate_fields(Ex_list, Ey_list, proj):
        ex_rot = jnp.stack([jnp.cos(proj[j]*jnp.pi/180) * Ex_list[j] +
                            jnp.sin(proj[j]*jnp.pi/180) * Ey_list[j] for j in range(3)])
        ey_rot = jnp.stack([-jnp.sin(proj[j]*jnp.pi/180) * Ex_list[j] +
                             jnp.cos(proj[j]*jnp.pi/180) * Ey_list[j] for j in range(3)])
        return ex_rot, ey_rot

    # Apply rotation for all planes
    def rotated_plane(plane_idx):
        return rotate_fields([Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], polar_proj_vals)

    ex_stack, ey_stack = jax.vmap(rotated_plane)(jnp.arange(len(second_plane)))

    # --- FFT with padding ---
    def fft_pad(field, mask):
        return jnp.fft.fftshift(jnp.fft.fft2(pad_jax(field * mask, Npadding)), axes=(-2,-1))

    E_ex = jax.vmap(lambda plane: jax.vmap(lambda comp: fft_pad(comp, zernike_mask_x))(plane))(ex_stack)
    E_ey = jax.vmap(lambda plane: jax.vmap(lambda comp: fft_pad(comp, zernike_mask_y))(plane))(ey_stack)

    # --- Compute M matrices ---
    def compute_plane_M(Eplane):
        E0, E1, E2 = Eplane  # shape: (3, N_pix, N_pix)
        M_plane = jnp.stack([
            jnp.stack([E0*jnp.conj(E0), E0*jnp.conj(E1), E0*jnp.conj(E2)]),
            jnp.stack([E1*jnp.conj(E0), E1*jnp.conj(E1), E1*jnp.conj(E2)]),
            jnp.stack([E2*jnp.conj(E0), E2*jnp.conj(E1), E2*jnp.conj(E2)])
        ])  # shape: (3,3,N_pix,N_pix)
        return M_plane

    Mx_list = jax.vmap(compute_plane_M)(E_ex)
    My_list = jax.vmap(compute_plane_M)(E_ey)

    M = jnp.stack([Mx_list, My_list], axis=1)  # (K_plane, 2, 3,3, N_pix, N_pix)

    return u, v, M

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
    def psf_projection(Mpol):
        # Mpol: (N_psf, K_plane, 3, 3, Nx, Ny)
        # einsum: rotate basis functions and compute weighted sum
        rotated = jnp.einsum('nij,njkpq->nikpq', R.transpose(0, 2, 1), jnp.einsum('nijkpq,nkj->nijpq', Mpol, R * lam_exp))
        diag = jnp.diagonal(rotated, axis1=2, axis2=3)  # take diagonal over rotated 3x3
        psf = jnp.sum(diag, axis=-1)  # sum over eigenvalues
        return jnp.clip(jnp.real(psf), a_min=0.0)

    psf_x = psf_projection(M[:, :, 0])
    psf_y = psf_projection(M[:, :, 1])

    psf = jnp.stack([psf_x, psf_y], axis=2)  # shape: (N_psf, K_plane, 2, Nx, Ny)

    # Normalize each PSF
    norm = jnp.sum(psf, axis=(2, 3, 4), keepdims=True)
    psf = psf * N_photons / norm

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
    # split key for randomness
    key_poiss, key_b, key_r, key_gamma = random.split(key, 4)
    
    # Gaussian background
    background = b + sigma_b * random.normal(key_b, PSF.shape)
    background = jnp.clip(background, a_min=0.0)
    
    # Poisson shot noise
    lam = PSF + background
    lam = jnp.clip(lam, a_min=1e-6)  # prevent zero
    poiss = random.poisson(key_poiss, lam)
    
    # Gaussian read noise
    read = sigma_r * random.normal(key_r, PSF.shape)
    
    # Gamma excess noise from EM gain
    gamma_conc = jnp.clip(poiss * QE, a_min=1e-6)
    gamma_rate = EM
    # sample Gamma: Gamma(k, theta) with mean k*theta; JAX uses concentration, rate
    excess = random.gamma(key_gamma, gamma_conc) / gamma_rate
    
    # Combine all terms
    noisy = poiss + read / (QE*EM) + bias / (QE*EM) + excess / (QE*EM)
    
    return noisy

def pad_jax(a, n):
    """Traceable JAX padding."""
    if a.ndim == 2:
        padded = jnp.zeros((a.shape[0] + n, a.shape[1] + n), dtype=a.dtype)
        padded = padded.at[n//2:-n//2, n//2:-n//2].set(a)
    elif a.ndim == 3:
        padded = jnp.zeros((a.shape[0], a.shape[1] + n, a.shape[2] + n), dtype=a.dtype)
        padded = padded.at[:, n//2:-n//2, n//2:-n//2].set(a)
    else:
        raise ValueError("Unsupported number of dimensions for padding.")
    return padded