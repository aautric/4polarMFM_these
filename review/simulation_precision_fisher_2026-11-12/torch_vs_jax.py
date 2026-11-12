# read-only: same PSF from the torch model (CRB set-up of crb_vs_unbiased_sim.py) and the JAX model (set-up of SGD_jax.py
# as in the simulations: lambda 620, J_dichroic [I, rot(45), I], polar_projections unused, SAF, planes 0.646/1/1.354)
import os, sys, numpy as np, torch
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '/mnt/c/Users/Amaury/Documents/GitHub/4polarMFM_these_simulation_review')
import jax, jax.numpy as jnp
import simu_PSF_polarMFM_JAX as SJ
from simu_PSF_polarMFM import *
device = 'cpu'
rot = lambda a: np.array([[np.cos(np.radians(a)), -np.sin(np.radians(a))], [np.sin(np.radians(a)), np.cos(np.radians(a))]])
planes = np.array([0.646, 1., 1.354])
rng = np.random.default_rng(3); n = 8
xp, yp, zp = rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n), rng.uniform(0.2, 1.2, n)
rho, eta, delta = rng.uniform(0, 180, n), rng.uniform(0, 180, n), rng.uniform(60, 120, n)
NNp = 2100.

def torch_psf(lam=617, proj=(0, 45, 0)):
    g = globals()
    g.update(NA=torch.tensor(1.4), mag=torch.tensor(100), lambd=torch.tensor(lam), f_tube=torch.tensor(200), l_pixel=torch.tensor(16),
             MAG=torch.tensor(200/150), zernike_coefs_x=torch.zeros((3, 15)), polar_projections=np.array(proj), J=np.array([[1, 0], [0, 1]]))
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../simu_precision/crb_vs_simulation.py')).read()
    exec('def build_M' + src.split('def build_M')[1].split('BCK_CRB = ')[0], g)
    M = g['build_M'](xp=torch.tensor(xp), yp=torch.tensor(yp), zp=torch.tensor(zp), d_=torch.tensor(-planes[1]*np.ones(n)),
                     second_plane_=torch.tensor(planes[1] - planes), SAF_=True)
    return PSF(rho=torch.tensor(rho), eta=torch.tensor(eta), delta=torch.tensor(delta), M=M, N_photons=torch.tensor(NNp*np.ones(n))).numpy()

def jax_psf(lam=620., J2=rot(45)):
    N = 80; NA, mag, f_tube, MAG, l_pixel = 1.4, 100, 200, 200/150, 16
    Jd = jnp.array(np.array([np.eye(2), J2, np.eye(2)]))
    xx, yy, th1, phi, (Ex0, Ex1, Ex2), (Ey0, Ey1, Ey2), r, r_cut, r_cut_saf, k_, f_o, costh2 = SJ.vectorial_BFP_perfect_focus_jax(N, NA=NA, mag=mag, lambd_nm=lam, f_tube_mm=f_tube, J_dichroic=Jd, SAF=True)
    u, v, Npad = SJ.padding_jax(r, r_cut, k_, f_o, N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lam, f_tube=f_tube, MAG=MAG)
    P = lambda f: SJ.pad_jax(f, Npad).astype(jnp.complex64)
    xx, yy, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2, costh2 = [P(f) for f in (xx, yy, th1, phi, Ex0, Ex1, Ex2, Ey0, Ey1, Ey2, costh2)]
    pm = P(jnp.stack([jnp.ones((N, N))]*3))
    zb = P(SJ.generate_zernike_base_jax(r_cut=r_cut, N=N, zernike_order=4, skip_indices={0, 1, 2, 3}))
    d = planes
    M = SJ.compute_M_jax(xp=jnp.array(xp, jnp.float32), yp=jnp.array(yp, jnp.float32), zp=jnp.array(zp, jnp.float32), d=jnp.array(-d[1]*np.ones(n), jnp.float32),
                         x=xx, y=yy, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2, Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, u=u, v=v,
                         phase_maskx=pm, phase_masky=pm, zernike_base=zb, zernike_coefs_x=jnp.zeros((3, 15)), zernike_coefs_y=jnp.zeros((3, 15)), second_plane=jnp.array([d[1]-d[0], 0, d[1]-d[2]]),
                         polar_projections=jnp.array([0, 45, 0]), lambd=lam, f_tube=f_tube, SAF=True, cos_th2=costh2)
    return np.array(SJ.PSF_jax(rho=jnp.array(rho, jnp.float32), eta=jnp.array(eta, jnp.float32), delta=jnp.array(delta, jnp.float32), M=M, N_photons=jnp.array(NNp*np.ones(n), jnp.float32)))

def crop(h, half=6):
    c = h.shape[-1]//2
    return h[..., c-half:c+half+1, c-half:c+half+1]

def compare(a, b, name):
    A, B = crop(a), crop(b)
    d = np.abs(A - B).sum((-2, -1))/B.sum((-2, -1)) # per PSF and channel, relative L1
    print(f'{name}: image sizes {a.shape[-1]} / {b.shape[-1]}; photons in 13x13 (all channels) {A.sum((1,2,3,4)).mean():.0f} / {B.sum((1,2,3,4)).mean():.0f}; '
          f'relative L1 difference per channel: median {np.median(d):.3f}, max {d.max():.3f}; per channel (plane, pol) max:\n{d.max(0).round(3)}')

hj = jax_psf()
for lam, proj, name in ((617, (0, 45, 0), 'torch as in the CRB (617, +45) vs JAX simulation (620, J2 rot 45)'),
                        (620, (0, 45, 0), 'torch 620, +45 vs JAX'),
                        (620, (0, -45, 0), 'torch 620, -45 vs JAX')):
    compare(torch_psf(lam, proj), hj, name)
