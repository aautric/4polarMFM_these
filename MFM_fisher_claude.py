#%% MFM FISHER

# -*- coding: utf-8 -*-
"""
Copy of MFM_fisher.ipynb (2026-10-28) as a script with cells. Only change: the Cramer-Rao bound is the mean over the
PSF of the inverse of each PSF's Fisher matrix, instead of the inverse of the mean Fisher matrix.
2026-10-28: uses fisher_4pMFM_claude.py (d gamma/d delta fixed: pi sin(delta/2) instead of 2 pi sin(delta)).
"""
#%%
from simu_PSF_polarMFM import *
from fisher_4pMFM_claude import * # copy of fisher_4pMFM.py with d gamma/d delta fixed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re
from tqdm import tqdm
import os
import json
import hashlib

#%%
# Cramer-Rao bound averaged over the PSF: each PSF has its own Fisher matrix, which is inverted first, and the
# covariance bounds are then averaged (before: the Fisher matrices were averaged, then inverted)
def crlb_cov(F):
    # F: (n_params, n_params, NPSF) Fisher matrix of each PSF -> mean over the PSF of its inverse, (n_params, n_params)
    return torch.mean(torch.linalg.inv(F.permute(2, 0, 1)), dim=0)

#%%
NPSF = 400
device='cuda'

#%%
# photons and noise of actin1 (D:\AMAURY\DATA\0_papier\actin1, run NPZ_2026-10-24_15h28, sensitivity 4.4, fits with
# deviance < 1.3): median N of the whole molecule (1216 per plane in the fit, PSF_jax normalised each plane until
# 2026-10-28), background per pixel, noise model of SGD_jax.py var = gain x mu + V (V = gain x shift)
N_actin1 = 3*1216 # photons, whole molecule
bck_actin1 = 22.2 # photons per pixel
gain_actin1 = 1.8
dark_var_actin1 = 0.74 # photons^2

#%%
N_photons=N_actin1 # number of photons collected 
N=80 # discretization of the BFP
l_pixel=16 #pixel size in micrometer
NA=1.4 # numerical aperture
mag=100 # first magnification
lambd=617 # wavelength in nm
f_tube=200 # tube lens focal in mm
MAG=200/150 # second magnification

N_photons=torch.tensor(N_photons, device=device)
N=torch.tensor(N, device=device)
l_pixel=torch.tensor(l_pixel, device=device)
NA=torch.tensor(NA, device=device)
mag=torch.tensor(mag, device=device)
lambd=torch.tensor(lambd, device=device)
f_tube=torch.tensor(f_tube, device=device)
MAG=torch.tensor(MAG, device=device)

#%%
xp=torch.tensor(np.random.rand(NPSF)*0.1, device=device)
yp=torch.tensor(np.random.rand(NPSF)*0.1, device=device)
z=torch.tensor(np.random.rand(NPSF)*1.+1., device=device)
d_=torch.tensor([-2. for i in range(NPSF)], device=device)
rho = torch.tensor([45 for k in range(NPSF)], device=device)
eta = torch.tensor([45 for k in range(NPSF)], device=device)
delta = torch.tensor([80 for k in range(NPSF)], device=device)
second_plane = torch.tensor([-0.365, 0, 0.365])
zernike_coefs_x = torch.zeros((3,15), device=device)
zernike_coefs_y = torch.zeros((3,15), device=device)
polar_projections = np.array([0, 45, 0])

#%%
J1 = np.array([[ 0.89642966        ,             -0.15750188 + 1j*  -0.41388938 ],[
      -0.22662663 + 1j*  0.37517267  ,   -0.88134384 + 1j*  -0.17010298 ]])
J2 = np.array([[ 0.5655862               ,      -0.77402043 + 1j*  0.27935567 ],[
      0.42339513 + 1j*  -0.7021969   ,  0.11452306 + 1j*  -0.5653576 ]])
J = np.array([J1, J2, J1])

J = np.array([[1,0],[0,1]])

#%%
SAF = False

#%%
if SAF:
    x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, r_cut_saf, k, f_o, costh2 = vectorial_BFP_perfect_focus(N, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF, device=device, J_dichroic=J)
else:
    costh2=None
    x, y, th1, phi, [Ex0, Ex1, Ex2], [Ey0, Ey1, Ey2], r, r_cut, k, f_o = vectorial_BFP_perfect_focus(N, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF, device=device, J_dichroic=J)

#%%
phase_mask=torch.ones((N,N)) 
#phase_mask=torch.exp(1j*phi)
phase_mask[x**2+y**2>r_cut**2]=1.

phase_mask = torch.stack([torch.ones((N,N), device=device), torch.tensor(phase_mask, device=device), torch.ones((N,N), device=device)])
zernike_base = generate_zernike_base(r_cut=r_cut, N=N, zernike_order=4, device=device)

#%%
u, v, Npadding = padding(r, r_cut, k, f_o, N=N, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, 
              f_tube=f_tube, MAG=MAG, device=device)
print(Npadding)

#%%
th1 = pad(th1, Npadding)
phi = pad(phi, Npadding)
Ex0 = pad(Ex0, Npadding)
Ex1 = pad(Ex1, Npadding)
Ex2 = pad(Ex2, Npadding)
Ey0 = pad(Ey0, Npadding)
Ey1 = pad(Ey1, Npadding)
Ey2 = pad(Ey2, Npadding)
phase_mask = pad(phase_mask, Npadding)
zernike_base = pad(zernike_base, Npadding)
if SAF:
    costh2 = pad(costh2, Npadding)

#%%
phase_mask.shape

#%%
M = compute_M(xp=xp, yp=yp, zp=z, d=d_, x=x, y=y, th1=th1, phi=phi, Ex0=Ex0, Ex1=Ex1, Ex2=Ex2,
                    Ey0=Ey0, Ey1=Ey1, Ey2=Ey2, r=r, r_cut=r_cut, k=k, f_o=f_o,
                    phase_maskx=phase_mask, phase_masky=phase_mask, zernike_base=zernike_base, zernike_coefs_x=zernike_coefs_x, 
                    zernike_coefs_y=zernike_coefs_x, second_plane=second_plane, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube, 
                    BFP_version=False, SAF=SAF, costh2=costh2)

#%%
psf = PSF(rho=rho, eta=eta, delta=delta, M=M, N_photons=N_photons)

#%%
RHO = torch.tensor(np.random.rand(NPSF)*2*np.pi, device=device)
ETA = torch.tensor(np.arccos(np.random.uniform(np.cos(0*np.pi/180), np.cos(90*np.pi/180), NPSF)), device=device)
DELTA = torch.tensor(50*np.pi/180+np.random.rand(NPSF)*(100*np.pi/180), device=device)#torch.tensor([70 for uu in range(NPSF)], device=device)#torch.tensor(0.0001*np.random.rand(NPSF), device=device)#
N = torch.tensor([N_actin1 for uu in range(NPSF)], device=device)#torch.tensor((2000+np.random.rand(NPSF)*6000), device=device)#*(torch.sin(ETA)**2)
bck = torch.tensor(bck_actin1, device=device)

#%%
u = plt.hist(ETA.cpu().numpy()*180/np.pi, bins=30)

#%%
F = fisher(M, RHO, ETA, DELTA, N, bck, gain=gain_actin1, dark_var=dark_var_actin1)

#%%
F_mean = torch.mean(F, dim=2) # mean Fisher matrix, only displayed
S = crlb_cov(F) # mean over the PSF of the covariance bound of each PSF
print(F_mean.cpu().detach().numpy())

#%%
plt.imshow(F_mean.cpu().detach().numpy(), vmin=0)
plt.xticks([])
plt.yticks([])
plt.show()

#%%
#torch.sqrt(F)*(180/np.pi)
torch.sqrt(S)*(180/np.pi)

#%%
s = torch.diagonal(S)
CORR = S/torch.sqrt(torch.einsum('i,j->ij', s, s))
fig, ax = plt.subplots()
ax.imshow(CORR.cpu().detach().numpy(), vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([])

#%%
#fig.savefig("C_fit.svg", format="svg")

#%% On ratiometric

#%%
# ratiometric bound: intensities of the 6 channels summed over a window of ratio_window x ratio_window px
# (None = whole image); ratio_fixed_eta = True: eta known, bound on (rho, delta) only, as in the ratiometric method
# (window=None + fixed_eta=True: pure polarisation-ratio bound; with the whole image eta and delta cannot be separated)
ratio_window = 11
ratio_fixed_eta = False

#%%
F = fisher_ratiometric(M, RHO, ETA, DELTA, N, bck, window=ratio_window, fixed_eta=ratio_fixed_eta, gain=gain_actin1, dark_var=dark_var_actin1)

#%%
F_mean = torch.mean(F, dim=2) # mean Fisher matrix, only displayed
S = crlb_cov(F) # mean over the PSF of the covariance bound of each PSF

#%%
fig, ax = plt.subplots()
ax.imshow(F_mean.cpu().detach().numpy(), vmin=0)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

#%%
torch.sqrt(S)*(180/np.pi)

#%%
s = torch.diagonal(S)
CORR = S/torch.sqrt(torch.einsum('i,j->ij', s, s))
fig, ax = plt.subplots()
ax.imshow(CORR.cpu().detach().numpy(), vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([])

#%%
#fig.savefig("C_ratio.svg", format="svg")

#%% WRT to eta: CRB vs simulated SGD
# Best performance: comparison of the Cramer-Rao bounds with the precision of the SGD fit (huber loss, the gold
# standard) on simulated data without read noise nor background fluctuation, Poisson + EMCCD excess noise:
# SGD_precision_claude.py run 2026-10-14_23h37_poisson_excess (D:\AMAURY\simu_precision_claude): N 3000 photons
# per molecule (1000 per plane: the simulation used PSF_jax normalised per plane, before 2026-10-28),
# background 25 photons per pixel, delta 90 deg, grid of 20 rho x 20 eta (bin centres 4.5 ... 175.5 deg), 80 molecules
# per point, x, y in [-0.2, 0.2] um, z in [0.2, 1.2] um, planes focused at 0.646 / 1 / 1.354 um, SAF on.
# Noise of the simulation (emccd_jax): variance per pixel 2 (s + b)/QE. The bounds are computed with Poisson noise
# only (gain = 1, dark_var = 0): ideal detector, best possible precision (EM excess would multiply them by sqrt(2/QE)).
sgd_results = '/mnt/d/AMAURY/simu_precision_claude/2026-10-14_23h37_poisson_excess/results_huber.npz'
out_folder = '/mnt/d/AMAURY/simu_precision_claude/MFM_fisher' # bounds, SGD precision and figures of this comparison
os.makedirs(out_folder, exist_ok=True)
QE_sim = 0.92
delta_sim = 90 # delta of the simulation
delta_crb = (80, 110) # delta of the bounds: uniform in this range (deg)
eta_crb = np.linspace(1, 179, 45) # deg
# N = 3000: photons of the whole molecule (PSF() of this script, and PSF_jax since 2026-10-28; the simulation
# was made when PSF_jax normalised each plane, with N = 1000 per plane)
# CRB: Poisson noise of the molecule only, NO background photons (final choice, 2026-10-28): best possible precision.
# It does not match the conditions of the simulated SGD (background 25, EM excess), which is shown for reference.
conditions = {'N 3000; CRB: Poisson only, no background; SGD huber: background 25, Poisson + EM excess': dict(N=3*1000, bck=0., gain=1., dark_var=0.),
              # second set: same with the background of actin1 (Poisson noise of the background included)
              f'N 3000; CRB: Poisson only, background {bck_actin1} (actin1); SGD huber: background 25, Poisson + EM excess': dict(N=3*1000, bck=bck_actin1, gain=1., dark_var=0.)}

#%%
# calibration images M with the geometry of the simulation (SAF on)
def build_M(xp, yp, zp, d_, second_plane_, SAF_):
    N_bfp = torch.tensor(80, device=device) # discretisation of the BFP (N is used for the photons further up)
    if SAF_:
        x_, y_, th1_, phi_, [Ex0_, Ex1_, Ex2_], [Ey0_, Ey1_, Ey2_], r_, r_cut_, r_cut_saf_, k_, f_o_, costh2_ = vectorial_BFP_perfect_focus(N_bfp, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF_, device=device, J_dichroic=J)
    else:
        costh2_ = None
        x_, y_, th1_, phi_, [Ex0_, Ex1_, Ex2_], [Ey0_, Ey1_, Ey2_], r_, r_cut_, k_, f_o_ = vectorial_BFP_perfect_focus(N_bfp, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF_, device=device, J_dichroic=J)
    pm = torch.stack([torch.ones((N_bfp, N_bfp), device=device)]*3)
    zb = generate_zernike_base(r_cut=r_cut_, N=N_bfp, zernike_order=4, device=device)
    _, _, npad = padding(r_, r_cut_, k_, f_o_, N=N_bfp, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, MAG=MAG, device=device)
    th1_, phi_, Ex0_, Ex1_, Ex2_, Ey0_, Ey1_, Ey2_, pm, zb = [pad(t, npad) for t in (th1_, phi_, Ex0_, Ex1_, Ex2_, Ey0_, Ey1_, Ey2_, pm, zb)]
    if SAF_:
        costh2_ = pad(costh2_, npad)
    return compute_M(xp=xp, yp=yp, zp=zp, d=d_, x=x_, y=y_, th1=th1_, phi=phi_, Ex0=Ex0_, Ex1=Ex1_, Ex2=Ex2_,
                     Ey0=Ey0_, Ey1=Ey1_, Ey2=Ey2_, r=r_, r_cut=r_cut_, k=k_, f_o=f_o_,
                     phase_maskx=pm, phase_masky=pm, zernike_base=zb, zernike_coefs_x=zernike_coefs_x,
                     zernike_coefs_y=zernike_coefs_x, second_plane=second_plane_, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube,
                     BFP_version=False, SAF=SAF_, costh2=costh2_)
planes_sim = np.array([0.646, 1., 1.354])
M_sim = build_M(xp=torch.tensor(np.random.uniform(-0.2, 0.2, NPSF), device=device),
                yp=torch.tensor(np.random.uniform(-0.2, 0.2, NPSF), device=device),
                zp=torch.tensor(np.random.uniform(0.2, 1.2, NPSF), device=device),
                d_=torch.tensor(-planes_sim[1]*np.ones(NPSF), device=device),
                second_plane_=torch.tensor(planes_sim[1]-planes_sim, device=device), SAF_=True)

#%%
# CRB (full PSF), standard ratiometric CRB (11 x 11 px window, rho, eta, delta free) and ratiometric CRB with eta
# known (11 x 11 px, rho, delta free), mean over the molecules of the inverse of their Fisher matrix, per eta;
# rho uniform in [0, 180[, delta uniform in delta_crb, x, y, z of M_sim; N, background and noise of the simulation
# Saved in out_folder/crb_<key>.npz, one file per set of parameters (key = hash of crb_params): computed only once,
# loaded afterwards (recompute_crb = True to force); the files of the earlier parameters are kept.
recompute_crb = False
cr_full, cr_ratio, cr_ratio_eta = {}, {}, {}
for name, c in conditions.items():
    # one file per condition (key = hash of its parameters)
    crb_params = dict(conditions={name: c}, delta_crb=delta_crb, eta_crb=eta_crb.tolist(), NPSF=NPSF, window=11,
                      geometry=dict(xy=0.2, z=(0.2, 1.2), planes=planes_sim.tolist(), SAF=True),
                      fisher='fisher_4pMFM_claude.py (d gamma/d delta fixed, normalisation term, gain/dark_var)')
    crb_key = hashlib.md5(json.dumps(crb_params, sort_keys=True).encode()).hexdigest()[:8]
    crb_file = os.path.join(out_folder, f'crb_{crb_key}.npz')
    if os.path.exists(crb_file) and not recompute_crb:
        c_ = np.load(crb_file)
        cr_full[name], cr_ratio[name], cr_ratio_eta[name] = c_['cr_full'][0], c_['cr_ratio'][0], c_['cr_ratio_eta'][0]
        print('CRB loaded from', crb_file)
        continue
    cr_full[name], cr_ratio[name], cr_ratio_eta[name] = [], [], []
    for eta_ in tqdm(eta_crb):
        RHO = torch.tensor(np.random.rand(NPSF)*np.pi, device=device)
        ETA = torch.tensor(np.radians(eta_)*np.ones(NPSF), device=device)
        DELTA = torch.tensor(np.radians(np.random.uniform(*delta_crb, NPSF)), device=device)
        N = torch.tensor(c['N']*np.ones(NPSF), device=device)
        bck = torch.tensor(float(c['bck']), device=device)
        F = fisher(M_sim, RHO, ETA, DELTA, N, bck, gain=c['gain'], dark_var=c['dark_var'])
        Fr = fisher_ratiometric(M_sim, RHO, ETA, DELTA, N, bck, window=11, fixed_eta=False, gain=c['gain'], dark_var=c['dark_var'])
        cr_full[name].append((torch.sqrt(torch.diagonal(crlb_cov(F)))*(180/np.pi)).cpu().numpy())
        cr_ratio[name].append((torch.sqrt(torch.diagonal(crlb_cov(Fr)))*(180/np.pi)).cpu().numpy())
        Fre = fisher_ratiometric(M_sim, RHO, ETA, DELTA, N, bck, window=11, fixed_eta=True, gain=c['gain'], dark_var=c['dark_var'])
        cre = (torch.sqrt(torch.diagonal(crlb_cov(Fre)))*(180/np.pi)).cpu().numpy() # rho, delta
        cr_ratio_eta[name].append(np.array([cre[0], np.nan, cre[1]]))
    cr_full[name], cr_ratio[name], cr_ratio_eta[name] = np.array(cr_full[name]), np.array(cr_ratio[name]), np.array(cr_ratio_eta[name]) # (eta, [rho, eta, delta])
    np.savez(crb_file, params=json.dumps(crb_params, sort_keys=True), names=[name],
             cr_full=[cr_full[name]], cr_ratio=[cr_ratio[name]], cr_ratio_eta=[cr_ratio_eta[name]])
    print('CRB computed and saved in', crb_file)

#%%
# simulated SGD (huber): errors of each molecule (rho and eta wrapped, equivalent orientation (rho+180, 180-eta)
# taken into account by SGD_precision_claude.py); precision per eta = rms over rho of the std over the molecules
res = np.load(sgd_results)
eta_listMC = np.unique(res['true_eta'])
rho_listMC = np.unique(res['true_rho'])
sgd_prec = np.zeros((len(eta_listMC), 3))
for i, e_ in enumerate(eta_listMC):
    for j, err in enumerate((res['error_rho'], res['error_eta'], res['error_delta'])):
        std_rho = [np.nanstd(err[(res['true_eta'] == e_) & (res['true_rho'] == r_)]) for r_ in rho_listMC]
        sgd_prec[i, j] = np.sqrt(np.mean(np.array(std_rho)**2))
# points of the current figure (CRB of crb_key + SGD), read by the figure cell
np.savez(os.path.join(out_folder, 'crb_vs_eta.npz'), eta_crb=eta_crb, eta_listMC=eta_listMC, sgd_prec=sgd_prec,
         names=list(conditions), cr_full=[cr_full[n] for n in conditions], cr_ratio=[cr_ratio[n] for n in conditions],
         cr_ratio_eta=[cr_ratio_eta[n] for n in conditions], sgd_results=sgd_results)

#%%
# figures, 12 cm wide (about half an A4 page); read the saved results, can be run alone
# 1. CRB, standard ratiometric CRB, ratiometric CRB with eta known, SGD;  2. the same with only the eta-known ratiometric CRB
r = np.load(os.path.join(out_folder, 'crb_vs_eta.npz'))
def crb_figure(ratiometric, filename, cond=0):
    # cond: index of the condition in `conditions` (0: no background, 1: actin background)
    with plt.rc_context({'font.size': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2}):
        fig, ax = plt.subplots(1, 3, figsize=(12/2.54, 1.5), sharex=True, sharey=True)
        for i, par in enumerate(['rho', 'eta', 'delta']):
            ax[i].plot(r['eta_crb'], r['cr_full'][cond][:, i], c='b', label='CRB')
            if 'standard' in ratiometric:
                ax[i].plot(r['eta_crb'], r['cr_ratio'][cond][:, i], c='r', label='CRB ratiometric')
            if 'eta_known' in ratiometric:
                ax[i].plot(r['eta_crb'], r['cr_ratio_eta'][cond][:, i], c='r', ls='--', label='CRB ratiometric, $\\eta$ known')
            ax[i].scatter(r['eta_listMC'], r['sgd_prec'][:, i], marker='+', c='k', s=14, linewidths=0.8, label='SGD (huber)')
            ax[i].set_xlabel('$\\eta$ ($\\degree$)', labelpad=1)
            ax[i].set_title('$\\%s$' % par, pad=2)
            ax[i].set_ylim((0, 20))
            ax[i].set_yticks([0, 5, 10, 15, 20])
            ax[i].set_xlim((0, 180))
            ax[i].set_xticks([0, 90, 180])
            ax[i].tick_params(length=2, pad=1.5)
            ax[i].grid(alpha=0.3, lw=0.4)
        ax[0].set_ylabel('precision ($\\degree$)', labelpad=1)
        fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False, fontsize=6,
                   handlelength=1.6, columnspacing=1., bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0, 1, 0.84))
        fig.savefig(os.path.join(out_folder, filename+'.png'), dpi=300)
        fig.savefig(os.path.join(out_folder, filename+'.svg'))
        plt.show()
crb_figure(('standard', 'eta_known'), 'CRB_vs_SGD_eta')
crb_figure(('eta_known',), 'CRB_vs_SGD_eta_ratio_eta_known')
crb_figure(('eta_known',), 'CRB_vs_SGD_eta_ratio_eta_known_actin_background', cond=1)

#%%
# Fisher and covariance matrices at the conditions of the figures (M_sim geometry, N, background and noise of
# `conditions`, delta uniform in delta_crb), orientations uniform on the sphere (rho uniform, cos(eta) uniform),
# NPSF molecules: mean Fisher matrix (rad^-2), mean over the molecules of the inverse = covariance bound (deg^2),
# its square-root diagonal (deg) and the correlation matrix (mean covariance and per molecule)
# background: photons per pixel (0: conditions of the figures; bck_actin1: actin-like background); eta_range (deg)
def fisher_matrices(background, filename, eta_range=(0, 180)):
    c = conditions[list(conditions)[0]]
    np.random.seed(2)
    RHO = torch.tensor(np.random.rand(NPSF)*np.pi, device=device)
    ETA = torch.tensor(np.arccos(np.random.uniform(np.cos(np.radians(eta_range[1])), np.cos(np.radians(eta_range[0])), NPSF)), device=device) # uniform on the sphere in eta_range
    DELTA = torch.tensor(np.radians(np.random.uniform(*delta_crb, NPSF)), device=device)
    N = torch.tensor(c['N']*np.ones(NPSF), device=device)
    bck = torch.tensor(float(background), device=device)
    kw = dict(gain=c['gain'], dark_var=c['dark_var'])
    matrices = {'CRB': (fisher(M_sim, RHO, ETA, DELTA, N, bck, **kw), ['rho', 'eta', 'delta']),
                'CRB ratiometric (11 x 11 px)': (fisher_ratiometric(M_sim, RHO, ETA, DELTA, N, bck, window=11, **kw), ['rho', 'eta', 'delta']),
                'CRB ratiometric, eta known': (fisher_ratiometric(M_sim, RHO, ETA, DELTA, N, bck, window=11, fixed_eta=True, **kw), ['rho', 'delta'])}
    lines = [f'Fisher and covariance matrices; CRB conditions: N {c["N"]} photons in total, background '
             f'{background}, gain {c["gain"]}, V {c["dark_var"]}, delta uniform in {delta_crb} deg, orientations uniform on the sphere, eta in {eta_range} deg, '
             f'{NPSF} molecules, geometry of the simulation (z 0.2-1.2 um, SAF on)', '']
    with plt.rc_context({'font.size': 7}):
        fig, ax = plt.subplots(2, 3, figsize=(12/2.54, 2.9))
        for k, (name, (F_, labels)) in enumerate(matrices.items()):
            Fm = torch.mean(F_, dim=2).cpu().numpy()                                      # mean Fisher matrix (rad^-2)
            C = crlb_cov(F_).cpu().numpy()*(180/np.pi)**2                                # covariance bound (deg^2)
            sd = np.sqrt(np.diag(C)); R = C/np.outer(sd, sd)
            lines += [name, '  mean Fisher matrix (rad^-2), ' + ' / '.join(labels)] + ['    ' + '  '.join(f'{v:12.1f}' for v in row) for row in Fm]
            lines += ['  covariance bound, mean over the molecules of F^-1 (deg^2)'] + ['    ' + '  '.join(f'{v:12.2f}' for v in row) for row in C]
            lines += ['  sqrt of the diagonal (deg): ' + ', '.join(f'{l} {v:.2f}' for l, v in zip(labels, sd))]
            lines += ['  correlation of the mean covariance'] + ['    ' + '  '.join(f'{v:6.2f}' for v in row) for row in R]
            # the sign of the correlations changes with the orientation and cancels in the mean: per-molecule values
            Ci = torch.linalg.inv(F_.permute(2, 0, 1)).cpu().numpy(); sdi = np.sqrt(np.einsum('aii->ai', Ci))
            Ri = np.abs(Ci/np.einsum('ai,aj->aij', sdi, sdi))
            lines += ['  |correlation| of each molecule, median'] + ['    ' + '  '.join(f'{v:6.2f}' for v in row) for row in np.median(Ri, axis=0)] + ['']
            tl = ['$\\%s$' % l for l in labels]
            for row, (Mx, title, fmt) in enumerate([(Fm, 'mean Fisher (rad$^{-2}$)', '{:.0f}'), (np.median(Ri, axis=0), '|correlation| per molecule, median', '{:.2f}')]):
                a_ = ax[row, k]
                a_.imshow(Mx if row else np.log10(np.abs(Mx)+1e-12), cmap='viridis' if row == 0 else 'Reds', vmin=None if row == 0 else 0, vmax=None if row == 0 else 1)
                for (u, v), val in np.ndenumerate(Mx):
                    a_.text(v, u, fmt.format(val), ha='center', va='center', fontsize=5.5, color='w' if row == 0 or val > 0.6 else 'k')
                a_.set_xticks(range(len(labels)), tl); a_.set_yticks(range(len(labels)), tl)
                a_.tick_params(length=0, pad=1)
                a_.set_title((name + '\n' if row == 0 else '') + title, fontsize=6, pad=2)
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(out_folder, filename+'.png'), dpi=300)
        fig.savefig(os.path.join(out_folder, filename+'.svg'))
        plt.show()
    open(os.path.join(out_folder, filename+'.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    #plt.savefig("CRB_MC.svg", format="svg")
fisher_matrices(0., 'fisher_covariance_matrices')
fisher_matrices(bck_actin1, 'fisher_covariance_matrices_actin_background')

#%%
rho_list = np.linspace(0,90,20)
rho_plot = (eta_list[1:]+eta_list[:-1])/2

#%%
cr_rho = []
cr_eta = []
cr_delta = []
cr_rho_r = []
cr_eta_r = []
cr_delta_r = []
for i, rho_lim in tqdm(enumerate(rho_list[:-1])):
    RHO = torch.tensor(np.random.uniform(rho_lim, rho_list[i+1]), device=device)
    ETA = torch.tensor(np.arccos(np.random.uniform(np.cos(20*np.pi/180), np.cos(90*np.pi/180), NPSF)), device=device)
    #plt.hist(ETA.cpu().numpy()*180/np.pi, bins=30)
    #plt.show()
    DELTA = torch.tensor(80*np.pi/180+np.random.rand(NPSF)*(10*np.pi/180), device=device)#torch.tensor(0.1*np.random.rand(NPSF), device=device)# 
    N = torch.tensor((N_actin1*np.array([1 for iii in range(NPSF)])), device=device)#*(torch.sin(ETA)**2)
    bck = torch.tensor(float(bck_actin1), device=device)
    F = fisher(M, RHO, ETA, DELTA, N, bck, gain=gain_actin1, dark_var=dark_var_actin1)
    Fr = fisher_ratiometric(M, RHO, ETA, DELTA, N, bck, window=ratio_window, fixed_eta=ratio_fixed_eta, gain=gain_actin1, dark_var=dark_var_actin1)
    cr = torch.sqrt(crlb_cov(F))*(180/np.pi)
    crr = torch.sqrt(crlb_cov(Fr))*(180/np.pi)
    #print(cr)
    cr_rho.append(cr[0,0].cpu().numpy())
    cr_eta.append(cr[1,1].cpu().numpy())
    cr_delta.append(cr[2,2].cpu().numpy())
    #print(cr_rho, cr_eta, cr_delta)
    cr_rho_r.append(crr[0,0].cpu().numpy())
    cr_eta_r.append(crr[1,1].cpu().numpy() if crr.shape[0] == 3 else np.nan) # no eta bound when eta is fixed
    cr_delta_r.append(crr[-1,-1].cpu().numpy())
    #print(cr_rho_r, cr_eta_r, cr_delta_r)

#%%
cr_rho2 = cr_rho
cr_eta2 = cr_eta 
cr_delta2 = cr_delta
cr_rho_r2 = cr_rho_r
cr_eta_r2 = cr_eta_r
cr_delta_r2 = cr_delta_r

#%%
cr_rho3 = cr_rho
cr_eta3 = cr_eta 
cr_delta3 = cr_delta
cr_rho_r3 = cr_rho_r
cr_eta_r3 = cr_eta_r
cr_delta_r3 = cr_delta_r

#%%
plt.rcParams['figure.figsize'] = [15,3]
fig, ax = plt.subplots(1,3)
ax[0].plot(rho_plot, cr_rho, c='b', label='$\\delta=0$, b=0')
ax[0].plot(rho_plot, cr_rho2, c='b', alpha=0.5, label='$\\delta=0$, b=50')
ax[0].plot(rho_plot, cr_rho3, c='r', alpha=0.8, label='$\\delta \\in$ [40,150], b=50')
ax[0].legend()
#ax[0].set_yscale('log')
ax[0].set_xlabel('$\\eta$ ($\\degree$)')
ax[0].set_ylabel('CRLB$_{\\rho}$ ($\\degree$)')
ax[0].set_ylim((0,22))
ax[0].grid()
ax[1].plot(rho_plot, cr_eta, c='b')
ax[1].plot(rho_plot, cr_eta2, c='b', alpha=0.5)
ax[1].plot(rho_plot, cr_eta3, c='r', alpha=0.8)
#ax[1].axhline(3.6)
#ax[1].set_yscale('log')
ax[1].set_xlabel('$\\eta$ ($\\degree$)')
ax[1].set_ylabel('CRLB$_{\\eta}$ ($\\degree$)')
ax[1].set_ylim((0,22))
ax[1].grid()
#ax[2].plot(eta_plot, cr_delta, c='b')
#ax[2].plot(eta_plot, cr_delta2, c='b', alpha=0.5)
ax[2].plot(rho_plot, cr_delta3, c='r', alpha=0.8)
#ax[2].set_yscale('log')
ax[2].set_xlabel('$\\eta$ ($\\degree$)')
ax[2].set_ylabel('CRLB$_{\\delta}$ ($\\degree$)')
ax[2].set_ylim((0,22))
ax[2].grid()
