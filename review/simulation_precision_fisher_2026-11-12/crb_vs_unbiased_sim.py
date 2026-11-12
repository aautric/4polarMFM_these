# 2026-11-12 (Claude): CRB vs eta (Fisher code of the topic mfm_fisher: fisher() of fisher_4pMFM.py, set-up and build_M
# of MFM_fisher_claude.py section 'WRT to eta') for the conditions of the unbiased simulations of 2026-11-10
# (D:\Amaury\simu_precision_claude\unbiased_2026-11-09): N 2100 photons per molecule, background 25 photons/pixel,
# delta uniform 60-120 deg, rho uniform, x, y in [-0.2, 0.2] um, z in [0.2, 1.2] um, planes 0.646 / 1 / 1.354 um, SAF on,
# ideal polarisation (J identity of the torch model, polar_projections 0/-45/0 = J2 rot(45) of the simulation, review item 18).
# CRB: Poisson noise of the molecule only, NO background (gain 1, V 0, bck 0), as the CRB of the topic (CRB_vs_SGD_eta.png)
# (user, 2026-11-12); the simulations have background 25. Compared with the simulated precision of
# nobg_poisson (Poisson only) and nobg_emccd (Poisson + EM excess + read noise 0.4), both WITHOUT background (2026-11-12). CRB on (rho, eta, delta) only, x, y, z, N known
# (optimistic), mean over the molecules of the inverse Fisher matrix (crlb_cov), per eta.
# Simulated precision per eta: sqrt(mean over the 20 rho of the std over the 80 molecules) (as CRB_vs_SGD_eta.png).
# Figures: crb_vs_unbiased_sgd.png (SGD_jax.py), crb_vs_unbiased_cnn.png (CNN + LM); crb_vs_unbiased_sim.npz / .txt.
# CPU (the GPU queue is not touched).
import os, sys, glob, json, hashlib
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/mnt/c/Users/Amaury/Documents/GitHub/4polarMFM_these')
from simu_PSF_polarMFM import *
from fisher_4pMFM import fisher
device = 'cpu'; torch.manual_seed(0); np.random.seed(0)
O = '/mnt/d/AMAURY/simu_precision_claude/MFM_fisher/'
SIM = '/mnt/d/Amaury/simu_precision_claude/unbiased_2026-11-09/'
NPSF = 400; QE = 0.92
eta_crb = np.linspace(1, 179, 45); delta_crb = (60, 120)
COND = {'Poisson': dict(gain=1., dark_var=0., case='crb')}
CASES = {'nobg_poisson': 'Poisson, no background', 'nobg_emccd': 'Poisson + EM excess + read noise, no background'}  # runs of run_2026-11-12_nobg.sh
NA = torch.tensor(1.4); mag = torch.tensor(100); lambd = torch.tensor(617); f_tube = torch.tensor(200)
l_pixel = torch.tensor(16); MAG = torch.tensor(200/150)
zernike_coefs_x = torch.zeros((3, 15)); polar_projections = np.array([0, -45, 0]); J = np.array([[1, 0], [0, 1]])

def crlb_cov(F):
    return torch.mean(torch.linalg.inv(F.permute(2, 0, 1)), dim=0)

def build_M(xp, yp, zp, d_, second_plane_, SAF_):  # copy of MFM_fisher_claude.py
    N_bfp = torch.tensor(80, device=device)
    x_, y_, th1_, phi_, [Ex0_, Ex1_, Ex2_], [Ey0_, Ey1_, Ey2_], r_, r_cut_, r_cut_saf_, k_, f_o_, costh2_ = vectorial_BFP_perfect_focus(N_bfp, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, SAF=SAF_, device=device, J_dichroic=J)
    pm = torch.stack([torch.ones((N_bfp, N_bfp), device=device)]*3)
    zb = generate_zernike_base(r_cut=r_cut_, N=N_bfp, zernike_order=4, device=device)
    _, _, npad = padding(r_, r_cut_, k_, f_o_, N=N_bfp, l_pixel=l_pixel, NA=NA, mag=mag, lambd=lambd, f_tube=f_tube, MAG=MAG, device=device)
    th1_, phi_, Ex0_, Ex1_, Ex2_, Ey0_, Ey1_, Ey2_, pm, zb = [pad(t, npad) for t in (th1_, phi_, Ex0_, Ex1_, Ex2_, Ey0_, Ey1_, Ey2_, pm, zb)]
    costh2_ = pad(costh2_, npad)
    return compute_M(xp=xp, yp=yp, zp=zp, d=d_, x=x_, y=y_, th1=th1_, phi=phi_, Ex0=Ex0_, Ex1=Ex1_, Ex2=Ex2_,
                     Ey0=Ey0_, Ey1=Ey1_, Ey2=Ey2_, r=r_, r_cut=r_cut_, k=k_, f_o=f_o_,
                     phase_maskx=pm, phase_masky=pm, zernike_base=zb, zernike_coefs_x=zernike_coefs_x,
                     zernike_coefs_y=zernike_coefs_x, second_plane=second_plane_, polar_projections=polar_projections, lambd=lambd, f_tube=f_tube,
                     BFP_version=False, SAF=SAF_, costh2=costh2_)

BCK_CRB = 0.
params = dict(N=2100, bck=BCK_CRB, delta=delta_crb, eta=eta_crb.tolist(), NPSF=NPSF, cond={k: [v['gain'], v['dark_var']] for k, v in COND.items()},
              geometry=dict(xy=0.2, z=(0.2, 1.2), planes=[0.646, 1., 1.354], SAF=True))
cache = O + 'crb_unbiased_' + hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8] + '.npz'
if os.path.exists(cache):
    crb = dict(np.load(cache)); print('CRB loaded from', cache)
else:
    planes = np.array([0.646, 1., 1.354])
    M = build_M(xp=torch.tensor(np.random.uniform(-0.2, 0.2, NPSF)), yp=torch.tensor(np.random.uniform(-0.2, 0.2, NPSF)),
                zp=torch.tensor(np.random.uniform(0.2, 1.2, NPSF)), d_=torch.tensor(-planes[1]*np.ones(NPSF)),
                second_plane_=torch.tensor(planes[1] - planes), SAF_=True)
    crb = {}
    for name, c in COND.items():
        out = []
        for e in eta_crb:
            RHO = torch.tensor(np.random.rand(NPSF)*np.pi); ETA = torch.tensor(np.radians(e)*np.ones(NPSF))
            DELTA = torch.tensor(np.radians(np.random.uniform(*delta_crb, NPSF)))
            F = fisher(M, RHO, ETA, DELTA, torch.tensor(2100.*np.ones(NPSF)), torch.tensor(BCK_CRB), device=device, gain=c['gain'], dark_var=c['dark_var'])
            out.append((torch.sqrt(torch.diagonal(crlb_cov(F)))*180/np.pi).numpy())
        crb[c['case']] = np.array(out); print(name, 'CRB at eta 90 (rho, eta, delta):', np.round(crb[c['case']][22], 2), flush=True)
    np.savez(cache, params=json.dumps(params), **crb)

def sim_prec(case, meth):
    run = [f for f in sorted(glob.glob(f'{SIM}2026-11-1*_{case}_{meth}')) if os.path.exists(f + '/results_huber.npz')][-1]
    r = np.load(run + '/results_huber.npz'); el = np.unique(r['true_eta']); rl = np.unique(r['true_rho'])
    p = np.array([[np.sqrt(np.mean([np.nanstd(err[(r['true_eta'] == e) & (r['true_rho'] == q)])**2 for q in rl]))
                   for err in (r['error_rho'], r['error_eta'], r['error_delta'])] for e in el])
    return el, p

lines = ['CRB Poisson only, no background (N 2100, delta 60-120, ideal J; rho, eta, delta free, x y z N known) vs simulated precision '
         '(sqrt(mean over rho of std^2)), deg, at eta ~45 / ~90:']
COL = {'nobg_poisson': 'tab:green', 'nobg_emccd': 'tab:orange'}
for meth, lab, fn in (('sgd', 'SGD_jax.py', 'crb_vs_unbiased_sgd.png'), ('cnn', 'CNN + LM (SGD_jax_CNN.py)', 'crb_vs_unbiased_cnn.png')):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    cr = crb['crb']
    for j in range(3):
        ax[j].plot(eta_crb, cr[:, j], 'k-', label='CRB, Poisson only, no background')
    for case, name in CASES.items():
        if not [f for f in glob.glob(f'{SIM}2026-11-1*_{case}_{meth}') if os.path.exists(f + '/results_huber.npz')]:
            continue
        el, p = sim_prec(case, meth)
        for j in range(3):
            ax[j].plot(el, p[:, j], 'o', ms=4, color=COL[case], label=f'{lab}, {name}')
        for e in (45, 90):
            i = np.argmin(np.abs(el - e)); k = np.argmin(np.abs(eta_crb - e))
            lines.append(f'{lab:26s} {name:34s} eta {el[i]:5.1f}: sim ' + ' '.join(f'{v:5.1f}' for v in p[i]) +
                         ' | CRB ' + ' '.join(f'{v:5.1f}' for v in cr[k]) + ' | ratio ' + ' '.join(f'{a/b:4.2f}' for a, b in zip(p[i], cr[k])))
    for j, par in enumerate(('rho', 'eta', 'delta')):
        ax[j].set_title(f'{par} precision (deg)'); ax[j].set_xlabel('true eta (deg)'); ax[j].grid(alpha=.3)
        ax[j].set_ylim(0, {0: 30, 1: 25, 2: 40}[j])
    ax[0].legend(fontsize=7)
    fig.suptitle(f'{lab}: precision of the unbiased simulation (points) vs CRB (line: Poisson only, no background); N 2100, delta 60-120, ideal J; simulations without background', fontsize=10)
    fig.tight_layout(); fig.savefig(O + fn, dpi=110); plt.close(fig)
open(O + 'crb_vs_unbiased_sim.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
