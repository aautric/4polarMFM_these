# 2026-11-12 (Claude): CRB vs the simulated precision of any runs of SGD_precision.py (replaces crb_vs_unbiased_sim.py):
#   python simu_precision/crb_vs_simulation.py <run folder> [<run folder> ...] --out <folder> [--npsf 400]
# CRB (fisher() of fisher_4pMFM.py, torch model simu_PSF_polarMFM.py, window 13 x 13 as the fit): Poisson noise of the
# molecule only, NO background (user, 2026-11-12), on (rho, eta, delta) only, x, y, z, N known (optimistic); mean over
# the molecules of the inverse Fisher matrix (crlb_cov), per eta (45 values 1-179 deg), rho uniform. The conditions come
# from each run: N, delta, x, y, z = a random subset of the true values of its simulation.npz; optics (lambda, NA,
# magnification, pixel, planes, SAF) from NPZ_<loss>\config.txt. Only the ideal polarisation of the simulations
# (J1 identity, J2 rot(45), rotations 0): torch polar_projections [0, -45, 0] (review item 18); else the run is skipped.
# One CRB per set of conditions, cached in --out (crb_<hash>.npz).
# Simulated precision per true eta: sqrt(mean over the rho of the grid of std^2): plain std (filled points, the CRB
# bounds the variance, outliers included) and robust std (1.4826 MAD, open points); outliers = % of the PSF with
# |error - median| > 5 robust std of their grid point (legend and text). CRB interpolated at the eta of the grid.
# Dashed lines: CRB x sqrt(2/QE), the bound of an ideal EMCCD (EM excess factor 2, QE of the run; no read noise), the
# reference for the runs with Poisson + read + excess (review item 21).
# Outputs in --out: crb_vs_simulation_<method>.png (method = sgd: SGD_jax.py, cnn: CNN + LM), crb_vs_simulation.txt.
# Runs on the CPU.
import os, sys, glob, json, hashlib, argparse
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root of the repo this script is in
from simu_PSF_polarMFM import *
from fisher_4pMFM import fisher
device = 'cpu'
ETA_CRB = np.linspace(1, 179, 45)
# optics of the torch model (module level: build_M below and the review scripts use them); set per run in crb()
NA = torch.tensor(1.4); mag = torch.tensor(100); lambd = torch.tensor(620); f_tube = torch.tensor(200)
l_pixel = torch.tensor(16); MAG = torch.tensor(200/150)
zernike_coefs_x = torch.zeros((3, 15)); polar_projections = np.array([0, -45, 0]); J = np.array([[1, 0], [0, 1]])

def crlb_cov(F):
    # Purpose: CRB covariance = mean over the molecules of the inverse of each Fisher matrix (F: 3, 3, NPSF)
    return torch.mean(torch.linalg.inv(F.permute(2, 0, 1)), dim=0)

def build_M(xp, yp, zp, d_, second_plane_, SAF_):  # as MFM_fisher.py
    # Purpose: basis images M of the torch model for molecules at xp, yp, zp (um), planes at -d_ + second_plane_
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

BCK_CRB = 0.  # CRB without background (end of the part read by the review scripts)


def read_config(path):
    cfg = {}
    for line in open(path):
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); cfg[k.strip()] = v.strip()
    return cfg


def run_info(run):
    # Purpose: what the CRB and the comparison need from a run folder: loss, method, optics, truth
    res = sorted(run.glob('results_*.npz'))[0]; loss = res.stem[len('results_'):]
    cfg = read_config(run / f'NPZ_{loss}' / 'config.txt')
    npz = sorted(run.glob(f'NPZ_{loss}/*.npz'))
    cnn = 'use_cnn' in np.load(npz[0]).files and bool(np.load(npz[0])['use_cnn'])
    J1, J2 = np.array(eval(cfg['J1'])), np.array(eval(cfg['J2']))
    ideal = (np.allclose(J1, np.eye(2)) and np.allclose(J2, [[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
             and float(cfg['rotation']) == 0 and float(cfg['rotation2']) == 0)
    return dict(run=run, res=res, cfg=cfg, method='cnn' if cnn else 'sgd', ideal=ideal, label=run.name.split('_', 2)[-1])


def crb(info, npsf, out):
    # Purpose: CRB vs eta for the conditions of one run (cached in out)
    global NA, mag, lambd, f_tube, l_pixel, MAG
    cfg = info['cfg']; sim = np.load(info['run'] / 'simulation.npz')
    rng = np.random.default_rng(0); k = rng.choice(len(sim['true_N']), npsf, replace=False)  # molecules of the run
    mp, ip = float(cfg['middle_plane']), float(cfg['interplane']); planes = np.array([mp - ip, mp, mp + ip])
    opt = {key: float(eval(cfg[key])) for key in ('lambda_emission', 'NA', 'mag', 'MAG', 'l_pixel', 'f_tube')}
    key = dict(opt, planes=planes.round(4).tolist(), SAF=cfg['SAF'], N=sim['true_N'][k].tolist(), delta=sim['true_delta'][k].round(3).tolist(),
               x=sim['true_x'][k].round(4).tolist(), y=sim['true_y'][k].round(4).tolist(), z=sim['true_z'][k].round(4).tolist(),
               eta=ETA_CRB.tolist(), bck=BCK_CRB, window=13, polar_projections=[0, -45, 0])
    cache = out / ('crb_' + hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8] + '.npz')
    if cache.exists():
        return np.load(cache)['crb']
    NA, mag, lambd, f_tube = torch.tensor(opt['NA']), torch.tensor(opt['mag']), torch.tensor(opt['lambda_emission']), torch.tensor(opt['f_tube'])
    l_pixel, MAG = torch.tensor(opt['l_pixel']), torch.tensor(opt['MAG'])
    M = build_M(xp=torch.tensor(sim['true_x'][k]), yp=torch.tensor(sim['true_y'][k]), zp=torch.tensor(sim['true_z'][k]),
                d_=torch.tensor(-mp*np.ones(npsf)), second_plane_=torch.tensor(mp - planes), SAF_=cfg['SAF'] == 'True')
    res = []
    for e in ETA_CRB:
        RHO = torch.tensor(rng.random(npsf)*np.pi); ETA = torch.tensor(np.radians(e)*np.ones(npsf))
        DELTA = torch.tensor(np.radians(sim['true_delta'][k]))
        F = fisher(M, RHO, ETA, DELTA, torch.tensor(sim['true_N'][k].astype(float)), torch.tensor(BCK_CRB), device=device)
        res.append((torch.sqrt(torch.diagonal(crlb_cov(F)))*180/np.pi).numpy())
    res = np.array(res); np.savez(cache, crb=res, key=json.dumps(key)); print('CRB computed:', cache, flush=True)
    return res


def sim_prec(res):
    # Purpose: per true eta, sqrt(mean over the rho of the grid of std^2), plain and robust std, and % of outliers
    r = np.load(res); el = np.unique(r['true_eta']); rl = np.unique(r['true_rho'])
    plain, robust, outl = (np.zeros((len(el), 3)) for _ in range(3))
    for i, e in enumerate(el):
        for j, err in enumerate((r['error_rho'], r['error_eta'], r['error_delta'])):
            sd, rs, no, n = [], [], 0, 0
            for q in rl:
                v = err[(r['true_eta'] == e) & (r['true_rho'] == q)]; v = v[np.isfinite(v)]
                if not v.size: continue
                m = np.median(v); mad = 1.4826*np.median(np.abs(v - m))
                sd.append(np.std(v)); rs.append(mad); no += np.sum(np.abs(v - m) > 5*mad); n += len(v)
            plain[i, j], robust[i, j], outl[i, j] = np.sqrt(np.mean(np.square(sd))), np.sqrt(np.mean(np.square(rs))), 100*no/max(n, 1)
    return el, plain, robust, outl


def main(argv):
    p = argparse.ArgumentParser(description='CRB (Poisson only, no background) vs simulated precision of runs of SGD_precision.py')
    p.add_argument('runs', nargs='+'); p.add_argument('--out', required=True, help='folder of the figures, text and CRB cache')
    p.add_argument('--npsf', type=int, default=400, help='molecules of the run used for the CRB')
    a = p.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    infos = [run_info(Path(r)) for r in a.runs]
    for i in infos:
        if not i['ideal']:
            print(f'{i["run"].name}: not the ideal polarisation (J1 identity, J2 rot(45), rotations 0): skipped')
    infos = [i for i in infos if i['ideal']]
    lines = ['CRB Poisson only, no background (13 x 13 window; rho, eta, delta free, x y z N known; conditions of each run) vs '
             'simulated precision (sqrt(mean over rho of std^2)), deg; CRB interpolated at the eta of the grid.',
             'columns: rho eta delta | plain std | robust std | CRB | ratio plain / CRB | outliers (% |error - median| > 5 robust std)']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for meth, lab in (('sgd', 'SGD_jax.py'), ('cnn', 'CNN + LM (SGD_jax_CNN.py)')):
        sel = [i for i in infos if i['method'] == meth]
        if not sel: continue
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
        for n, i in enumerate(sel):
            cr = crb(i, a.npsf, out); col = colors[n % len(colors)]
            emccd = np.sqrt(2/float(i['cfg']['QE']))  # EMCCD reference factor, QE of the run
            el, pl, pr, ou = sim_prec(i['res'])
            cri = np.stack([np.interp(el, ETA_CRB, cr[:, j]) for j in range(3)], axis=1)
            for j in range(3):
                ax[j].plot(ETA_CRB, cr[:, j], '-', color=col, lw=1, label=f'CRB, conditions of {i["label"]}')
                # reference for an EMCCD (review item 21, user 2026-11-12): the EM excess multiplies the variance by
                # 2/QE, so the bound of an ideal EMCCD (no read noise) is CRB x sqrt(2/QE)
                ax[j].plot(ETA_CRB, cr[:, j]*emccd, '--', color=col, lw=0.8, label=f'CRB x sqrt(2/QE) = x{emccd:.2f} (EMCCD excess)')
                ax[j].plot(el, pl[:, j], 'o', ms=4, color=col, label=f'{i["label"]}: std (outliers {np.mean(ou[:, j]):.1f} %)')
                ax[j].plot(el, pr[:, j], 'o', ms=4, mfc='none', color=col, alpha=.7, label=f'{i["label"]}: robust std')
            for e in (45, 90):
                k = np.argmin(np.abs(el - e))
                lines.append(f'{lab:26s} {i["label"]:24s} eta {el[k]:5.1f}: sim ' + ' '.join(f'{v:5.1f}' for v in pl[k]) +
                             ' | robust ' + ' '.join(f'{v:5.1f}' for v in pr[k]) + ' | CRB ' + ' '.join(f'{v:5.1f}' for v in cri[k]) +
                             ' | ratio ' + ' '.join(f'{x/y:4.2f}' for x, y in zip(pl[k], cri[k])) + ' | outliers ' + ' '.join(f'{v:4.1f}' for v in ou[k]) + ' %')
            lines.append(f'{lab:26s} {i["label"]:24s} EMCCD reference: CRB x sqrt(2/QE) = CRB x {emccd:.3f}')
            mid = (el > 20) & (el < 160)
            lines.append(f'{lab:26s} {i["label"]:24s} all eta: outliers ' + ' '.join(f'{v:4.1f}' for v in ou.mean(0)) +
                         ' %, median ratio plain / CRB (eta 20-160) ' + ' '.join(f'{np.median((pl/cri)[mid, j]):4.2f}' for j in range(3)))
        for j, par in enumerate(('rho', 'eta', 'delta')):
            ax[j].set_title(f'{par} precision (deg)'); ax[j].set_xlabel('true eta (deg)'); ax[j].grid(alpha=.3)
            ax[j].set_ylim(0, {0: 30, 1: 25, 2: 40}[j]); ax[j].legend(fontsize=5.5)
        fig.suptitle(f'{lab}: simulated precision (filled: std, open: robust std; outliers = |error - median| > 5 robust std)\n'
                     'vs CRB (lines: Poisson only, no background, conditions of each run)', fontsize=9.5)
        fig.tight_layout(); fig.savefig(out / f'crb_vs_simulation_{meth}.png', dpi=110); plt.close(fig)
    (out / 'crb_vs_simulation.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:])
