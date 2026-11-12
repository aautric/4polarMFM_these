import os
import sys, numpy as np, torch
sys.path.insert(0, '/mnt/c/Users/Amaury/Documents/GitHub/4polarMFM_these_simulation_review')
from simu_PSF_polarMFM import *
from fisher_4pMFM import fisher
device = 'cpu'; NPSF = 200
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../simu_precision/crb_vs_unbiased_sim.py')).read()
planes = np.array([0.646, 1., 1.354])
out = {}
for lam, proj in ((617, 45), (620, 45), (620, -45)):
    g = globals()
    g.update(NA=torch.tensor(1.4), mag=torch.tensor(100), lambd=torch.tensor(lam), f_tube=torch.tensor(200), l_pixel=torch.tensor(16),
             MAG=torch.tensor(200/150), zernike_coefs_x=torch.zeros((3, 15)), polar_projections=np.array([0, proj, 0]), J=np.eye(2))
    exec('def build_M' + src.split('def build_M')[1].split('BCK_CRB = ')[0], g)
    np.random.seed(0); torch.manual_seed(0)
    M = build_M(xp=torch.tensor(np.random.uniform(-0.2, 0.2, NPSF)), yp=torch.tensor(np.random.uniform(-0.2, 0.2, NPSF)),
                zp=torch.tensor(np.random.uniform(0.2, 1.2, NPSF)), d_=torch.tensor(-planes[1]*np.ones(NPSF)),
                second_plane_=torch.tensor(planes[1] - planes), SAF_=True)
    res = []
    for e in (20, 45, 90, 135):
        RHO = torch.tensor(np.random.rand(NPSF)*np.pi); ETA = torch.tensor(np.radians(e)*np.ones(NPSF))
        DELTA = torch.tensor(np.radians(np.random.uniform(60, 120, NPSF)))
        F = fisher(M, RHO, ETA, DELTA, torch.tensor(2100.*np.ones(NPSF)), torch.tensor(0.), device=device)
        res.append((torch.sqrt(torch.diagonal(torch.mean(torch.linalg.inv(F.permute(2, 0, 1)), 0)))*180/np.pi).numpy())
    out[(lam, proj)] = np.array(res)
    print(f'lambda {lam}, middle plane {proj:+d}: CRB (rho, eta, delta) deg at eta 20 / 45 / 90 / 135:', ' | '.join(' '.join(f'{v:.2f}' for v in r) for r in res), flush=True)
