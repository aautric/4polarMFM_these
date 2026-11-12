import os
# finite-difference check of fisher() of fisher_4pMFM.py (worktree simulation-review), set-up of crb_vs_unbiased_sim.py
import sys, numpy as np, torch
sys.path.insert(0, '/mnt/c/Users/Amaury/Documents/GitHub/4polarMFM_these_simulation_review')
from simu_PSF_polarMFM import *
from fisher_4pMFM import fisher
device = 'cpu'; np.random.seed(1)
NA = torch.tensor(1.4); mag = torch.tensor(100); lambd = torch.tensor(617); f_tube = torch.tensor(200)
l_pixel = torch.tensor(16); MAG = torch.tensor(200/150)
zernike_coefs_x = torch.zeros((3, 15)); polar_projections = np.array([0, 45, 0]); J = np.array([[1, 0], [0, 1]])
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../simu_precision/crb_vs_unbiased_sim.py')).read().split('def build_M')[1].join(['def build_M', '']).split('BCK_CRB')[0]) if False else None
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../simu_precision/crb_vs_unbiased_sim.py')).read()
exec('def build_M' + src.split('def build_M')[1].split('BCK_CRB = ')[0])
n = 6; planes = np.array([0.646, 1., 1.354])
M = build_M(xp=torch.tensor(np.random.uniform(-0.2, 0.2, n)), yp=torch.tensor(np.random.uniform(-0.2, 0.2, n)),
            zp=torch.tensor(np.random.uniform(0.2, 1.2, n)), d_=torch.tensor(-planes[1]*np.ones(n)),
            second_plane_=torch.tensor(planes[1] - planes), SAF_=True)
RHO = torch.tensor(np.random.rand(n)*np.pi); ETA = torch.tensor(np.radians([10, 30, 60, 90, 120, 160.]))
DELTA = torch.tensor(np.radians(np.random.uniform(60, 120, n))); Np = torch.tensor(2100.*np.ones(n))
for bck in (0., 25.):
    F = fisher(M, RHO, ETA, DELTA, Np, torch.tensor(bck), device=device)
    psf = PSF(rho=RHO*180/np.pi, eta=ETA*180/np.pi, delta=DELTA*180/np.pi, M=M, N_photons=Np)
    NN = psf.shape[3]; w = slice(NN//2-6, NN//2+7) # 13 x 13 centred, the default window of fisher()
    h = 1e-3; grads = []
    for k in range(3):
        p = [RHO.clone(), ETA.clone(), DELTA.clone()]; m = [RHO.clone(), ETA.clone(), DELTA.clone()]
        p[k] = p[k] + h; m[k] = m[k] - h
        fp = PSF(rho=p[0]*180/np.pi, eta=p[1]*180/np.pi, delta=p[2]*180/np.pi, M=M, N_photons=Np).double()
        fm = PSF(rho=m[0]*180/np.pi, eta=m[1]*180/np.pi, delta=m[2]*180/np.pi, M=M, N_photons=Np).double()
        grads.append(((fp - fm)/(2*h))[..., w, w])
    G = torch.stack(grads); s = psf.double()[..., w, w]
    Ffd = torch.einsum('uapbcd,vapbcd->uva', G, G/(s + bck)[None])
    if bck == 0:
        print('psf shape', tuple(psf.shape), 'window', w, '(13 x 13); pixels with psf == 0 in the window:', int((s == 0).sum()))
        tot = psf.double().sum((1,2,3,4)); win = s.sum((1,2,3,4)); print('fraction of N in the window:', np.round((win/tot).numpy(), 3))
    rel = (F.double() - Ffd).abs()/Ffd.abs().amax((0, 1))[None, None]
    print(f'background {bck}: max relative difference fisher vs finite differences (per PSF, / largest element):', np.round(rel.amax((0, 1)).numpy(), 4))
    crb = torch.sqrt(torch.diagonal(torch.linalg.inv(F.double().permute(2, 0, 1)), dim1=1, dim2=2))*180/np.pi
    crbfd = torch.sqrt(torch.diagonal(torch.linalg.inv(Ffd.permute(2, 0, 1)), dim1=1, dim2=2))*180/np.pi
    print('  eta', np.degrees(ETA.numpy()).round(), '\n  CRB fisher (rho, eta, delta deg)\n', crb.numpy().round(2), '\n  CRB finite diff\n', crbfd.numpy().round(2))
