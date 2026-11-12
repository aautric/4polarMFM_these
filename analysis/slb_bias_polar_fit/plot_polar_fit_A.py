# 2026-11-10 (Claude): fit of method A of the last polarisation calibration (2026-11-04, rotating polariser, 37 angles):
# intensity of the 6 channels (lamp-normalised, data_AB.npz of calibration_polar_claude.py) vs the polariser angle with
# the fit I = k_i (u + v cos 2theta + w sin 2theta), residuals in units of the error, and the per-image factor k_i.
import os, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # root of the 4polarMFM_these repo
OUT = os.environ.get('OUT_DIR', os.getcwd()).rstrip('/') + '/'  # outputs: $OUT_DIR, else the current folder
os.makedirs(OUT, exist_ok=True)
import calibration_polar_claude as cp
P = '/mnt/d/Amaury/simu_precision_claude/2026-11-04_polar_calib/'
d = np.load(P + 'data_AB.npz')
A = dict(theta=d['A_theta'], I=d['A_I'], E=d['A_E'])
f = cp.fit_A(A)
th = np.degrees(A['theta']); tt = np.linspace(th.min(), th.max(), 400); Tt = np.radians(tt)
names = ['plane 0 x', 'plane 0 y', 'plane 1 x', 'plane 1 y', 'plane 2 x', 'plane 2 y']  # order of data_AB.npz (as the fit txt)
fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2, 1]))
for c in range(6):
    col = plt.cm.tab10(c)
    u, v, w = f['coef'][:, c]
    ax[0].errorbar(th, A['I'][:, c]/f['k'], A['E'][:, c]/f['k'], fmt='o', ms=3, color=col)
    norm, V, psi = f['resp'][c]
    ax[0].plot(tt, u + v*np.cos(2*Tt) + w*np.sin(2*Tt), '-', color=col, lw=1,
               label=f'{names[c]}: V {V:.3f}, psi {np.degrees(psi):+.1f} deg')
    ax[1].plot(th, f['res'][:, c], 'o-', ms=3, lw=0.6, color=col)
ax[0].set_ylabel('intensity / lamp / k_i'); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title(f'Method A, 2026-11-04 (37 polariser angles): I = k_i (u + v cos 2θ + w sin 2θ) per channel, '
                f'χ²/n = {f["chi2"]:.1f}', fontsize=10)
ax[1].axhline(0, color='0.5', lw=0.8); ax[1].set_ylabel('residual / error')
ax[2].plot(th, f['k'], 'o-', ms=3, color='k'); ax[2].set_ylabel('k_i (image factor)'); ax[2].set_xlabel('polariser angle θ (deg, scale of the mount)')
fig.tight_layout(); fig.savefig(OUT + 'polar_calibration_2026-11-04_methodA_fit.png', dpi=110)
print('saved; chi2', f['chi2'], 'theta range', th.min(), th.max())
