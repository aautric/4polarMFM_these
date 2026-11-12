# 2026-11-12 (Claude): table SGD_jax.py vs SGD_jax_CNN.py (CNN start + per-PSF LM) on the same simulated PSF (seed 0) of
# the unbiased simulations (SGD_precision_claude.py, N 2100, true delta uniform 60-120 per PSF, middle_plane 1.0, huber,
# 20 x 20 (rho, eta) grid x 80 PSF, ideal J, J2 rot(45)). The 3 noise types of the user:
#   nobg_poisson  Poisson (no background, perfect detector)
#   nobg_emccd    Poisson + read + excess (EMCCD QE 0.92, EM 200, read noise 0.4 photons), no background
#   case2_emccd   Poisson + read + excess + background 25 photons/pixel (with its Poisson noise)
# Per grid point: bias = median error, precision = robust std (as the maps); then the median over the 400 points.
# Columns: |bias| / precision of rho, eta, delta (deg), z, x (nm), N (relative); slope of found vs true delta and its
# value at true 90 minus 90; % of PSF where this method has the lower deviance; same truth in both runs.
# Output: compare_sgd_cnn.txt in the analysis folder of the runs. (Figures removed 2026-11-12, user: table only;
# maps: polar_maps.py; CRB: crb_vs_unbiased_sim.py.)
import glob, numpy as np, pandas as pd
B = '/mnt/d/Amaury/simu_precision_claude/unbiased_2026-11-09/'; O = B + 'analysis/'  # runs and outputs (D:); folder renamed 2026-11-09_unbiased_sgd_vs_cnn after the merge
CASES = [('nobg_poisson', 'Poisson'), ('nobg_emccd', 'Poisson + read + excess'), ('case2_emccd', 'Poisson + read + excess + bg 25')]
METH = [('sgd', 'SGD_jax.py'), ('cnn', 'CNN + LM')]
PAR = [('rho', 'deg'), ('eta', 'deg'), ('delta', 'deg'), ('z', 'nm'), ('x', 'nm'), ('N', 'relative')]
folder = lambda c, m: [f for f in sorted(glob.glob(f'{B}2026-11-1*_{c}_{m}')) if 'CRASHED' not in f][-1]  # not the *_lossN test
S, R = {}, {}
for c, _ in CASES:
    for m, _ in METH:
        f = folder(c, m); S[c, m] = pd.read_csv(f + '/summary.csv'); R[c, m] = dict(np.load(f + '/results_huber.npz'))
lines = ['Unbiased simulations SGD_jax.py vs CNN + LM, same PSF (32000 per case, N 2100, delta 60-120, huber)',
         'per grid point (rho, eta): bias = median error, precision = robust std (1.4826 MAD); then median over the 400 points', '']
hdr = f'{"case":32s} {"method":10s}' + ''.join(f' {p:>14s}' for p, _ in PAR) + '   delta slope/offset   dev<other  paired'
lines += ['|bias| / precision (median over the grid points):', hdr]
for c, cl in CASES:
    for m, ml in METH:
        s = S[c, m]; r = R[c, m]
        row = f'{cl:32s} {ml:10s}' + ''.join(f' {np.median(np.abs(s["bias_"+p])):6.2f}/{np.median(s["sigma_"+p]):6.2f}' for p, _ in PAR)
        p1 = np.polyfit(r['true_delta'], r['found_delta'], 1)
        o = R[c, 'cnn' if m == 'sgd' else 'sgd']; same = np.allclose(r['true_rho'], o['true_rho'])
        better = np.mean(r['found_deviance'] < o['found_deviance'])*100
        lines.append(row + f'   {p1[0]:.3f} / {np.polyval(p1, 90) - 90:+5.1f} at 90   {better:5.1f} %   {same}')
lines.append('(N bias/precision relative; delta slope of found vs true over all PSF and its value at true 90 minus 90; '
             'dev<other: % of PSF where this method has the lower deviance; paired: same true values in both runs)')
lines += ['', 'paired deviance (same PSF): SGD - CNN']
for c, cl in CASES:
    dd = R[c, 'sgd']['found_deviance'] - R[c, 'cnn']['found_deviance']
    big = dd > 0.01
    lines.append(f'{cl:32s} median {np.median(dd):+.5f}, CNN lower on {np.mean(dd > 0)*100:.1f} %, SGD worse by > 0.01 on {big.mean()*100:.1f} % '
                 f'(their |delta error| SGD {np.median(np.abs(R[c, "sgd"]["error_delta"][big])):.1f} vs CNN {np.median(np.abs(R[c, "cnn"]["error_delta"][big])):.1f} deg)')
open(O + 'compare_sgd_cnn.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
