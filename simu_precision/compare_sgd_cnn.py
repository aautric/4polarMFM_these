# 2026-11-12: its figures (vs_eta, delta_found_vs_true, errors_vs_true_delta, paired_deviance) were removed at the user request (too complicated); only compare_sgd_cnn.txt is kept. Maps: polar_maps_cnn_start.py
# 2026-11-12 (Claude): analysis of the unbiased simulations of 2026-11-09/10 (run_overnight_gpu.sh, SGD_precision_claude.py):
# same simulated PSF (seed 0) fitted by SGD_jax.py ('sgd') and SGD_jax_CNN.py (CNN start + per-PSF LM, 'cnn').
# N 2100, background 25, true delta uniform 60-120 per PSF, middle_plane 1.0, huber, 20 x 20 (rho, eta) grid x 80 PSF.
# case1 Poisson only; case2 + EM excess noise + read noise 0.4; case3 = case2 + background fluctuation sigma_b 3.
# Outputs (this folder): vs_eta.png (bias and precision vs true eta, median over rho), delta_found_vs_true.png,
# errors_vs_true_delta.png, paired_deviance.png, compare_sgd_cnn.txt
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = '/mnt/d/Amaury/simu_precision_claude/unbiased_2026-11-09/'; O = B + 'analysis/'  # runs and outputs (D:); folder renamed 2026-11-09_unbiased_sgd_vs_cnn after the merge
CASES = [('case1_poisson', 'Poisson only'), ('case2_emccd', '+ EM excess, read noise'), ('case3_emccd_bgnoise', '+ background fluct.')]
METH = [('sgd', 'SGD_jax.py', '--'), ('cnn', 'CNN + LM', '-')]
COL = {'case1_poisson': 'tab:green', 'case2_emccd': 'tab:orange', 'case3_emccd_bgnoise': 'tab:red'}
PAR = [('rho', 'deg'), ('eta', 'deg'), ('delta', 'deg'), ('z', 'nm'), ('x', 'nm'), ('N', 'relative')]
folder = lambda c, m: [f for f in sorted(glob.glob(f'{B}2026-11-1*_{c}_{m}')) if 'CRASHED' not in f][-1]
S, R = {}, {}
for c, _ in CASES:
    for m, _, _ in METH:
        f = folder(c, m); S[c, m] = pd.read_csv(f + '/summary.csv'); R[c, m] = dict(np.load(f + '/results_huber.npz'))
rs = lambda v: 1.4826*np.median(np.abs(v - np.median(v)))
lines = ['Unbiased simulations SGD_jax.py vs CNN + LM, same PSF (32000 per case, N 2100, bg 25, delta 60-120, huber)',
         'per grid point (rho, eta): bias = median error, precision = robust std (1.4826 MAD); then median over the 400 points', '']
# 1. table
hdr = f'{"case":22s} {"method":10s}' + ''.join(f' {p:>14s}' for p, _ in PAR) + '   delta slope/offset   dev<other  paired'
lines += ['|bias| / precision (median over the grid points):', hdr]
for c, cl in CASES:
    for m, ml, _ in METH:
        s = S[c, m]; r = R[c, m]
        row = f'{c:22s} {ml:10s}' + ''.join(f' {np.median(np.abs(s["bias_"+p])):6.2f}/{np.median(s["sigma_"+p]):6.2f}' for p, _ in PAR)
        p1 = np.polyfit(r['true_delta'], r['found_delta'], 1)
        o = R[c, 'cnn' if m == 'sgd' else 'sgd']; same = np.allclose(r['true_rho'], o['true_rho'])
        better = np.mean(r['found_deviance'] < o['found_deviance'])*100
        lines.append(row + f'   {p1[0]:.3f} / {np.polyval(p1, 90) - 90:+5.1f} at 90   {better:5.1f} %   {same}')
lines.append('(N bias/precision relative; delta slope of found vs true over all PSF and its value at true 90 minus 90; '
             'dev<other: % of PSF where this method has the lower deviance; paired: same true values in both runs)')
# 2. bias / precision vs eta
fig, ax = plt.subplots(2, len(PAR), figsize=(4*len(PAR), 7.5))
for j, (p, u) in enumerate(PAR):
    for c, cl in CASES:
        for m, ml, ls in METH:
            g = S[c, m].groupby('eta'); e = np.array(list(g.groups))
            ax[0, j].plot(e, g['bias_'+p].median().values, ls, color=COL[c], lw=1.5, label=f'{cl}, {ml}')
            ax[1, j].plot(e, g['sigma_'+p].median().values, ls, color=COL[c], lw=1.5)
    ax[0, j].set_title(f'{p}: bias ({u})'); ax[1, j].set_title(f'{p}: precision ({u})')
    for a in ax[:, j]: a.grid(alpha=.3); a.set_xlabel('true eta (deg)')
    ax[0, j].axhline(0, color='0.4', lw=.8)
ax[1, 0].set_ylim(0, 40); ax[0, 0].legend(fontsize=7)
fig.suptitle('Unbiased simulations: bias (median error) and precision (robust std) vs true eta, median over rho; solid CNN + LM, dashed SGD_jax.py')
fig.tight_layout(); fig.savefig(O + 'vs_eta.png', dpi=85); plt.close(fig)
# 3. delta found vs true
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
for i, (m, ml, _) in enumerate(METH):
    for k, (c, cl) in enumerate(CASES):
        r = R[c, m]; a = ax[i, k]
        a.hist2d(r['true_delta'], r['found_delta'], bins=[np.arange(60, 121, 2), np.arange(0, 181, 3)], cmap='Greys')
        b = np.arange(60, 121, 5); mid = (b[:-1] + b[1:])/2; ib = np.digitize(r['true_delta'], b) - 1
        med = [np.median(r['found_delta'][ib == q]) for q in range(len(mid))]
        a.plot(mid, med, 'o-', color=COL[c], label='median per bin'); a.plot([60, 120], [60, 120], 'b--', lw=1, label='found = true')
        p1 = np.polyfit(r['true_delta'], r['found_delta'], 1)
        a.set_title(f'{ml}, {cl}\nslope {p1[0]:.3f}, found - true at 90: {np.polyval(p1, 90) - 90:+.1f}; at delta<5: {np.mean(r["found_delta"] < 5)*100:.1f} %', fontsize=9)
        a.set_xlabel('true delta (deg)'); a.set_ylabel('found delta (deg)'); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(O + 'delta_found_vs_true.png', dpi=85); plt.close(fig)
# 4. errors vs true delta (all PSF with eta 30-150, where rho is defined)
fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
for j, p in enumerate(['rho', 'eta', 'delta', 'z']):
    for c, cl in CASES:
        for m, ml, ls in METH:
            r = R[c, m]; sel = (r['true_eta'] > 30) & (r['true_eta'] < 150)
            b = np.arange(60, 121, 5); mid = (b[:-1] + b[1:])/2; ib = np.digitize(r['true_delta'][sel], b) - 1; err = r['error_'+p][sel]
            ax[0, j].plot(mid, [np.median(err[ib == q]) for q in range(len(mid))], ls, color=COL[c], label=f'{cl}, {ml}')
            ax[1, j].plot(mid, [rs(err[ib == q]) for q in range(len(mid))], ls, color=COL[c])
    ax[0, j].set_title(f'{p} bias'); ax[1, j].set_title(f'{p} precision (robust std)')
    for a in ax[:, j]: a.grid(alpha=.3); a.set_xlabel('true delta (deg)')
    ax[0, j].axhline(0, color='0.4', lw=.8)
ax[0, 0].legend(fontsize=7)
fig.suptitle('Errors vs true delta (true eta 30-150 deg); solid CNN + LM, dashed SGD_jax.py')
fig.tight_layout(); fig.savefig(O + 'errors_vs_true_delta.png', dpi=85); plt.close(fig)
# 5. paired deviance
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
lines += ['', 'paired deviance (same PSF): SGD - CNN']
for k, (c, cl) in enumerate(CASES):
    dd = R[c, 'sgd']['found_deviance'] - R[c, 'cnn']['found_deviance']
    ax[k].hist(np.clip(dd, -0.05, 0.05), bins=np.linspace(-0.05, 0.05, 101), color=COL[c])
    ax[k].axvline(0, color='k', lw=.8); ax[k].set_yscale('log'); ax[k].set_xlabel('deviance SGD - deviance CNN + LM (clipped +-0.05)')
    ax[k].set_title(f'{cl}: CNN lower on {np.mean(dd > 0)*100:.1f} %, median {np.median(dd):+.4f}', fontsize=9)
    big = dd > 0.01
    lines.append(f'{c:22s} median {np.median(dd):+.5f}, CNN lower on {np.mean(dd > 0)*100:.1f} %, SGD worse by > 0.01 on {big.mean()*100:.1f} % '
                 f'(their |delta error| SGD {np.median(np.abs(R[c, "sgd"]["error_delta"][big])):.1f} vs CNN {np.median(np.abs(R[c, "cnn"]["error_delta"][big])):.1f} deg)')
fig.tight_layout(); fig.savefig(O + 'paired_deviance.png', dpi=85); plt.close(fig)
open(O + 'compare_sgd_cnn.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
