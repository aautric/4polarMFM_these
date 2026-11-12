# 2026-11-12 (Claude): table SGD_jax.py vs SGD_jax_CNN.py (CNN start + per-PSF LM) on the same simulated PSF:
#   python simu_precision/compare_sgd_cnn.py <SGD run> <CNN run> [<SGD run> <CNN run> ...] --out <file.txt>
# Each pair = two runs of SGD_precision.py with the same seed and options (same truth and noise; checked: "paired").
# Per grid point (rho, eta): bias = median error, precision = robust std (as the maps); then the median over the
# grid points. Columns: |bias| / precision of rho, eta, delta (deg), z, x (nm), N (relative); slope of found vs true
# delta and its value at true 90 minus 90; % of PSF where this method has the lower deviance (same deviance formula in
# both pipelines, review item 20); paired = same true values in both runs. Then the paired deviance SGD - CNN.
# (Table only, figures removed 2026-11-12 at the user's request; maps: polar_maps.py; CRB: crb_vs_simulation.py.)
import sys, argparse, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import SGD_precision as P

PAR = [('rho', 'deg'), ('eta', 'deg'), ('delta', 'deg'), ('z', 'nm'), ('x', 'nm'), ('N', 'relative')]


def load(run):
    # Purpose: statistics per grid point (as summary.csv) and the per-PSF results of the first loss of a run
    run = Path(run); res = sorted(run.glob('results_*.npz'))[0]
    r = dict(np.load(res)); sim = dict(np.load(run / 'simulation.npz'))
    rows = P.statistics(r['index'], {k[6:]: r[k] for k in r if k.startswith('error_')}, sim)
    return run.name.split('_', 2)[-1], pd.DataFrame(rows), r


def main(argv):
    p = argparse.ArgumentParser(description='SGD_jax.py vs CNN + LM on the same simulated PSF')
    p.add_argument('runs', nargs='+', help='pairs: SGD run folder, CNN run folder'); p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if len(a.runs) % 2:
        raise SystemExit('give pairs of run folders: SGD run, CNN run')
    lines = ['SGD_jax.py vs CNN + LM, same PSF; per grid point bias = median error, precision = robust std (1.4826 MAD); '
             'then median over the grid points', '',
             f'{"run":32s} {"method":10s}' + ''.join(f' {p:>14s}' for p, _ in PAR) + '   delta slope/offset   dev<other  paired']
    pairs = [(load(a.runs[k]), load(a.runs[k + 1])) for k in range(0, len(a.runs), 2)]
    for (ls, ss, rs), (lc, sc, rc) in pairs:
        same = len(rs['true_rho']) == len(rc['true_rho']) and np.allclose(rs['true_rho'], rc['true_rho'])
        for lab, ml, s, r, o in ((ls, 'SGD_jax.py', ss, rs, rc), (lc, 'CNN + LM', sc, rc, rs)):
            p1 = np.polyfit(r['true_delta'], r['found_delta'], 1)
            better = np.mean(r['found_deviance'] < o['found_deviance'])*100 if same else np.nan
            lines.append(f'{lab:32s} {ml:10s}' + ''.join(f' {np.median(np.abs(s["bias_"+p])):6.2f}/{np.median(s["sigma_"+p]):6.2f}' for p, _ in PAR)
                         + f'   {p1[0]:.3f} / {np.polyval(p1, 90) - 90:+5.1f} at 90   {better:5.1f} %   {same}')
    lines.append('(N relative; delta slope of found vs true over all PSF and its value at true 90 minus 90; dev<other: % of PSF '
                 'where this method has the lower deviance; paired: same true values in both runs)')
    lines += ['', 'paired deviance (same PSF): SGD - CNN']
    for (ls, _, rs), (lc, _, rc) in pairs:
        if len(rs['found_deviance']) != len(rc['found_deviance']): continue
        dd = rs['found_deviance'] - rc['found_deviance']; big = dd > 0.01
        lines.append(f'{ls} / {lc}: median {np.median(dd):+.5f}, CNN lower on {np.mean(dd > 0)*100:.1f} %, SGD worse by > 0.01 on '
                     f'{big.mean()*100:.1f} % (their |delta error| SGD {np.median(np.abs(rs["error_delta"][big])):.1f} vs CNN '
                     f'{np.median(np.abs(rc["error_delta"][big])):.1f} deg)')
    Path(a.out).write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:])
