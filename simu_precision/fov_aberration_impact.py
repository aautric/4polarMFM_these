# 2026-11-13 (Claude): effect of the field-dependent aberrations of the DATA (SGD_precision.py --data-fov-aberrations,
# the fit has none) on the fitted parameters, against the same run without aberrations:
#   python simu_precision/fov_aberration_impact.py <reference run> <aberrated run> [<aberrated run> ...] --out <folder>
# Reference and aberrated runs: same seed and options except the aberrations -> same truth (checked: "paired"); the
# field positions have their own generator, so the other draws are those of the reference. Per PSF:
#   d = error(aberrated) - error(reference) (rho, eta wrapped to [-90, 90[), the effect of the aberrations on that PSF
#   (the noise of the two runs is not the same: the EMCCD draws follow the mean image, so d also holds noise).
# Figure fov_aberration_impact.png: one row per aberrated run; first column = RMS of the wavefront of the data
# (Zernike Noll 4-15 of the 3 planes and 2 channels, m-lambda) over the field (col, row of the channel images, px);
# then the median of d per field bin (8 x 8) for x, y, z (nm), rho, eta, delta (deg). Colour scale symmetric, per panel.
# Text fov_aberration_impact.txt: per run and parameter, median error and plain std of the reference and aberrated run,
# median and range over the field bins of d; the 3 Zernike coefficients (mean of the 2 channels) most correlated with d
# (the field model is a polynomial of degree <= 2: the coefficients are strongly correlated with each other, so this
# points to candidates, not to a cause).
import sys, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path

PAR = [('x', 'nm'), ('y', 'nm'), ('z', 'nm'), ('rho', 'deg'), ('eta', 'deg'), ('delta', 'deg')]
NB = 8  # field bins per axis


def wrap(v):
    return (v + 90) % 180 - 90


def load(run):
    # Purpose: per-PSF results of the first loss of a run (indexed by the PSF of simulation.npz) and its simulation
    run = Path(run); res = sorted(run.glob('results_*.npz'))[0]
    r = dict(np.load(res)); sim = dict(np.load(run / 'simulation.npz'))
    n = len(sim['true_rho']); e = {}
    for p, _ in PAR:
        v = np.full(n, np.nan); v[r['index']] = r['error_' + p]; e[p] = v
    return run.name.split('_', 2)[-1], e, sim


def main(argv):
    p = argparse.ArgumentParser(description='effect of the field-dependent aberrations of the data on the fit')
    p.add_argument('reference'); p.add_argument('runs', nargs='+'); p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lr, er, sr = load(a.reference)
    lines = [f'reference (no aberrations): {a.reference}', 'd = error(aberrated) - error(reference), same PSF', '']
    fig, ax = plt.subplots(len(a.runs), len(PAR) + 1, figsize=(3.1*(len(PAR) + 1), 2.9*len(a.runs)), squeeze=False)
    for i, run in enumerate(a.runs):
        la, ea, sa = load(run)
        if 'field_col' not in sa:
            raise SystemExit(f'{run}: no field positions (not a --data-fov-aberrations run)')
        paired = all(np.allclose(sr['true_' + k], sa['true_' + k]) for k in ('rho', 'eta', 'delta', 'x', 'y', 'z', 'N'))
        col, row = sa['field_col'], sa['field_row']
        zx, zy = sa['data_zx'], sa['data_zy']  # (n, 3, 15) rad
        rms = np.sqrt(np.sum(zx[:, :, 3:]**2 + zy[:, :, 3:]**2, axis=(1, 2))/6)/(2*np.pi)*1000  # m-lambda
        cb = np.linspace(col.min(), col.max(), NB + 1); rb = np.linspace(row.min(), row.max(), NB + 1)
        ic = np.clip(np.digitize(col, cb) - 1, 0, NB - 1); ir = np.clip(np.digitize(row, rb) - 1, 0, NB - 1)

        def binned(v):
            m = np.full((NB, NB), np.nan)
            for r_ in range(NB):
                for c_ in range(NB):
                    s = (ir == r_) & (ic == c_) & np.isfinite(v)
                    if s.sum() >= 20:
                        m[r_, c_] = np.median(v[s])
            return m

        ext = [cb[0], cb[-1], rb[-1], rb[0]]
        im = ax[i, 0].imshow(binned(rms), extent=ext, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax[i, 0]); ax[i, 0].set_title(f'{la}\nRMS wavefront (m$\\lambda$)', fontsize=8)
        ax[i, 0].set_ylabel('row (px)')
        lines += [f'{la}  ({run})', f'  paired with the reference: {paired};  RMS wavefront of the data (Noll 4-15) '
                  f'median {np.median(rms):.0f} m-lambda, 5-95 % {np.percentile(rms, 5):.0f}-{np.percentile(rms, 95):.0f}',
                  f'  {"param":6s} {"ref median/std":>16s} {"aberr median/std":>17s} {"d median":>9s} {"d bins min/max":>16s}'
                  '   most correlated Zernike (plane, Noll: r)']
        za = (zx + zy).reshape(len(col), -1)/2  # mean of the 2 channels, index 15*plane + Noll - 1
        for j, (k, u) in enumerate(PAR):
            d = ea[k] - er[k]
            if k in ('rho', 'eta'):
                d = wrap(d)
            ok = np.isfinite(d)
            m = binned(d); lim = np.nanmax(np.abs(m)) or 1
            im = ax[i, j + 1].imshow(m, extent=ext, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
            plt.colorbar(im, ax=ax[i, j + 1]); ax[i, j + 1].set_title(f'median d {k} ({u})', fontsize=8)
            cor = np.array([np.corrcoef(za[ok, q], d[ok])[0, 1] if np.std(za[ok, q]) > 0 else 0 for q in range(za.shape[1])])
            top = np.argsort(-np.abs(np.nan_to_num(cor)))[:3]
            fr, fa = er[k][np.isfinite(er[k])], ea[k][np.isfinite(ea[k])]
            lines.append(f'  {k:6s} {np.median(fr):7.2f}/{np.std(fr):7.2f} {np.median(fa):8.2f}/{np.std(fa):7.2f} '
                         f'{np.median(d[ok]):9.2f} {np.nanmin(m):7.2f}/{np.nanmax(m):7.2f}   '
                         + ', '.join(f'({q//15}, Z{q%15 + 1}: {cor[q]:+.2f})' for q in top))
        lines.append('')
    for x in ax[-1]:
        x.set_xlabel('col (px)')
    fig.suptitle(f'field-dependent aberrations in the data only, effect on the fit (d = aberrated - reference {lr})', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'fov_aberration_impact.png', dpi=110)
    (out / 'fov_aberration_impact.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print('->', out)


if __name__ == '__main__':
    main(sys.argv[1:])
