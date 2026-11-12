# 2026-11-12 (Claude): usual polar maps (bias and precision per simulated (rho, eta), as polar_summary_* of
# SGD_precision_claude.py) for the CNN runs of the unbiased simulations, with 2 more rows: bias and precision of the CNN
# first guess alone (intermediate result, before the LM fit) vs the truth.
# The CNN start was not saved per PSF: cnn_first_guess (cnn_first_guess.py, same checkpoint) is run again on the saved
# simulated PSF (simulation.npz, the same data the runs fitted), with the z table of SGD_jax_CNN.py (CNN trained at
# middle_plane 1.4, runs at 1.0). CPU only (JAX_PLATFORMS=cpu, no CUDA), the GPU queue is not touched.
# Errors as collect() of SGD_precision_claude.py (rho wrapped, eta flipped when rho crosses 0/180, N relative).
# Colour scales (2026-11-12, user): each row from its 5th to its 95th percentile (bias: white kept at 0, TwoSlopeNorm).
# The start values are reused from cnn_start_<case>.npz when they exist (no CNN rerun).
# Outputs here: polar_summary_<case>_cnn_with_start.png, cnn_start_<case>.npz (start values), polar_maps_cnn_start.txt
import os, sys, glob, re
os.environ['JAX_PLATFORMS'] = 'cpu'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, G); os.chdir(G)  # the repo this script is in (was main)
import SGD_precision_claude as P
B = '/mnt/d/Amaury/simu_precision_claude/unbiased_2026-11-09/'; O = B + 'analysis/'  # runs and outputs (D:); folder renamed 2026-11-09_unbiased_sgd_vs_cnn after the merge
CASES = ['case1_poisson', 'case2_emccd', 'case3_emccd_bgnoise']
tmp = Path('/tmp/claude-1000/-mnt-c-Users-Amaury-Documents-GitHub-4polarMFM-these/de72ebfc-bf70-4cdb-bc45-45a2e63f4851/scratchpad/cnn_start_tmp'); tmp.mkdir(exist_ok=True)

# namespace of SGD_jax_CNN.py with the run settings (middle_plane 1.0), then its CNN z table
P.SGD_SCRIPT = P.HERE / 'SGD_jax_CNN.py'
a = P.parse_arguments(['--preset', 'actin', '--N', '2100', '--delta-range', '60', '120', '--losses', 'huber', '--cnn', '--out', str(tmp)])
calibration = {'J1': 'np.eye(2)', 'J2': 'rot(45)', 'rotation': '0', 'rotation2': '0', 'rho_offset': '0',
               'unitary_J': 'False', 'y_gain_J1': '1.', 'y_gain_J2': '1.'}
NEED = not all(os.path.exists(O + f'cnn_start_{c}.npz') for c in CASES)
if NEED:
  ns, _, source = P.build_pipeline(dict(calibration, loss2='huber', middle_plane=repr(-a.d)), tmp, 'huber', False, a)
  i0 = source.index('def cnn_z_equivalence'); i1 = source.index('\n', source.index('cnn_z_table = ', i0))
  exec(compile(re.sub(r'(?m)^    ', '', source[source.rfind('\n', 0, i0) + 1:i1]) if source[i0 - 4:i0] == '    ' else source[i0:i1], 'cnn_z', 'exec'), ns)
  print('middle_plane', float(ns['middle_plane']), 'CNN z table: z_cnn - z at 0.25/0.75/1.25 um:',
      [round(1000*(np.interp(z, *ns['cnn_z_table']) - z)) for z in (0.25, 0.75, 1.25)], flush=True)

def errors(found, sim):
    t = {k: sim['true_'+k] for k in ('rho', 'eta', 'delta', 'N', 'x', 'y', 'z')}
    err = {}; raw = found['rho'] - t['rho']; err['rho'] = P.wrap(raw)
    eta = found['eta'].copy(); crossed = np.abs(err['rho'] - raw) > 1; eta[crossed] = 180 - eta[crossed]
    err['eta'] = P.wrap(eta - t['eta']); err['delta'] = found['delta'] - t['delta']
    for k in ('x', 'y', 'z'): err[k] = found[k] - 1000*t[k]
    err['N'] = (found['N'] - t['N'])/t['N']
    return err

PAR = [('rho', 'deg', 1), ('eta', 'deg', 1), ('delta', 'deg', 1), ('N', '%', 100)]
ROWS = [('final (CNN + LM)', 'bias'), ('final (CNN + LM)', 'sigma'), ('CNN first guess', 'bias'), ('CNN first guess', 'sigma')]
lines = ['Polar maps of the CNN runs + CNN first guess alone (median over the 400 grid points of |bias| / precision)']
for case in CASES:
    run = [f for f in sorted(glob.glob(f'{B}2026-11-1*_{case}_cnn')) if 'CRASHED' not in f][-1]
    sim = dict(np.load(run + '/simulation.npz'))
    if NEED:
        out = [ns['cnn_first_guess'](sim['data'][k:k + 2000], z_table=ns['cnn_z_table']) for k in range(0, len(sim['data']), 2000)]
        x0, y0, z0, N0, rho0, eta0, delta0 = [np.concatenate([np.asarray(o[j]) for o in out]).astype(float) for j in range(7)]
        np.savez_compressed(O + f'cnn_start_{case}.npz', x=x0, y=y0, z=z0, N=N0, rho=rho0, eta=eta0, delta=delta0)
    else:
        c0 = np.load(O + f'cnn_start_{case}.npz'); x0, y0, z0, N0, rho0, eta0, delta0 = [c0[k] for k in ('x', 'y', 'z', 'N', 'rho', 'eta', 'delta')]
    start = dict(rho=rho0, eta=eta0, delta=delta0, N=N0, x=1000*x0, y=1000*y0, z=1000*z0)
    idx = np.arange(len(sim['data']))
    rows_start = P.statistics(idx, errors(start, sim), sim)
    import pandas as pd
    final = pd.read_csv(run + '/summary.csv').to_dict('records')
    data = {'final (CNN + LM)': final, 'CNN first guess': rows_start}
    for lab, rows in data.items():
        lines.append(f'{case:22s} {lab:18s} ' + '  '.join(f'{k} {np.nanmedian(np.abs([r["bias_"+k] for r in rows]))*s:5.2f}/{np.nanmedian([r["sigma_"+k] for r in rows])*s:5.2f}'
                                                       for k, u, s in PAR + [('z', 'nm', 1), ('x', 'nm', 1)]))
    # figure: the usual maps, 4 rows (final bias, final precision, CNN bias, CNN precision), colour scale per row
    fig, axs = plt.subplots(4, len(PAR), figsize=(3.3*len(PAR), 13.5), subplot_kw=dict(projection='polar'), layout='constrained')
    for ri, (lab, kind) in enumerate(ROWS):
        for c, (k, unit, s) in enumerate(PAR):
            v = s*np.array([r[f'{kind}_{k}'] for r in data[lab]], float); f = v[np.isfinite(v)]
            vmin, vmax = np.percentile(f, [5, 95]); cmap = 'coolwarm' if kind == 'bias' else 'viridis'
            im = P.disc_map(axs[ri, c], data[lab], v, cmap, vmin, vmax)
            if kind == 'bias' and vmin < 0 < vmax:
                im.set_norm(matplotlib.colors.TwoSlopeNorm(0., vmin, vmax))
            med = np.nanmedian(np.abs(v)) if kind == 'bias' else np.nanmedian(v)
            axs[ri, c].set_title(f'{k}, {"median |bias|" if kind == "bias" else "median precision"} {med:.1f} {unit}', fontsize=9)
            cb = fig.colorbar(im, ax=axs[ri, c], location='bottom', shrink=0.8, aspect=18, pad=0.02)
            cb.set_label(f'{k} {"bias" if kind == "bias" else "precision"} ({unit})', fontsize=8); cb.ax.tick_params(labelsize=7)
            if c == 0:
                axs[ri, c].text(-0.25, 0.5, f'{lab}\n{"bias (median error)" if kind == "bias" else "precision (robust std)"}', transform=axs[ri, c].transAxes,
                                rotation=90, ha='center', va='center', fontsize=10, fontweight='bold')
    fig.suptitle(f'{case}, CNN + LM run: rows 1-2 final fit, rows 3-4 CNN first guess alone (before LM), vs the truth\n'
                 'angle = rho, radius = eta (circles at 30, 60 deg; eta > 90 at rho+180, 180-eta); colour scale of each map from its 5th to its 95th percentile (bias: white = 0)', fontsize=11)
    fig.savefig(O + f'polar_summary_{case}_cnn_with_start.png', dpi=90); plt.close(fig)
    print(lines[-2]); print(lines[-1], flush=True)
open(O + 'polar_maps_cnn_start.txt', 'w').write('\n'.join(lines) + '\n(|bias| / precision; rho eta delta deg, N relative x100 = %, z x nm)\n')

# usual maps of the SGD_jax.py runs with the same colour convention (2 rows: bias, precision), polar_summary_<case>_sgd.png
import pandas as pd
for case in CASES:
    run = [f for f in sorted(glob.glob(f'{B}2026-11-1*_{case}_sgd')) if 'CRASHED' not in f][-1]
    rows = pd.read_csv(run + '/summary.csv').to_dict('records')
    fig, axs = plt.subplots(2, len(PAR), figsize=(3.3*len(PAR), 7.2), subplot_kw=dict(projection='polar'), layout='constrained')
    for ri, kind in enumerate(('bias', 'sigma')):
        for c, (k, unit, s) in enumerate(PAR):
            v = s*np.array([r[f'{kind}_{k}'] for r in rows], float); f = v[np.isfinite(v)]
            vmin, vmax = np.percentile(f, [5, 95]); cmap = 'coolwarm' if kind == 'bias' else 'viridis'
            im = P.disc_map(axs[ri, c], rows, v, cmap, vmin, vmax)
            if kind == 'bias' and vmin < 0 < vmax:
                im.set_norm(matplotlib.colors.TwoSlopeNorm(0., vmin, vmax))
            med = np.nanmedian(np.abs(v)) if kind == 'bias' else np.nanmedian(v)
            axs[ri, c].set_title(f'{k}, {"median |bias|" if kind == "bias" else "median precision"} {med:.1f} {unit}', fontsize=9)
            cb = fig.colorbar(im, ax=axs[ri, c], location='bottom', shrink=0.8, aspect=18, pad=0.02)
            cb.set_label(f'{k} {"bias" if kind == "bias" else "precision"} ({unit})', fontsize=8); cb.ax.tick_params(labelsize=7)
            if c == 0:
                axs[ri, c].text(-0.25, 0.5, f'SGD_jax.py\n{"bias (median error)" if kind == "bias" else "precision (robust std)"}', transform=axs[ri, c].transAxes,
                                rotation=90, ha='center', va='center', fontsize=10, fontweight='bold')
    fig.suptitle(f'{case}, SGD_jax.py run vs the truth\nangle = rho, radius = eta (circles at 30, 60 deg; eta > 90 at rho+180, 180-eta); '
                 'colour scale of each map from its 5th to its 95th percentile (bias: white = 0)', fontsize=11)
    fig.savefig(O + f'polar_summary_{case}_sgd.png', dpi=90); plt.close(fig)
