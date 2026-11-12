# 2026-11-12 (Claude): polar maps of any run of SGD_precision.py (simulated precision):
#   python simu_precision/polar_maps.py <run folder> [<run folder> ...] [--redo-cnn]
# For each run folder (the one with simulation.npz, results_<loss>.npz, NPZ_<loss>\) and each loss it fitted:
#   rows 1-2: bias (median of the error) and precision (robust std, 1.4826 MAD) per simulated (rho, eta), for rho,
#             eta, delta, N, from results_<loss>.npz (statistics() of SGD_precision.py, as summary.csv);
#   rows 3-4, only if the run used the CNN (SGD_jax_CNN.py): the same for the CNN first guess alone (before the LM).
#   The run does not save the CNN start (nan in its npz): cnn_first_guess is run again on simulation.npz (the same
#   patches the run fitted, same checkpoint) with the z table of the run, rebuilt with the settings of the run read
#   from NPZ_<loss>\config.txt (middle_plane, J1, J2, rotations, rho_offset, y gains, unitary_J, registration_z and
#   fov_aberrations when None; the optics, lambda,
#   NA, pixel, planes, SAF, must be those of the code, else the script stops). Cached in the run folder as
#   cnn_start_<loss>.npz (--redo-cnn: computed again). CPU only (the GPU is not used).
# Polar map: angle = rho, radius = eta (eta > 90 drawn at rho + 180, 180 - eta, see disc_map); colour scale of each map
# from its 5th to its 95th percentile (bias: white = 0).
# Outputs in the run folder: polar_maps_<loss>.png, polar_maps_<loss>.txt (median over the grid of |bias| / precision).
# Replaces polar_maps_cnn_start.py (fixed list of runs and settings, 2026-11-12).
import os, sys, glob, re, argparse, tempfile
os.environ['JAX_PLATFORMS'] = 'cpu'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, G); os.chdir(G)  # the repo this script is in
sys.path.insert(0, os.path.join(G, 'simu_precision')); import SGD_precision as P

PAR = [('rho', 'deg', 1), ('eta', 'deg', 1), ('delta', 'deg', 1), ('N', '%', 100)]  # N error is relative -> %
TEXT = PAR + [('z', 'nm', 1), ('x', 'nm', 1)]
CALIB = ('J1', 'J2', 'rotation', 'rotation2', 'rho_offset', 'unitary_J', 'y_gain_J1', 'y_gain_J2', 'middle_plane')
OPTICS = ('lambda_emission', 'NA', 'mag', 'MAG', 'l_pixel', 'f_tube', 'SAF', 'interplane')


def read_config(path):
    # Purpose: the settings of the run, config.txt written by SGD_jax*.py in NPZ_<loss>\ ("name = value" lines)
    cfg = {}
    for line in open(path):
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); cfg[k.strip()] = v.strip()
    return cfg


def cnn_start(run, loss, cfg):
    # Purpose: CNN first guess of every simulated PSF of the run, as the run computed it (see the header)
    P.SGD_SCRIPT = P.HERE / 'SGD_jax_CNN.py'
    over = {k: (f'np.array({cfg[k]})' if k in ('J1', 'J2') else cfg[k]) for k in CALIB if k in cfg}
    over['loss2'] = repr(loss)
    for k in ('registration_z', 'fov_aberrations'):  # None in the simulations: no correction, as the run
        if cfg.get(k) == 'None': over[k] = 'None'
    tmp = Path(tempfile.mkdtemp(prefix='polar_maps_'))
    a = P.parse_arguments(['--preset', 'poisson', '--losses', loss, '--cnn', '--out', str(tmp),
                           '--d', str(-float(cfg['middle_plane'])), '--interplane', cfg['interplane']])
    ns, _, source = P.build_pipeline(over, tmp, loss, False, a)
    for k in OPTICS:  # optics not overridden: the code must have the values of the run
        if k in cfg and not np.allclose(float(np.asarray(ns[k])), float(eval(cfg[k]))):
            raise SystemExit(f'{run}: {k} = {cfg[k]} in the run, {float(np.asarray(ns[k]))} in the code: CNN start not reproducible')
    # z table of the run: the code of SGD_jax_CNN.py from "def cnn_z_equivalence" to the line "cnn_z_table = ..."
    i0 = source.index('def cnn_z_equivalence'); i1 = source.index('\n', source.index('cnn_z_table = ', i0))
    exec(compile(source[i0:i1], 'cnn_z', 'exec'), ns)
    sim = np.load(run / 'simulation.npz')['data']
    out = [ns['cnn_first_guess'](sim[k:k + 2000], z_table=ns['cnn_z_table']) for k in range(0, len(sim), 2000)]
    x0, y0, z0, N0, rho0, eta0, delta0 = [np.concatenate([np.asarray(o[j]) for o in out]).astype(float) for j in range(7)]
    return dict(rho=rho0, eta=eta0, delta=delta0, N=N0, x=1000*x0, y=1000*y0, z=1000*z0)  # x, y, z in nm


def errors(found, sim):
    # Purpose: errors of the CNN start vs the truth, as collect() of SGD_precision.py (rho wrapped, eta flipped
    # when rho crosses 0/180, x y z in nm, N relative)
    t = {k: sim['true_'+k] for k in ('rho', 'eta', 'delta', 'N', 'x', 'y', 'z')}
    err = {}; raw = found['rho'] - t['rho']; err['rho'] = P.wrap(raw)
    eta = found['eta'].copy(); crossed = np.abs(err['rho'] - raw) > 1; eta[crossed] = 180 - eta[crossed]
    err['eta'] = P.wrap(eta - t['eta']); err['delta'] = found['delta'] - t['delta']
    for k in ('x', 'y', 'z'): err[k] = found[k] - 1000*t[k]
    err['N'] = (found['N'] - t['N'])/t['N']
    return err


def median_line(rows):
    return '  '.join(f'{k} {np.nanmedian(np.abs([r["bias_"+k] for r in rows]))*s:6.2f}/{np.nanmedian([r["sigma_"+k] for r in rows])*s:6.2f}'
                     for k, u, s in TEXT)


def figure(run, loss, blocks, out):
    # Purpose: the maps, 2 rows (bias, precision) per block (final fit; CNN first guess if any)
    rows = [(lab, kind) for lab in blocks for kind in ('bias', 'sigma')]
    fig, axs = plt.subplots(len(rows), len(PAR), figsize=(3.3*len(PAR), 3.4*len(rows) + 0.8), subplot_kw=dict(projection='polar'),
                            layout='constrained', squeeze=False)
    for ri, (lab, kind) in enumerate(rows):
        for c, (k, unit, s) in enumerate(PAR):
            v = s*np.array([r[f'{kind}_{k}'] for r in blocks[lab]], float)
            if (kind, k) == ('bias', 'eta'):
                v = P.eta_bias_on_disc(blocks[lab], v)  # + = towards the edge of the disc everywhere (review item 12)
            f = v[np.isfinite(v)]
            vmin, vmax = np.percentile(f, [5, 95]) if f.size else (0, 1)
            im = P.disc_map(axs[ri, c], blocks[lab], v, 'coolwarm' if kind == 'bias' else 'viridis', vmin, vmax)
            if kind == 'bias' and vmin < 0 < vmax:
                im.set_norm(matplotlib.colors.TwoSlopeNorm(0., vmin, vmax))
            med = np.nanmedian(np.abs(v)) if kind == 'bias' else np.nanmedian(v)
            axs[ri, c].set_title(f'{k}, {"median |bias|" if kind == "bias" else "median precision"} {med:.1f} {unit}\n{P.map_noise(blocks[lab], k, kind, s)}', fontsize=8)
            cb = fig.colorbar(im, ax=axs[ri, c], location='bottom', shrink=0.8, aspect=18, pad=0.02)
            cb.set_label(f'{k} {"bias" if kind == "bias" else "precision"} ({unit})', fontsize=8); cb.ax.tick_params(labelsize=7)
            if c == 0:
                axs[ri, c].text(-0.25, 0.5, f'{lab}\n{"bias (median error)" if kind == "bias" else "precision (robust std)"}',
                                transform=axs[ri, c].transAxes, rotation=90, ha='center', va='center', fontsize=10, fontweight='bold')
    fig.suptitle(f'{run.name}, loss {loss}: ' + ' / '.join(blocks) + ' vs the truth\n'
                 'angle = rho, radius = eta (circles at 30, 60 deg; eta > 90 at rho+180, 180-eta); colour scale of each map from its '
                 '5th to its 95th percentile (bias: white = 0; eta bias: + = towards the edge of the disc, more in-plane)', fontsize=11)
    fig.savefig(out, dpi=90); plt.close(fig)


def main(argv):
    p = argparse.ArgumentParser(description='polar maps of runs of SGD_precision.py')
    p.add_argument('runs', nargs='+', help='run folders (with simulation.npz and results_<loss>.npz)')
    p.add_argument('--redo-cnn', action='store_true', help='compute the CNN start again even if cnn_start_<loss>.npz exists')
    a = p.parse_args(argv)
    for run in map(Path, a.runs):
        sim = dict(np.load(run / 'simulation.npz'))
        for res in sorted(run.glob('results_*.npz')):
            loss = res.stem[len('results_'):]
            r = np.load(res)
            err = {k[6:]: r[k] for k in r.files if k.startswith('error_')}
            blocks = {'final fit': P.statistics(r['index'], err, sim)}
            first = sorted(run.glob(f'NPZ_{loss}/*.npz'), key=lambda f: int(f.stem) if f.stem.isdigit() else -1)
            npz0 = np.load(first[-1]) if first else {}
            if 'use_cnn' in getattr(npz0, 'files', []) and bool(npz0['use_cnn']):
                blocks = {'final fit (CNN + LM)': blocks['final fit']}
                cache = run / f'cnn_start_{loss}.npz'
                if cache.exists() and not a.redo_cnn:
                    start = dict(np.load(cache))
                else:
                    start = cnn_start(run, loss, read_config(run / f'NPZ_{loss}' / 'config.txt'))
                    np.savez_compressed(cache, **start)
                blocks['CNN first guess'] = P.statistics(np.arange(len(sim['data'])), errors(start, sim), sim)
            try:
                figure(run, loss, blocks, run / f'polar_maps_{loss}.png')
            except ValueError as e:  # disc_map needs a regular grid with an even number of eta bins
                print(f'{run.name} {loss}: no polar maps ({e})')
            lines = [f'{run.name}, loss {loss}: median over the grid points of |bias| / precision (robust std); rho eta delta deg, N %, z x nm']
            lines += [f'{lab:22s} {median_line(rows)}' for lab, rows in blocks.items()]
            (run / f'polar_maps_{loss}.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
