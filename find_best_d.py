# -*- coding: utf-8 -*-
"""
Finds the position of the focal planes (middle_plane, d = [middle_plane - interplane, middle_plane, middle_plane +
interplane]) that fits a data set best, before its full run. The same PSF are fitted with several middle_plane (short
runs of SGD_batch.py, all the other parameters shared) and their deviance compared: with SAF and the index mismatch the
PSF shape depends on the absolute depth, so the best middle_plane has the lowest deviance. The PSF of each run are
matched to the run at the reference middle_plane (same frame, < MATCH nm in xy) and only the PSF good at the reference
and found in all the runs are compared (paired differences: the deviance changes by ~1e-4 between the values, much
less than its spread between PSF). Best value: vertex of a parabola through the mean deviances around the minimum. If
the minimum is at an end of the values, values are added beyond it (--extend).

usage (with the python of the environment, e.g. /home/amaury/exit/envs/env_linux/bin/python):
    python find_best_d.py <RAW_DATA folder> [name=value ...] [--values=0.5,0.6,0.7,0.8,0.9] [--batches=150]
                          [--ref=0.7] [--extend=2] [--timeout=1] [--out=<folder>] [--cnn]
        name=value: parameters of SGD_jax.py / SGD_jax_CNN.py shared by all the runs (as for SGD_batch.py run),
            e.g. lambda_emission=620 sensitivity=4.4 rho_offset=-6.2 loss2=huber. Not middle_plane nor batch_nb.
        --values: middle_plane tested (um). --batches: batch_nb of each run (NPSF PSF per batch, from
            last_frame_processed: the same PSF in every run). --ref: reference middle_plane of the comparison.
        --extend: values added at most beyond an end of the values when the minimum is there (step of the values).
        --timeout: hours before a run is stopped (hang), then tried once more.
        --out: folder of the results (default <RAW_DATA folder>/find_best_d_<date>). If it already has runs.txt,
            the values already run are not run again (restart after a stop).
        --cnn: SGD_jax_CNN.py (CNN start and per-PSF LM fit) instead of SGD_jax.py
    python find_best_d.py analyse <find_best_d folder> [--ref=0.7]
        only the comparison, on the runs of runs.txt

The runs are made one after the other (one GPU job at a time; waits while another SGD_batch.py runs). Each NPZ folder
and its fit csv are moved to the results folder (NPZ_mp<value>_<run>, fit_mp<value>_<run>.csv). Results: find_best_d.png, find_best_d.txt (best middle_plane, mean
deviance, median z, N, delta, eta of each value), runs.txt (middle_plane and run of each value), logs/.
~8 min per value with --cnn and 150 batches.
"""
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

HERE = Path(__file__).resolve().parent
BATCH_SCRIPT = HERE / 'SGD_batch.py'
MATCH = 100. # nm, largest xy distance of the same PSF in two runs
DEV_MAX, N_MIN = 1.5, 200. # PSF compared: deviance < DEV_MAX and N > N_MIN at the reference middle_plane
COLS = ['frame', 'x', 'y', 'z', 'delta', 'eta', 'N_photon', 'deviance']


def log(message):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)


# ---------------------------------------------------------------- runs

def gpu_busy():
    # another fit running (SGD_batch.py run/resume): one GPU job at a time
    ps = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
    return any(re.search(r'python.*SGD_batch\.py (run|resume)', line) for line in ps.splitlines()
               if int(line.split()[0]) != os.getpid())


def run_one(data, mp, shared, batches, cnn, out, timeout):
    # one run of SGD_batch.py at middle_plane = mp, tried twice; returns the name of the run (NPZ_<run>) or None
    (out / 'logs').mkdir(exist_ok=True)
    command = ([sys.executable, str(BATCH_SCRIPT), 'run', str(data), f'middle_plane={mp}', f'batch_nb={batches}']
               + shared + ['--no-figures', '--post-process'] + (['--cnn'] if cnn else []))
    env = dict(os.environ, MPLBACKEND='Agg', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    for attempt in (1, 2):
        if gpu_busy():
            log('another fit is running, waiting')
            while gpu_busy():
                time.sleep(60)
        log_file = out / 'logs' / f'mp{mp:g}_try{attempt}.log'
        log(f'middle_plane {mp:g}: run {attempt}, log {log_file}')
        start = time.time()
        with open(log_file, 'w') as output:
            try:
                status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                                        timeout=timeout*3600).returncode
            except subprocess.TimeoutExpired:
                status = 'timeout'
        text = log_file.read_text(errors='replace')
        done = re.findall(r'npz files in (.+)$', text, re.M)
        if status == 0 and done:
            npz = Path(done[-1].strip())
            run = npz.name.replace('NPZ_', '', 1)
            csv = npz.parent / f'fit_{run}.csv'
            if csv.is_file():
                # stored as NPZ_mp<value>_<run>, fit_mp<value>_<run>.csv: two short runs can have the same date
                run = f'mp{mp:g}_{run}'
                shutil.move(str(npz), str(out / f'NPZ_{run}')); shutil.move(str(csv), str(out / f'fit_{run}.csv'))
                log(f'middle_plane {mp:g}: done in {(time.time()-start)/60:.0f} min, run {run}')
                with open(out / 'runs.txt', 'a') as f:
                    f.write(f'{mp:g} {run}\n')
                return run
        log(f'middle_plane {mp:g}: run {attempt} FAILED ({status}), see {log_file}')
        # the NPZ folder of the failed run is kept aside, not deleted
        started = re.findall(r'(?:Saving in: |saving in )(.+)$', text, re.M)
        if started and Path(started[-1].strip()).is_dir():
            npz = Path(started[-1].strip())
            shutil.move(str(npz), str(out / f'failed_{npz.name}'))
    return None


# ---------------------------------------------------------------- comparison

def read_runs(out):
    runs = {}
    for line in (out / 'runs.txt').read_text().splitlines():
        if line.strip():
            mp, run = line.split()
            runs[float(mp)] = run # the last run of a value is the one used
    return dict(sorted(runs.items()))


def analyse(out, ref=0.7):
    # paired comparison of the deviance of the runs of runs.txt; returns the best middle_plane, -1/+1 if the minimum
    # is at the lower/upper end of the values (else 0), and the step between the values
    import pandas as pd
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    runs = read_runs(out)
    mps = np.array(list(runs))
    if len(mps) < 2:
        raise SystemExit('fewer than 2 runs in runs.txt')
    ref = mps[np.argmin(np.abs(mps - ref))] # the run closest to the reference asked
    load = lambda run: pd.read_csv(out / f'fit_{run}.csv', sep=';', comment='#', usecols=COLS)
    r0 = load(runs[ref])
    groups0 = r0.groupby('frame').indices
    dev, other = {}, {}
    for mp, run in runs.items():
        d = load(run); idx = np.full(len(r0), -1)
        groups = d.groupby('frame').indices
        for f, i in groups0.items():
            j = groups.get(f)
            if j is None: continue
            dist, jj = cKDTree(d[['x', 'y']].values[j]).query(r0[['x', 'y']].values[i])
            idx[i] = np.where(dist < MATCH, j[jj], -1)
        ok = idx >= 0
        dev[mp] = np.where(ok, d.deviance.values[np.maximum(idx, 0)], np.nan)
        other[mp] = {c: np.where(ok, d[c].values[np.maximum(idx, 0)], np.nan) for c in ('z', 'N_photon', 'delta', 'eta')}
    m = ((r0.deviance < DEV_MAX) & (r0.N_photon > N_MIN)).values & np.all([np.isfinite(v) for v in dev.values()], axis=0)
    n = m.sum()
    if n < 100:
        raise SystemExit(f'only {n} PSF good at {ref:g} and found in all the runs')
    mean = np.array([dev[p][m].mean() for p in mps])
    diff = [dev[p][m] - dev[ref][m] for p in mps]
    se = np.array([x.std()/np.sqrt(n) for x in diff])
    # parabola through the mean deviances of the values around the minimum (2 on each side at most)
    i = int(np.argmin(mean)); near = slice(max(i-2, 0), min(i+3, len(mps)))
    best, c = mps[i], None
    if len(mps[near]) >= 3:
        c = np.polyfit(mps[near], mean[near], 2)
        if c[0] > 0 and mps[near].min() <= -c[1]/(2*c[0]) <= mps[near].max():
            best = -c[1]/(2*c[0])
    edge = -1 if i == 0 else (1 if i == len(mps) - 1 else 0)
    step = np.min(np.diff(mps))
    zmed = {p: np.median(other[p]['z'][m]) for p in mps}
    lines = [f'{out}', f'deviance vs middle_plane: PSF matched to the run at {ref:g} (same frame, < {MATCH:.0f} nm), good = '
             f'deviance < {DEV_MAX} and N > {N_MIN:.0f} at {ref:g} and found in all the runs: {n} PSF of {len(r0)}',
             f'd(dev) = mean paired difference to {ref:g} +- standard error', '']
    for p, mu, df, s in zip(mps, mean, diff, se):
        o = other[p]
        lines.append(f'middle_plane {p:5.2f} ({runs[p]}): mean dev {mu:.5f}  d(dev) {df.mean():+.5f} +- {s:.5f}  better '
                     f'than {ref:g} for {100*(df < 0).mean():4.1f} %  median z {zmed[p]:5.0f} nm  N {np.median(o["N_photon"][m]):5.0f}'
                     f'  delta {np.median(o["delta"][m]):5.1f}  eta {np.median(o["eta"][m]):5.1f}')
    lines += ['', f'best middle_plane: {best:.3f} um' + (' (parabola)' if best != mps[i] else ' (lowest value, no parabola)')
              + {-1: ', minimum at the LOWER end of the values', 1: ', minimum at the UPPER end of the values', 0: ''}[edge]]
    lines.append(f'median z shift vs {ref:g}: {np.interp(best, mps, [zmed[p] for p in mps]) - zmed[ref]:+.0f} nm')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].errorbar(mps, [x.mean() for x in diff], se, fmt='o-', color='k', capsize=2)
    if c is not None and c[0] > 0:
        xx = np.linspace(mps[near].min(), mps[near].max(), 100)
        ax[0].plot(xx, np.polyval(c, xx) - mean[mps == ref][0], '--', color='tab:orange', label=f'parabola, best {best:.2f} um')
        ax[0].legend(fontsize=9)
    ax[0].axhline(0, color='0.6', lw=.8); ax[0].axvline(best, color='tab:orange', lw=.8, ls=':')
    ax[0].set_xlabel('middle_plane (um)'); ax[0].set_ylabel(f'mean deviance - mean at {ref:g}')
    ax[0].set_title(f'{n} PSF, the same in every run', fontsize=10)
    # where in z the choice matters: difference vs the z fitted at the reference, 100 nm bins of > 50 PSF
    zb = np.arange(0, 1201, 100); z0 = other[ref]['z']
    for p, col in zip(mps, plt.cm.coolwarm(np.linspace(0, 1, len(mps)))):
        if p == ref: continue
        y = [np.mean((dev[p] - dev[ref])[m & (z0 >= lo) & (z0 < lo + 100)]) if np.sum(m & (z0 >= lo) & (z0 < lo + 100)) > 50
             else np.nan for lo in zb]
        ax[1].plot(zb + 50, y, 'o-', ms=3, color=col, label=f'{p:g} - {ref:g}')
    ax[1].axhline(0, color='0.6', lw=.8); ax[1].legend(fontsize=8)
    ax[1].set_xlabel(f'fitted z at middle_plane {ref:g} (nm)'); ax[1].set_ylabel('mean deviance difference')
    fig.suptitle(f'{out.parent.parent.name}/{out.parent.name}: deviance vs middle_plane (planes middle_plane -+ interplane)',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'find_best_d.png', dpi=90); plt.close(fig)
    (out / 'find_best_d.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return best, edge, step


# ---------------------------------------------------------------- command line

def main(arguments):
    options = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '') for a in arguments if a.startswith('--'))
    words = [a for a in arguments if not a.startswith('--')]
    unknown = set(options) - {'values', 'batches', 'ref', 'extend', 'timeout', 'out', 'cnn'}
    if not words or unknown:
        raise SystemExit(__doc__)
    ref = float(options.get('ref', 0.7))
    if words[0] == 'analyse':
        analyse(Path(words[1]), ref)
        return
    data, shared = Path(words[0]), words[1:]
    if not (data / 'reconstruction').is_dir():
        raise SystemExit(f'{data / "reconstruction"} does not exist')
    for a in shared:
        if '=' not in a or a.split('=', 1)[0] in ('middle_plane', 'batch_nb'):
            raise SystemExit(f'"{a}": shared parameters are name=value, without middle_plane and batch_nb')
    values = sorted(float(v) for v in options.get('values', '0.5,0.6,0.7,0.8,0.9').split(','))
    batches, extend, timeout = int(options.get('batches', 150)), int(options.get('extend', 2)), float(options.get('timeout', 1))
    cnn = 'cnn' in options
    out = Path(options['out']) if 'out' in options else data / f'find_best_d_{datetime.now():%Y-%m-%d_%Hh%M}'
    out.mkdir(parents=True, exist_ok=True)
    # the settings of the runs, checked on a restart (the runs of runs.txt must be comparable with the new ones)
    settings = f'batch_nb={batches} cnn={cnn} ' + ' '.join(shared)
    if (out / 'settings.txt').is_file() and (out / 'settings.txt').read_text().strip() != settings:
        raise SystemExit(f'{out} was run with other settings: {(out / "settings.txt").read_text().strip()}')
    (out / 'settings.txt').write_text(settings + '\n')
    log(f'results in {out}')

    def run_missing(vals):
        done = read_runs(out) if (out / 'runs.txt').is_file() else {}
        for mp in vals:
            if any(abs(mp - p) < 1e-6 for p in done):
                log(f'middle_plane {mp:g}: already run ({done[min(done, key=lambda p: abs(p - mp))]})')
                continue
            if run_one(data, mp, shared, batches, cnn, out, timeout) is None:
                log(f'middle_plane {mp:g}: left out')

    run_missing(values)
    best, edge, step = analyse(out, ref)
    for _ in range(extend):
        if edge == 0:
            break
        mps = list(read_runs(out))
        new = round((max(mps) + step) if edge > 0 else (min(mps) - step), 3)
        if new <= 0:
            break
        log(f'minimum at the {"upper" if edge > 0 else "lower"} end: middle_plane {new:g} added')
        run_missing([new])
        best, edge, step = analyse(out, ref)
    log(f'best middle_plane {best:.3f} um' + (' (still at an end of the values)' if edge else '')
        + f', for the full run: middle_plane={best:.2f}')


if __name__ == '__main__':
    main(sys.argv[1:])
