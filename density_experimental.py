# -*- coding: utf-8 -*-
"""
Density of emitters and contamination of the patches in an experimental data set, across the frames and the field of
view. Runs on CPU, reads only, writes its figures and a summary in <data folder>/density_analysis/.

1. density (pre-localisation, reconstruction/pre_loc_deltaz_*.csv, all the frames): detections per frame, fraction of
   the detections with another detection in the same frame closer than the half size of the SGD patch (6 px, 0.72 um
   + 0.06 um of position, r_patch) or than r_neighbour (PSF tails), maps across the field and curves over the frames,
   nearest neighbour distances
2. fit quality vs density (fit_*.csv of SGD_jax.py, if present): deviance as a function of the number of other
   fitted localisations within r_neighbour in the same frame
3. outlier pixels in the background (a sample of reconstructed frames): outside the detections (masked with a margin),
   pixels more than n_sigma robust standard deviations above the local background (median filter), per frame and
   across the field: isolated events (clock induced charge, cosmic rays, out of focus spikes)
4. fit quality vs outlier pixels in the patch (fit_*.csv and a sample of reconstructed frames): for each fitted
   localisation, the number of outlier pixels (same definition as 3) in its 13 x 13 patch over the 6 channels, out of
   a disc of core_radius px around its centre (its own PSF); deviance as a function of this number. The fit
   coordinates of SGD_jax.py are row = -y, col = x (checked against the pre-localisation: 18 nm)
5. examples (--examples): for each number of outlier pixels and each number of neighbours (boxes of 4 and 2), the
   localisation with the deviance closest to the median of its box: data patch (cut as SGD_jax.py does), fit (model of
   SGD_jax.py with the calibration of the config.txt of the run and the fitted parameters of its npz) and residual.
   The deviance recomputed from them is printed next to the one saved by the fit (they must be equal)

usage (python of the environment):
    python density_experimental.py <data folder> [--r-patch 0.78] [--r-neighbour 1.5] [--bin 2] [--n-frames 200]
    e.g. python density_experimental.py /mnt/d/Amaury/DATA/0_papier/actin2
"""
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
if __name__ == '__main__': # imported from analyse_SGD.py: keep the backend of the kernel
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree

PIXEL = 0.12 # um, pixel of the reconstructed frames in the sample


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', help='folder with the raw files and reconstruction/')
    p.add_argument('--r-patch', type=float, default=0.78, help='half size of the SGD patch (um)')
    p.add_argument('--r-neighbour', type=float, default=1.5, help='distance of a neighbour that can reach the patch (um)')
    p.add_argument('--r-duplicate', type=float, default=0.1, help='two fits of the same frame closer than this are the same '
                   'emitter fitted twice: only the one with the lowest deviance is kept (um)')
    p.add_argument('--bin', type=float, default=2., help='bin of the maps (um)')
    p.add_argument('--frame-bin', type=int, default=500, help='frames per point of the curves')
    p.add_argument('--n-frames', type=int, default=200, help='reconstructed frames sampled for the outlier pixels')
    p.add_argument('--n-sigma', type=float, default=5., help='threshold of the outlier pixels (robust std)')
    p.add_argument('--mask-radius', type=int, default=8, help='pixels masked around each detection (half size)')
    p.add_argument('--n-fit-frames', type=int, default=1000, help='frames sampled for the outlier pixels in the patches')
    p.add_argument('--core-radius', type=float, default=3., help='disc around the fitted position left out of the count (px)')
    p.add_argument('--recompute-outliers', action='store_true', help='recompute the outlier pixels of the patches even if '
                   'density_analysis/outliers_per_patch.npz exists')
    p.add_argument('--min-box', type=int, default=15, help='minimum number of localisations for a box of the box plot')
    p.add_argument('--examples', action='store_true', help='figures of data and fit for each box (needs the npz of the fit)')
    return p.parse_args()


def save_figure(fig, path, **kwargs):
    # saves and closes the figure; if the file is locked (open in a viewer on Windows), writes <name>_new.png instead
    path = Path(path)
    try:
        fig.savefig(path, **kwargs)
    except OSError:
        alt = path.with_name(path.stem + '_new' + path.suffix)
        fig.savefig(alt, **kwargs)
        print(f'WARNING: {path.name} is locked (open in a viewer?), figure written as {alt.name}')
    plt.close(fig)


def neighbours_per_frame(frame, xy, radii, p=2):
    # for each localisation: number of other localisations of its frame within each radius, nearest neighbour distance.
    # p = 2: in a disc; p = np.inf: in a square of half size radius (the SGD patch)
    counts = {r: np.zeros(len(frame), dtype=int) for r in radii}
    nearest = np.full(len(frame), np.inf)
    order = np.argsort(frame, kind='stable')
    frames, starts = np.unique(frame[order], return_index=True)
    for idx in np.split(order, starts[1:]):
        if len(idx) < 2:
            continue
        tree = cKDTree(xy[idx])
        for r in radii:
            counts[r][idx] = np.array([len(n) - 1 for n in tree.query_ball_point(xy[idx], r, p=p)])
        d, _ = tree.query(xy[idx], k=2)
        nearest[idx] = d[:, 1]
    return counts, nearest


def map_2d(x, y, values, bin_um, extent, statistic='mean'):
    nx, ny = int(np.ceil(extent[0]/bin_um)), int(np.ceil(extent[1]/bin_um))
    edges = (np.linspace(0, nx*bin_um, nx + 1), np.linspace(0, ny*bin_um, ny + 1))
    if statistic == 'count':
        h, _, _ = np.histogram2d(x, y, bins=edges)
        return h, edges
    s, _, _ = np.histogram2d(x, y, bins=edges, weights=values)
    n, _, _ = np.histogram2d(x, y, bins=edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s/n, np.nan), edges


def show_map(ax, m, edges, title, label, cmap='viridis', vmin=None, vmax=None):
    im = ax.pcolormesh(edges[1], edges[0], m, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xlabel('col (um)'); ax.set_ylabel('row (um)'); ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label=label, fraction=0.046)


def run_parameters(config_file):
    # overrides of SGD_jax.py (calibration and planes of the run) and interplane, from the config.txt of a run
    cfg = {}
    for line in Path(config_file).read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, _, v = line.partition('=')
            cfg[k.strip()] = v.strip()
    overrides = {k: cfg[k] for k in ('middle_plane', 'rotation', 'rotation2', 'rho_offset', 'unitary_J', 'y_gain_J1',
                                     'y_gain_J2', 'loss2') if k in cfg}
    for k in ('J1', 'J2'):
        if k in cfg:
            overrides[k] = 'np.array(' + cfg[k] + ')'
    return overrides, float(cfg['interplane']), float(cfg.get('rho_offset', 0))


def make_examples(data, fit_file, groups, out, alt_rho=None):
    # data and fit of the localisations of groups = {folder name: [(k, frame, x, y), ...]}, model of SGD_jax.py.
    # alt_rho = {folder name: rho} or {folder name: {'rho': rho, 'eta': eta}}: one more row, the model with the same
    # fitted parameters but rho (and eta) replaced by these values, and its deviance: what the fit would give if it had
    # converged there
    alt_rho = {k: v if isinstance(v, dict) else {'rho': v} for k, v in (alt_rho or {}).items()}
    import os, sys, tempfile, argparse as ap
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import SGD_precision_claude as S
    import jax.numpy as jnp
    npz = data / ('NPZ_' + Path(fit_file).stem[4:]) # fit_<date>.csv <- NPZ_<date>
    overrides, interplane, rho_offset = run_parameters(npz / 'config.txt')
    with tempfile.TemporaryDirectory() as tmp:
        ns, _, _ = S.build_pipeline(overrides, tmp, overrides.get('loss2', 'poisson'), False,
                                    ap.Namespace(interplane=interplane, sigma_r=0., warps=None))
    conv = ns['sensitivity']/(ns['QE']*ns['EM'])
    NPSF, dim = int(ns['NPSF']), int(ns['dim_simu'])
    keys = ('frame', 'x', 'y', 'z', 'N_photons', 'rho', 'eta', 'delta', 'deviance', 'x_start', 'y_start')
    fit = {k: [] for k in keys + ('background', 'gain', 'shift')}
    for f in sorted(npz.glob('[0-9]*.npz'), key=lambda f: int(f.stem)):
        z = np.load(f)
        for k in keys:
            fit[k].append(z[k])
        fit['background'].append(z['background_array_found'].reshape(-1, 3, 2))
        fit['gain'].append(z['noise_gain']); fit['shift'].append(z['noise_shift'])
    fit = {k: np.concatenate(v) for k, v in fit.items()}
    pad = lambda v: jnp.array(np.resize(v, NPSF), dtype=jnp.float32)
    ii, jj = np.mgrid[-6:7, -6:7]
    for name, items in groups.items():
        folder = out / name
        folder.mkdir(exist_ok=True)
        for old in folder.glob(f'{name}_*.png'): # figures of a previous run (the boxes can change)
            old.unlink()
        for k, frame, fx, fy in items:
            idx = np.where((fit['frame'] == frame) & (np.abs(fit['x'] - fx) < 1e-3) & (np.abs(fit['y'] - fy) < 1e-3))[0][0]
            xs, ys = fit['x_start'][idx], fit['y_start'][idx]
            xo = (fit['x'][idx] - np.trunc(xs/120)*120)*0.9952298/1000 # fitted offset from the patch centre (um)
            yo = (fit['y'][idx] - np.trunc(ys/120)*120)*0.9952298/1000
            M = ns['compute_M_jax'](xp=pad(xo), yp=pad(yo), zp=pad(fit['z'][idx]/1000), d=ns['d_'], x=ns['xx'], y=ns['yy'],
                                    th1=ns['th1'], phi=ns['phi'], Ex0=ns['Ex0'], Ex1=ns['Ex1'], Ex2=ns['Ex2'], Ey0=ns['Ey0'],
                                    Ey1=ns['Ey1'], Ey2=ns['Ey2'], u=ns['u'], v=ns['v'], phase_maskx=ns['phase_mask'],
                                    phase_masky=ns['phase_mask'], zernike_base=ns['zernike_base'],
                                    zernike_coefs_x=jnp.zeros((3, 15)), zernike_coefs_y=jnp.zeros((3, 15)),
                                    second_plane=ns['second_plane'], polar_projections=ns['polar_projections'],
                                    lambd=ns['lambd'], f_tube=ns['f_tube'])
            def model_at(rho_value, eta_value):
                h = ns['PSF_jax'](rho=pad(rho_value - rho_offset), eta=pad(eta_value), delta=pad(fit['delta'][idx]),
                                  M=M, N_photons=pad(fit['N_photons'][idx]))
                h = np.array(h*ns['channel_gain']()[None, :, :, None, None])[0, :, :, dim-6:dim+7, dim-6:dim+7]
                return h + fit['background'][idx][:, :, None, None]
            model = model_at(fit['rho'][idx], fit['eta'][idx])
            img = tifffile.imread(str(data / 'reconstruction' / f'{int(frame)}.tif')).astype(float)*conv
            cut = lambda im: np.asarray(ns['extract_raw_xy'](im, np.array([-ys]), np.array([xs])))[0][::-1, :, ::-1, :]
            patch = cut(img)
            g, sh = fit['gain'][idx][:, :, None, None], fit['shift'][idx][:, :, None, None]
            def deviance_of(m):
                ms, ds = np.maximum(m + sh, 1e-3), patch + sh
                return 2*np.sum((ms - ds + np.where(ds > 0, ds*np.log(np.maximum(ds, 1e-3)/ms), 0.))/g)/patch.size
            dev = deviance_of(model)
            alt = name in alt_rho
            if alt:
                alt_label = ', '.join(f'{k} {v:g}' for k, v in alt_rho[name].items())
                model_alt = model_at(alt_rho[name].get('rho', fit['rho'][idx]), alt_rho[name].get('eta', fit['eta'][idx]))
                dev_alt = deviance_of(model_alt)
            resid = (patch - model)/np.sqrt(np.maximum(g*(model + sh), 1e-3))
            # marks: outlier pixels (> 5 sigma above the local background) and the other fits of the frame in the patch
            res_img = img - np.stack([ndimage.median_filter(ch, 15) for ch in img])
            sig = np.array([1.4826*np.median(np.abs(r - np.median(r))) for r in res_img])[:, None, None]
            outl = cut((res_img > 5*sig).astype(float)) > 0.5
            others = np.where((fit['frame'] == frame) & (np.arange(len(fit['frame'])) != idx))[0]
            d_col = (fit['x'][others] - fit['x'][idx])/120; d_row = (-fit['y'][others] + fit['y'][idx])/120
            inside = (np.abs(d_col) <= 6.5) & (np.abs(d_row) <= 6.5)
            rows = ['data', 'fit'] + ([f'fit at {alt_label}\ndeviance {dev_alt:.2f}'] if alt else []) + ['residual (sigma)']
            fig, ax = plt.subplots(len(rows), 6, figsize=(15, 2.5*len(rows)))
            vmax = np.percentile(np.concatenate([patch.ravel(), model.ravel()]), 99.5)
            for p_ in range(3):
                for q in range(2):
                    col = 2*p_ + q
                    ax[0, col].imshow(patch[p_, q], cmap='gray', vmin=0, vmax=vmax)
                    ax[1, col].imshow(model[p_, q], cmap='gray', vmin=0, vmax=vmax)
                    if alt:
                        ax[2, col].imshow(model_alt[p_, q], cmap='gray', vmin=0, vmax=vmax)
                    im = ax[-1, col].imshow(resid[p_, q], cmap='coolwarm', vmin=-5, vmax=5)
                    yy, xx = np.where(outl[p_, q] & (ii**2 + jj**2 > 9))
                    ax[0, col].scatter(xx, yy, s=40, facecolors='none', edgecolors='r', lw=1.2)
                    ax[-1, col].scatter(xx, yy, s=40, facecolors='none', edgecolors='k', lw=1.2)
                    ax[0, col].scatter(6 + d_col[inside], 6 - d_row[inside], s=80, marker='+', c='cyan', lw=1.5)
                    ax[0, col].set_title(f'plane {p_}, {"xy"[q]}', fontsize=10)
                    for r in range(len(rows)):
                        ax[r, col].set_xticks([]); ax[r, col].set_yticks([])
            for r, lab in enumerate(rows):
                ax[r, 0].set_ylabel(lab, fontsize=11)
            fig.colorbar(im, ax=ax[-1, :].tolist(), fraction=0.02, label='(data - fit)/sqrt(var)')
            fig.suptitle(f'{name}: {k} | deviance {fit["deviance"][idx]:.2f} (recomputed {dev:.2f}) | frame {int(frame)}, '
                         f'N {fit["N_photons"][idx]:.0f}, rho {fit["rho"][idx]:.0f}, eta {fit["eta"][idx]:.0f}, '
                         f'delta {fit["delta"][idx]:.0f}, z {fit["z"][idx]:.0f} nm | red circles: outlier pixels, '
                         f'cyan +: other fits of the frame', fontsize=10)
            save_figure(fig, folder / f'{name}_{k:02d}.png', dpi=100, bbox_inches='tight')
            print(f'   {name} {k:2d}: deviance saved {fit["deviance"][idx]:.3f}, recomputed {dev:.3f}'
                  + (f', at {alt_label} {dev_alt:.3f}' if alt else ''))


def eta_and_deviance_figure(path, counts, dev, eta, xlabel, min_box, what):
    # mean eta (+/- standard error, top) and box plot of the deviance (bottom) for each integer value of counts with at
    # least min_box localisations, on a linear axis; boxes coloured by their number of localisations (log scale)
    values = [k for k in range(int(counts.max()) + 1) if np.sum(counts == k) >= min_box]
    fig, (ax_eta, ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6]))
    eta_mean = [np.nanmean(eta[counts == k]) for k in values]
    eta_sem = [np.nanstd(eta[counts == k], ddof=1)/np.sqrt(np.sum(np.isfinite(eta[counts == k]))) for k in values]
    ax_eta.errorbar(values, eta_mean, yerr=eta_sem, fmt='o-', capsize=3)
    ax_eta.set(ylabel='mean $\\eta$ (deg)', title=f'mean $\\eta$ (+/- standard error) and fit quality vs {what} '
               f'(at least {min_box} localisations)')
    ax_eta.grid(alpha=0.3)
    n_k = np.array([np.sum(counts == k) for k in values])
    norm_k = matplotlib.colors.LogNorm(vmin=n_k.min(), vmax=max(n_k.max(), n_k.min() + 1))
    boxes = ax.boxplot([dev[counts == k] for k in values], positions=values, widths=0.6, showfliers=False,
                       patch_artist=True, medianprops=dict(color='k'), manage_ticks=False)
    for box, c_ in zip(boxes['boxes'], n_k):
        box.set_facecolor(plt.cm.viridis(norm_k(c_)))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_k, cmap='viridis'), ax=[ax_eta, ax], label='localisations per box',
                 fraction=0.03, pad=0.02)
    ax.set_xlim(-1, max(values) + 1)
    ax.set(xlabel=xlabel, ylabel='deviance')
    ax.grid(alpha=0.3)
    save_figure(fig, path, dpi=110, bbox_inches='tight')
    return values, eta_mean


def main():
    a = parse_arguments()
    data = Path(a.data)
    out = data / 'density_analysis'
    out.mkdir(exist_ok=True)
    lines = []
    def report(text):
        print(text); lines.append(text)

    # ------------------------------------------------------------- 1. density (pre-localisation)
    pre_file = sorted(glob.glob(str(data / 'reconstruction' / 'pre_loc_deltaz_*.csv')))[-1]
    pre = pd.read_csv(pre_file)
    fr = pre['frame'].values
    xy = pre[['row', 'col']].values/1000 # um
    first_frame = sorted(glob.glob(str(data / 'reconstruction' / '*.tif')), key=lambda f: int(Path(f).stem) if Path(f).stem.isdigit() else -1)
    shape = tifffile.imread(first_frame[-1]).shape # (6, rows, cols)
    extent = (shape[1]*PIXEL, shape[2]*PIXEL)
    n_frames_total = fr.max() - fr.min() + 1
    report(f'{data}: {len(pre)} detections in {Path(pre_file).name}, frames {fr.min()} to {fr.max()}, field {extent[0]:.1f} x {extent[1]:.1f} um')
    counts, nearest = neighbours_per_frame(fr, xy, (a.r_patch, a.r_neighbour))
    in_patch = counts[a.r_patch] > 0
    near = counts[a.r_neighbour] > 0
    per_frame = np.bincount(fr - fr.min(), minlength=n_frames_total)
    report(f'detections per frame: mean {per_frame.mean():.1f}, median {np.median(per_frame):.0f}, 95th percentile {np.percentile(per_frame, 95):.0f}; '
           f'density {per_frame.mean()/(extent[0]*extent[1]):.3f} per um^2 and per frame')
    report(f'detections with a neighbour in the same frame: closer than {a.r_patch} um (inside the patch) {in_patch.mean()*100:.1f} %, '
           f'closer than {a.r_neighbour} um {near.mean()*100:.1f} %; nearest neighbour median {np.median(nearest[np.isfinite(nearest)]):.2f} um')

    # curves over the frames
    tb = (fr - fr.min())//a.frame_bin
    t_frames = fr.min() + (np.arange(tb.max() + 1) + 0.5)*a.frame_bin
    det_t = np.bincount(tb)/a.frame_bin
    patch_t = np.bincount(tb, weights=in_patch)/np.maximum(np.bincount(tb), 1)
    near_t = np.bincount(tb, weights=near)/np.maximum(np.bincount(tb), 1)
    thirds = np.array_split(np.arange(len(t_frames)), 3)
    report('over the acquisition (first / middle / last third): detections per frame ' +
           ' / '.join(f'{det_t[t].mean():.1f}' for t in thirds) + f'; with a neighbour < {a.r_patch} um ' +
           ' / '.join(f'{patch_t[t].mean()*100:.1f} %' for t in thirds))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(t_frames, det_t); ax[0].set(xlabel='frame', ylabel='detections per frame', title='density over the frames')
    ax[1].plot(t_frames, patch_t*100, label=f'< {a.r_patch} um (patch)'); ax[1].plot(t_frames, near_t*100, label=f'< {a.r_neighbour} um')
    ax[1].set(xlabel='frame', ylabel='% of detections with a neighbour', title='neighbours in the same frame'); ax[1].legend()
    for t, name in zip(thirds, ('first third', 'middle third', 'last third')):
        sel = np.isin(tb, t) & np.isfinite(nearest)
        ax[2].hist(nearest[sel], bins=np.linspace(0, 8, 81), histtype='step', density=True, label=name)
    ax[2].axvline(a.r_patch, color='k', ls='--', lw=1); ax[2].axvline(a.r_neighbour, color='k', ls=':', lw=1)
    ax[2].set(xlabel='nearest neighbour in the same frame (um)', ylabel='density', title='nearest neighbour distance'); ax[2].legend()
    fig.tight_layout(); save_figure(fig, out / 'density_over_frames.png', dpi=110)

    # maps across the field
    count_map, edges = map_2d(xy[:, 0], xy[:, 1], None, a.bin, extent, 'count')
    dens_map = count_map/(a.bin**2*n_frames_total)
    patch_map, _ = map_2d(xy[:, 0], xy[:, 1], in_patch.astype(float), a.bin, extent)
    near_map, _ = map_2d(xy[:, 0], xy[:, 1], near.astype(float), a.bin, extent)
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    show_map(ax[0], dens_map*1000, edges, f'density (bins {a.bin} um)', 'detections per um$^2$ per 1000 frames', 'magma')
    show_map(ax[1], patch_map*100, edges, f'% with a neighbour < {a.r_patch} um', '%', 'viridis', 0, np.nanpercentile(patch_map*100, 98))
    show_map(ax[2], near_map*100, edges, f'% with a neighbour < {a.r_neighbour} um', '%', 'viridis', 0, np.nanpercentile(near_map*100, 98))
    fig.tight_layout(); save_figure(fig, out / 'density_maps.png', dpi=110)
    report(f'across the field: density per um^2 per 1000 frames median {np.median(dens_map[dens_map > 0])*1000:.2f}, '
           f'95th percentile {np.percentile(dens_map[dens_map > 0], 95)*1000:.2f}, max {dens_map.max()*1000:.2f}; '
           f'% with a neighbour < {a.r_patch} um: median {np.nanmedian(patch_map)*100:.1f} %, 95th percentile {np.nanpercentile(patch_map, 95)*100:.1f} %')

    # ------------------------------------------------------------- 2. fit quality vs density
    fits = sorted(glob.glob(str(data / 'fit_*.csv')))
    if fits:
        fit = pd.read_csv(fits[-1], comment='#', sep=';', usecols=['frame', 'x', 'y', 'eta', 'deviance', 'N_photon'])
        fit = fit[np.isfinite(fit.deviance)].reset_index(drop=True)
        n_fit = len(fit)
        # the same emitter fitted twice in a frame (detected twice by the pre-localisation): fits closer than
        # r_duplicate, only the one with the lowest deviance is kept (else each counts the other as a neighbour)
        duplicate = np.zeros(len(fit), dtype=bool)
        for idx in fit.groupby('frame').indices.values():
            if len(idx) < 2:
                continue
            for i, j in cKDTree(fit[['x', 'y']].values[idx]/1000).query_pairs(a.r_duplicate):
                i, j = idx[i], idx[j]
                if not (duplicate[i] or duplicate[j]):
                    duplicate[j if fit.deviance[j] >= fit.deviance[i] else i] = True
        fit = fit[~duplicate].reset_index(drop=True)
        # neighbours whose centre is in the SGD patch (13 x 13 px: square of half size 6.5 px = r_patch)
        c, _ = neighbours_per_frame(fit.frame.values.astype(int), fit[['x', 'y']].values/1000, (a.r_patch,), p=np.inf)
        n_neigh = c[a.r_patch]
        report(f'fit {Path(fits[-1]).name}: {n_fit} localisations (frames {int(fit.frame.min())}-{int(fit.frame.max())}), '
               f'{duplicate.sum()} duplicates removed (same frame, < {a.r_duplicate*1000:.0f} nm: the same emitter fitted twice)')
        for k in range(n_neigh.max() + 1):
            sel = n_neigh == k
            if sel.sum():
                report(f'   {k} fitted neighbours in the patch: {sel.sum()} localisations ({sel.mean()*100:.1f} %), '
                       f'median deviance {np.median(fit.deviance[sel]):.3f}, deviance > 1.4: {np.mean(fit.deviance[sel] > 1.4)*100:.1f} %, '
                       f'mean eta {np.mean(fit.eta[sel]):.1f}')
        eta_and_deviance_figure(out / 'fit_quality_vs_neighbours.png', n_neigh, fit.deviance.values, fit.eta.values,
                                f'fitted neighbours in the patch (centre within {a.r_patch} um in x and y, same frame)',
                                a.min_box, 'neighbours in the patch')
        # deviance across the field, in the frame of the pre-localisation (row = -y, col = x of the fit): the same
        # orientation as the density maps
        dev_map, edges_f = map_2d(-fit.y.values/1000, fit.x.values/1000, fit.deviance.values, a.bin, extent)
        fig, ax = plt.subplots(figsize=(6, 6))
        show_map(ax, dev_map, edges_f, f'mean deviance across the field (bins {a.bin} um)', 'deviance', 'coolwarm',
                 np.nanpercentile(dev_map, 2), np.nanpercentile(dev_map, 98))
        fig.tight_layout(); save_figure(fig, out / 'deviance_map.png', dpi=110)

    # ------------------------------------------------------------- 3. outlier pixels in the background
    tifs = {int(Path(f).stem): f for f in glob.glob(str(data / 'reconstruction' / '*.tif')) if Path(f).stem.isdigit()}
    sample = np.unique(np.linspace(fr.min(), fr.max(), a.n_frames).astype(int))
    sample = [s for s in sample if s in tifs]
    by_frame = dict(tuple(pre.groupby('frame')))
    rate, hits = [], np.zeros(shape[1:])
    looked = np.zeros(shape[1:])
    for s in sample:
        img = tifffile.imread(tifs[s]).astype(float) # (6, rows, cols)
        mask = np.zeros(shape[1:], dtype=bool)
        if s in by_frame:
            rc = np.round(by_frame[s][['row', 'col']].values/120).astype(int)
            for r, c in rc:
                mask[max(0, r - a.mask_radius):r + a.mask_radius + 1, max(0, c - a.mask_radius):c + a.mask_radius + 1] = True
        border = np.zeros_like(mask); border[:8] = border[-8:] = True; border[:, :8] = border[:, -8:] = True
        free = ~mask & ~border
        n_out = 0
        for ch in img:
            res = ch - ndimage.median_filter(ch, size=15) # local background removed
            sigma = 1.4826*np.median(np.abs(res[free] - np.median(res[free])))
            out_px = free & (res > a.n_sigma*sigma)
            n_out += out_px.sum(); hits += out_px
        looked += free*len(img)
        rate.append(n_out/(free.sum()*len(img)))
    rate = np.array(rate)
    report(f'outlier pixels in the background (> {a.n_sigma} robust sigma above the local background, {len(sample)} frames, '
           f'{a.mask_radius} px masked around each detection): {rate.mean()*1e3:.2f} per 1000 pixels '
           f'(first / middle / last third ' + ' / '.join(f'{t.mean()*1e3:.2f}' for t in np.array_split(rate, 3)) +
           f'); for a Gaussian noise one expects {0.5*(1 - __import__("math").erf(a.n_sigma/np.sqrt(2)))*1e3:.4f}')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(sample, rate*1e3, '.-'); ax[0].set(xlabel='frame', ylabel='outlier pixels per 1000', title=f'background outliers (> {a.n_sigma} sigma)')
    with np.errstate(invalid='ignore', divide='ignore'):
        omap = np.where(looked > 0, hits/looked, np.nan)
    im = ax[1].imshow(ndimage.uniform_filter(np.nan_to_num(omap), 9)*1e3, cmap='magma', extent=(0, extent[1], extent[0], 0))
    ax[1].set(xlabel='col (um)', ylabel='row (um)', title='background outliers across the field (smoothed)')
    plt.colorbar(im, ax=ax[1], label='per 1000 pixels', fraction=0.046)
    fig.tight_layout(); save_figure(fig, out / 'background_outliers.png', dpi=110)

    # ------------------------------------------------------------- 4. fit quality vs outlier pixels in the patch
    cache = out / 'outliers_per_patch.npz'
    use_cache = fits and cache.exists() and not a.recompute_outliers and 'frame' in np.load(cache).files # older caches
    if use_cache:                                                                                     # had no positions
        z = np.load(cache)
        n_out, dev, n_pick = z['n_out'], z['deviance'], int(z['n_frames'])
        where = np.column_stack((z['frame'], z['x'], z['y']))
        print(f'outlier pixels of the patches read from {cache}')
    elif fits:
        fit_all = pd.read_csv(fits[-1], comment='#', sep=';', usecols=['frame', 'x', 'y', 'deviance'])
        fit_all = fit_all[np.isfinite(fit_all.deviance)]
        fit_frames = np.unique(fit_all.frame.values.astype(int))
        pick = fit_frames[np.unique(np.linspace(0, len(fit_frames) - 1, a.n_fit_frames).astype(int))]
        pick = [f for f in pick if f in tifs]
        ii, jj = np.mgrid[-6:7, -6:7]
        outside_core = ii**2 + jj**2 > a.core_radius**2
        by_fit_frame = dict(tuple(fit_all.groupby(fit_all.frame.astype(int))))
        n_out, dev, where = [], [], [] # where: frame, x, y of the fit (to find the localisation again)
        for count, f in enumerate(pick):
            if count % 100 == 0:
                print(f'   outlier pixels in the patches: frame {count}/{len(pick)}', flush=True)
            img = tifffile.imread(tifs[f]).astype(float)
            outl = np.zeros(img.shape, dtype=bool)
            for k, ch in enumerate(img):
                res = ch - ndimage.median_filter(ch, size=15)
                sigma = 1.4826*np.median(np.abs(res - np.median(res)))
                outl[k] = res > a.n_sigma*sigma
            sel = by_fit_frame[f]
            rows = np.round(-sel.y.values/120).astype(int); cols = np.round(sel.x.values/120).astype(int)
            for r, c, d, fx, fy in zip(rows, cols, sel.deviance.values, sel.x.values, sel.y.values):
                if 6 <= r < shape[1] - 6 and 6 <= c < shape[2] - 6:
                    n_out.append(int(np.sum(outl[:, r-6:r+7, c-6:c+7] & outside_core))); dev.append(d); where.append((f, fx, fy))
        n_out, dev, n_pick, where = np.array(n_out), np.array(dev), len(pick), np.array(where)
        np.savez(cache, n_out=n_out, deviance=dev, n_frames=n_pick, frame=np.array(where)[:, 0], x=np.array(where)[:, 1],
                 y=np.array(where)[:, 2])
    if fits:
        edges_n = [0, 1, 2, 3, 4, 5, 6, 8, 11, 16, 26, 41, 10**6]
        labels_n = ['0', '1', '2', '3', '4', '5', '6-7', '8-10', '11-15', '16-25', '26-40', '>40']
        report(f'fit quality vs outlier pixels in the patch ({n_pick} frames, {len(dev)} localisations, outliers > '
               f'{a.n_sigma} sigma over the 6 channels out of a {a.core_radius:g} px disc): median {np.median(n_out):.0f}, '
               f'mean {n_out.mean():.1f}; Spearman correlation with the deviance {pd.Series(n_out).corr(pd.Series(dev), method="spearman"):.2f}')
        med, q1, q3, cnt = [], [], [], []
        for lo, hi, lab in zip(edges_n[:-1], edges_n[1:], labels_n):
            sel = (n_out >= lo) & (n_out < hi)
            cnt.append(sel.sum())
            med.append(np.median(dev[sel]) if sel.any() else np.nan)
            q1.append(np.percentile(dev[sel], 25) if sel.any() else np.nan); q3.append(np.percentile(dev[sel], 75) if sel.any() else np.nan)
            if sel.any():
                report(f'   {lab:>5s} outlier pixels: {sel.sum():6d} localisations ({sel.mean()*100:4.1f} %), median deviance {med[-1]:.3f}, '
                       f'deviance > 1.4: {np.mean(dev[sel] > 1.4)*100:.1f} %')
        # eta of each localisation, from the fit (found again by its frame and position)
        eta_fit = pd.read_csv(fits[-1], comment='#', sep=';', usecols=['frame', 'x', 'y', 'eta'])
        eta_of = pd.merge(pd.DataFrame({'frame': where[:, 0], 'x': where[:, 1], 'y': where[:, 2]}), eta_fit,
                          on=['frame', 'x', 'y'], how='left').eta.values
        values, eta_mean = eta_and_deviance_figure(out / 'fit_quality_vs_outliers.png', n_out, dev, eta_of,
                                                   'outlier pixels in the patch (out of the PSF core)', a.min_box, 'outlier pixels')
        report('   mean eta per number of outlier pixels: ' + ', '.join(f'{k}: {m:.1f}' for k, m in zip(values, eta_mean)))

    # ------------------------------------------------------------- 5. examples of data and fit for each box
    if fits and a.examples:
        def typical(counts, dev, frames, xs, ys):
            # for each value with at least min_box localisations: the one with the deviance closest to the box median
            items = []
            for k in range(int(counts.max()) + 1):
                sel = np.where(counts == k)[0]
                if len(sel) >= a.min_box:
                    i = sel[np.argmin(np.abs(dev[sel] - np.median(dev[sel])))]
                    items.append((k, frames[i], xs[i], ys[i]))
            return items
        groups = {'neighbours': typical(n_neigh, fit.deviance.values, fit.frame.values, fit.x.values, fit.y.values),
                  'outliers': typical(n_out, dev, where[:, 0], where[:, 1], where[:, 2])}
        print('examples of data and fit:')
        make_examples(data, fits[-1], groups, out)

    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print(f'figures and summary in {out}')


if __name__ == '__main__':
    main()
