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
    return p.parse_args()


def neighbours_per_frame(frame, xy, radii):
    # for each localisation: number of other localisations of its frame within each radius, nearest neighbour distance
    counts = {r: np.zeros(len(frame), dtype=int) for r in radii}
    nearest = np.full(len(frame), np.inf)
    order = np.argsort(frame, kind='stable')
    frames, starts = np.unique(frame[order], return_index=True)
    for idx in np.split(order, starts[1:]):
        if len(idx) < 2:
            continue
        tree = cKDTree(xy[idx])
        for r in radii:
            counts[r][idx] = np.array([len(n) - 1 for n in tree.query_ball_point(xy[idx], r)])
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
    fig.tight_layout(); fig.savefig(out / 'density_over_frames.png', dpi=110); plt.close(fig)

    # maps across the field
    count_map, edges = map_2d(xy[:, 0], xy[:, 1], None, a.bin, extent, 'count')
    dens_map = count_map/(a.bin**2*n_frames_total)
    patch_map, _ = map_2d(xy[:, 0], xy[:, 1], in_patch.astype(float), a.bin, extent)
    near_map, _ = map_2d(xy[:, 0], xy[:, 1], near.astype(float), a.bin, extent)
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    show_map(ax[0], dens_map*1000, edges, f'density (bins {a.bin} um)', 'detections per um$^2$ per 1000 frames', 'magma')
    show_map(ax[1], patch_map*100, edges, f'% with a neighbour < {a.r_patch} um', '%', 'viridis', 0, np.nanpercentile(patch_map*100, 98))
    show_map(ax[2], near_map*100, edges, f'% with a neighbour < {a.r_neighbour} um', '%', 'viridis', 0, np.nanpercentile(near_map*100, 98))
    fig.tight_layout(); fig.savefig(out / 'density_maps.png', dpi=110); plt.close(fig)
    report(f'across the field: density per um^2 per 1000 frames median {np.median(dens_map[dens_map > 0])*1000:.2f}, '
           f'95th percentile {np.percentile(dens_map[dens_map > 0], 95)*1000:.2f}, max {dens_map.max()*1000:.2f}; '
           f'% with a neighbour < {a.r_patch} um: median {np.nanmedian(patch_map)*100:.1f} %, 95th percentile {np.nanpercentile(patch_map, 95)*100:.1f} %')

    # ------------------------------------------------------------- 2. fit quality vs density
    fits = sorted(glob.glob(str(data / 'fit_*.csv')))
    if fits:
        fit = pd.read_csv(fits[-1], comment='#', sep=';', usecols=['frame', 'x', 'y', 'deviance', 'N_photon'])
        fit = fit[np.isfinite(fit.deviance)]
        c, _ = neighbours_per_frame(fit.frame.values.astype(int), fit[['x', 'y']].values/1000, (a.r_neighbour,))
        n = np.minimum(c[a.r_neighbour], 4)
        report(f'fit {Path(fits[-1]).name}: {len(fit)} localisations (frames {int(fit.frame.min())}-{int(fit.frame.max())})')
        rows = []
        for k in range(5):
            sel = n == k
            if sel.sum():
                rows.append((k, sel.sum(), np.median(fit.deviance[sel]), np.mean(fit.deviance[sel] > 1.4)))
                report(f'   {k}{"+" if k == 4 else ""} fitted neighbours < {a.r_neighbour} um: {sel.sum()} localisations ({sel.mean()*100:.1f} %), '
                       f'median deviance {rows[-1][2]:.3f}, deviance > 1.4: {rows[-1][3]*100:.1f} %')
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].boxplot([fit.deviance[n == k] for k, *_ in rows], labels=[f'{k}{"+" if k == 4 else ""}' for k, *_ in rows], showfliers=False)
        ax[0].set(xlabel=f'fitted neighbours < {a.r_neighbour} um in the frame', ylabel='deviance', title='fit quality vs neighbours')
        # the fit coordinates are not the ones of the pre-localisation (SGD_jax.py swaps and flips the axes): map
        # from their own minimum
        fx, fy = (fit.x.values - fit.x.min())/1000, (fit.y.values - fit.y.min())/1000
        dev_map, edges_f = map_2d(fx, fy, fit.deviance.values, a.bin, (fx.max() + 1e-6, fy.max() + 1e-6))
        show_map(ax[1], dev_map, edges_f, 'mean deviance across the field (fit coordinates)', 'deviance', 'coolwarm',
                 np.nanpercentile(dev_map, 2), np.nanpercentile(dev_map, 98))
        ax[1].set_xlabel('y - min (um)'); ax[1].set_ylabel('x - min (um)')
        fig.tight_layout(); fig.savefig(out / 'fit_quality_vs_density.png', dpi=110); plt.close(fig)

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
    fig.tight_layout(); fig.savefig(out / 'background_outliers.png', dpi=110); plt.close(fig)

    # ------------------------------------------------------------- 4. fit quality vs outlier pixels in the patch
    cache = out / 'outliers_per_patch.npz'
    if fits and cache.exists() and not a.recompute_outliers:
        z = np.load(cache)
        n_out, dev, n_pick = z['n_out'], z['deviance'], int(z['n_frames'])
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
        n_out, dev = [], []
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
            for r, c, d in zip(rows, cols, sel.deviance.values):
                if 6 <= r < shape[1] - 6 and 6 <= c < shape[2] - 6:
                    n_out.append(int(np.sum(outl[:, r-6:r+7, c-6:c+7] & outside_core))); dev.append(d)
        n_out, dev, n_pick = np.array(n_out), np.array(dev), len(pick)
        np.savez(cache, n_out=n_out, deviance=dev, n_frames=n_pick)
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
        # box plot of the deviance for each number of outlier pixels, at its value on a linear axis
        values = [k for k in range(n_out.max() + 1) if np.sum(n_out == k) >= a.min_box]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.boxplot([dev[n_out == k] for k in values], positions=values, widths=0.6, showfliers=False,
                   medianprops=dict(color='C1'), manage_ticks=False)
        ax.set_xlim(-1, max(values) + 1)
        ax.set(xlabel='outlier pixels in the patch (out of the PSF core)', ylabel='deviance',
               title=f'fit quality vs outlier pixels (boxes with at least {a.min_box} localisations)')
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / 'fit_quality_vs_outliers.png', dpi=110); plt.close(fig)

    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print(f'figures and summary in {out}')


if __name__ == '__main__':
    main()
