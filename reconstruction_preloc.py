# RECONSTRUCTION + PRE-LOCALISATION without the GUI (Claude, 2026-11-13; user: "everything in the reconstruction script, the
# last way we did: do not miss emitters and combine"). Headless counterpart of data_preprocessing.py: calls the functions
# of reconstruction_functions.py (was reconstruction_juliette.py). Same steps and settings as the last data set processed
# (cricket sperm 2026-11-06, polMFM_runs/2026_11_06_cricksperm/2026-11-10_cricket_local/run_cricket_local.sh); the code
# of the four headless scripts it ran is merged here unchanged in what it computes:
#   reconstruction_headless.py (2026-11-03), preloc_headless.py (2026-11-03), preloc_sum.py (2026-11-06),
#   combine_preloc.py (2026-11-06), all from D:\AMAURY\DATA\1_nuclei\claude_scripts.
#
# 1. RECONSTRUCTION: reconstruction_stack of reconstruction_functions.py, its dialogs answered by the arguments (code of
#    the reconstruction not changed): baseline, 6 channels, warps of the calibration -> reconstruction/<frame>.tif,
#    dark_variance.txt, calibration_used.txt. The raw files image_Pos0.ome.tif, image_Pos0_1.ome.tif, ... in this order.
#    The calibration is copied into <RAW_DATA>/calibr/ when it is elsewhere.
# 2. OLD PRE-LOC (method of reconstruction_functions.py, find_pre_loc / pre_loc_dataset): islands above ONE vmin in the background-subtracted max
#    projection of the 3 planes; vmin tuned on SAMPLE frames so that most frames have LO-HI emitters; islands of up to
#    OLD_MAX_SIZE px (40: the brightest molecules too); fixed objects (> 20 x the bright pixels in > 30 % of the frames)
#    masked (+ 12 px). -> reconstruction/preloc_old/pre_loc_deltaz_<dz>.csv, preloc_vmin.png / .txt, preloc_mask.npy
# 3. NEW PRE-LOC (sum of the 6 channels: the light of a molecule spread over planes and polarisations adds up):
#    S - temporal median background (blocks of BLOCK frames, SAMPLES frames each, linear in between), smoothed
#    (gaussian SMOOTH px) = R; local maxima (7 x 7) of R above T(frame), BORDER px from the edges; sub-pixel position =
#    gaussian through the 3 pixels around the maximum. T(frame): K x the noise of R at anchor frames (lower half of R),
#    linear in between (reconstruction/preloc_sum_thresholds.txt, figure preloc_sum_anchors.png to check; the
#    thresholds were NOT edited by hand for the cricket data). A thresholds file can be given instead (--thresholds).
#    -> reconstruction/preloc_sum/pre_loc_deltaz_<dz>.csv
# 4. COMBINED LIST: the new list + the old detections the new one missed (no new one within MATCH px in the same frame)
#    that blink (old detections within NEAR px in at most MAX_ON of the frames f +- WIN: not a fixed object), not within
#    BORDER px of the edges; duplicates (< MATCH px, same frame) removed, the new one kept.
#    -> reconstruction/pre_loc_deltaz_<dz>.csv (columns frame,row,col (nm),source new/old): the ONLY csv of
#    reconstruction/, read by SGD_jax.py (interplane from its name, <dz> = delta z of the calibration, nm).
#
# usage: python reconstruction_preloc.py <RAW_DATA folder> <calibration.npz> [--thresholds <file>] [--skip-reconstruction]
#   --skip-reconstruction: reconstruction/ exists already (e.g. to redo the pre-loc with thresholds edited by hand)
# Without it, an existing reconstruction/ is not overwritten (stop). The calibration (calibration.npz) is made before, with
# data_preprocessing.py (calibration cell).
import sys, os, re, csv, shutil
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruction_functions as rj

# settings of the cricket run (2026-11-10)
SAMPLE, LO, HI, OLD_MAX_SIZE = 500, 1, 10, 40              # old pre-loc (PRELOC_MAX_SIZE=40 for the cricket data)
K, ANCHORS = 6., [0, 1000, 10000, 30000, 40000]             # new pre-loc: threshold = K x noise at the anchor frames
BLOCK, SAMPLES, SMOOTH, BORDER = 500, 50, 1.5, 6            # new pre-loc: background, smoothing, edge
MATCH, NEAR, WIN, MAX_ON = 2., 1.5, 250, 100                # combination (MAX_ON 10 until 2026-11-06 evening)
SHAPE = (215, 160)                                          # channel images (rows, cols), for the edge of the old ones


# ------------------------------------------------------------------ 1. reconstruction (reconstruction_headless.py)
def reconstruct(raw, calib):
    files = sorted(raw.glob('image_Pos0*.ome.tif'), key=lambda p: int(m.group(1)) if (m := re.search(r'_(\d+)\.ome\.tif$', p.name)) else 0)
    print('files:', [f.name for f in files])
    if (raw / 'reconstruction').exists():
        sys.exit(f'{raw / "reconstruction"} exists already: not overwritten (--skip-reconstruction to only redo the pre-loc)')
    (raw / 'calibr').mkdir(exist_ok=True)
    for name in ('calibration.npz', 'registration_z.npz', 'registration_z.png', 'calibration_beads.png'):
        if (calib.parent / name).exists() and (calib.parent / name).resolve() != (raw / 'calibr' / name).resolve():
            shutil.copy2(calib.parent / name, raw / 'calibr' / name)

    class NoWindow:
        def __init__(self, *a, **k): pass
        def __getattr__(self, name): return lambda *a, **k: None
    rj.tk.Tk = NoWindow
    rj.messagebox.askyesno = lambda *a, **k: True
    rj.filedialog.askopenfilename = lambda *a, **k: str(raw / 'calibr' / 'calibration.npz')
    rj.filedialog.askopenfilenames = lambda *a, **k: [str(f) for f in files]
    rj.reconstruction_stack(str(raw))
    n = len([p for p in (raw / 'reconstruction').glob('*.tif') if p.stem.isdigit()])
    print(f'{n} reconstructed frames in {raw / "reconstruction"}')


# ------------------------------------------------------------------ 2. old pre-loc (preloc_headless.py)
def preloc_old(raw, deltaz, out):
    rec = raw / 'reconstruction'
    frames = sorted(int(p.stem) for p in rec.glob('*.tif') if p.stem.isdigit())
    pick = np.unique(np.linspace(0, len(frames) - 1, SAMPLE).astype(int))

    def mip(i):  # background-subtracted maximum projection of the 3 planes, as in pre_loc_single_frame
        fr = tifffile.imread(rec / f'{frames[i]}.tif')
        m = np.maximum(np.maximum(fr[0] + fr[1], fr[2] + fr[3]), fr[4] + fr[5])
        return np.clip(m - rj.uniform_filter(m, size=100), 0, None)
    mips = [mip(i) for i in pick]
    bright = np.median([np.percentile(m, 99) for m in mips])
    persistent = np.mean([m > 20*bright for m in mips], axis=0) > 0.3
    mask = rj.ndimage.binary_dilation(persistent, iterations=12)
    np.save(rec / 'preloc_mask.npy', mask)
    if mask.any():
        lab, n = rj.ndimage.label(persistent)
        print(f'{n} fixed object(s) masked at (row, col)', [tuple(np.round(c).astype(int)) for c in rj.ndimage.center_of_mass(persistent, lab, range(1, n + 1))], f'({mask.sum()} px)')
    mips = [np.where(mask, 0, m) for m in mips]
    single = rj.pre_loc_single_frame

    def pre_loc_masked(frame, vmin, **k):
        c = single(frame, vmin, **k)  # nm (row, col)
        if len(c) == 0 or not mask.any():
            return c
        r, q = np.clip(np.round(c/120).astype(int), 0, np.array(mask.shape) - 1).T
        return c[~mask[r, q]]
    rj.pre_loc_single_frame = pre_loc_masked
    grid = np.unique(np.round(np.geomspace(np.percentile([m.max() for m in mips], 5)/20, np.percentile([m.max() for m in mips], 95), 60), 1))
    counts = np.array([[len(rj.find_pre_loc(m, v, max_size=OLD_MAX_SIZE)[0]) for m in mips] for v in grid])  # (vmin, frame)
    outside = np.mean((counts < LO) | (counts > HI), axis=1)
    best = outside.min(); vmin = grid[np.where(outside <= best + 1e-9)[0].max()]
    c = np.array([len(rj.find_pre_loc(m, vmin, max_size=OLD_MAX_SIZE)[0]) for m in mips])
    summary = (f'max_size {OLD_MAX_SIZE}, vmin {vmin} on {len(pick)} frames out of {len(frames)}: {np.mean(c < LO)*100:.0f} % frames with < {LO}, '
               f'{np.mean(c > HI)*100:.0f} % with > {HI}, median {np.median(c):.0f} per frame, mean {c.mean():.1f} '
               f'(-> ~{c.mean()*len(frames):.0f} pre-localisations)')
    print(summary); (rec / 'preloc_vmin.txt').write_text(summary + '\n')
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].semilogx(grid, outside*100, 'o-'); ax[0].axvline(vmin, color='r'); ax[0].set(xlabel='vmin', ylabel=f'% frames outside [{LO}, {HI}]')
    ax[1].plot(np.array(frames)[pick], c, '.', ms=3); ax[1].axhline(LO, color='k', lw=.5); ax[1].axhline(HI, color='k', lw=.5)
    ax[1].set(xlabel='frame', ylabel='pre-localisations per frame', title=f'vmin {vmin}')
    ax[2].hist(c, bins=np.arange(-.5, c.max() + 1.5)); ax[2].set(xlabel='pre-localisations per frame', ylabel='frames')
    fig.suptitle(str(raw)); fig.tight_layout(); fig.savefig(rec / 'preloc_vmin.png', dpi=110); plt.close(fig)
    rj.filedialog.askdirectory = lambda *a, **k: str(raw)
    written = rj.pre_loc_dataset(vmin, deltaz, max_size=OLD_MAX_SIZE)  # reconstruction/pre_loc_deltaz_<dz>.csv
    rj.pre_loc_single_frame = single
    out.parent.mkdir(exist_ok=True); shutil.move(str(written), str(out))
    print('old pre-loc ->', out)


# ------------------------------------------------------------------ 3. new pre-loc (preloc_sum.py)
def frame_list(rec):
    return sorted(int(p.stem) for p in rec.glob('*.tif') if p.stem.isdigit())

def read_sum(rec, f):
    return tifffile.imread(rec / f'{f}.tif').astype(np.float32).sum(0)

class Background:
    # temporal median of the summed frames, per block of BLOCK frames, cached
    def __init__(self, rec, frames):
        self.rec, self.frames, self.cache = rec, np.array(frames), {}
    def block(self, b):
        if b not in self.cache:
            sel = self.frames[(self.frames >= b*BLOCK) & (self.frames < (b + 1)*BLOCK)]
            if len(sel) == 0:
                sel = self.frames[np.argsort(np.abs(self.frames - (b + .5)*BLOCK))[:SAMPLES]]
            sel = sel[np.unique(np.linspace(0, len(sel) - 1, SAMPLES).astype(int))]
            self.cache[b] = np.median([read_sum(self.rec, f) for f in sel], axis=0)
            if len(self.cache) > 4:
                self.cache.pop(min(self.cache))
        return self.cache[b]
    def __call__(self, f):
        t = f/BLOCK - .5; b0 = int(np.floor(t)); w = t - b0
        last = (self.frames.max())//BLOCK
        b0c, b1c = min(max(b0, 0), last), min(max(b0 + 1, 0), last)
        return (1 - w)*self.block(b0c) + w*self.block(b1c) if b0c != b1c else self.block(b0c)

def residual(rec, bg, f, raw=False):
    r = read_sum(rec, f) - bg(f)
    return (ndimage.gaussian_filter(r, SMOOTH), r) if raw else ndimage.gaussian_filter(r, SMOOTH)

def noise(R):
    m = np.median(R); lo = R[R < m]
    return np.sqrt(np.mean((lo - m)**2)), m

def detect(R, T, raw=None):
    # raw: S - background not smoothed, for the position (else R)
    m = np.median(R); A = R - m if raw is None else raw - np.median(raw)
    pk = (R == ndimage.maximum_filter(R, 7)) & (R - m > T)
    pk[:BORDER] = pk[-BORDER:] = False; pk[:, :BORDER] = pk[:, -BORDER:] = False
    out = []
    L = np.log(np.maximum(R - m, 1e-3))
    for r, c in zip(*np.nonzero(pk)):
        # gaussian through the 3 pixels around the maximum of R in each direction (R is smooth: the peak is ~gaussian)
        g = lambda a, b, d: 0.5*(a - d)/(a - 2*b + d) if a - 2*b + d < 0 else 0.
        dr, dc = g(L[r-1, c], L[r, c], L[r+1, c]), g(L[r, c-1], L[r, c], L[r, c+1])
        out.append((r + np.clip(dr, -1, 1), c + np.clip(dc, -1, 1), R[r, c] - m))
    return np.array(out).reshape(-1, 3)

def thresholds(path):
    a = np.loadtxt(path, comments='#', ndmin=2); a = a[np.argsort(a[:, 0])]
    return lambda f: float(np.interp(f, a[:, 0], a[:, 1]))

def preloc_sum_suggest(rec, frames, bg):
    anchors = [min(frames, key=lambda f: abs(f - a)) for a in ANCHORS]
    lines, fig = [], plt.figure(figsize=(4*len(anchors), 9))
    for i, a in enumerate(anchors):
        sig = np.median([noise(residual(rec, bg, f))[0] for f in range(a, a + 10) if f in set(frames)])
        T = K*sig; lines.append(f'{a} {T:.1f}')
        R = residual(rec, bg, a); d = detect(R, T); dn = detect(-R + 2*np.median(R), T)
        ax = fig.add_subplot(1, len(anchors), i + 1)
        ax.imshow(np.clip(R - np.median(R), 0, None), cmap='gray', vmax=np.percentile(R, 99.9))
        ax.plot(d[:, 1], d[:, 0], 'o', mfc='none', mec='lime', ms=12)
        ax.set_title(f'frame {a}: T {T:.0f} ({K:g} x noise {sig:.0f})\n{len(d)} detections, {len(dn)} on the negative image'); ax.axis('off')
    out = rec / 'preloc_sum_thresholds.txt'
    out.write_text('# frame threshold (summed image, smoothed, background subtracted); set by hand, linear in between\n'
                   f'# suggested by preloc_sum.py suggest: {K:g} x noise at each anchor\n' + '\n'.join(lines) + '\n')
    fig.tight_layout(); fig.savefig(rec / 'preloc_sum_anchors.png', dpi=80); plt.close(fig)
    print('\n'.join(lines)); print('written', out, rec / 'preloc_sum_anchors.png')
    return out

def preloc_sum_run(rec, frames, bg, thresholds_file, out):
    Tf = thresholds(thresholds_file); lo, hi = frames[0], frames[-1]
    out.parent.mkdir(exist_ok=True); n = 0
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frame', 'row', 'col'])
        for f in frames:
            if f < lo or f > hi: continue
            Rs, Rr = residual(rec, bg, f, raw=True)
            for r, c, _ in detect(Rs, Tf(f), raw=Rr):
                w.writerow([f, r*120, c*120]); n += 1
            if f % 5000 == 0: print(f'frame {f}: {n} detections', flush=True); fh.flush()
    print(f'{n} detections in frames {lo}-{hi} ({n/max(hi - lo + 1, 1):.2f} per frame) -> {out}')


# ------------------------------------------------------------------ 4. combined list (combine_preloc.py)
def combine(old_csv, new_csv, out_csv):
    old, new = pd.read_csv(old_csv), pd.read_csv(new_csv)
    old.columns = new.columns = ['frame', 'row', 'col']
    nf = {f: g[['row', 'col']].values/120 for f, g in new.groupby('frame')}
    missed = np.ones(len(old), bool)
    for f, i in old.groupby('frame').indices.items():
        if f in nf: missed[i] = cKDTree(nf[f]).query(old[['row', 'col']].values[i]/120)[0] > MATCH
    # frames with an old detection near the same position: box of NEAR px x WIN frames (Chebyshev, frame scaled)
    X = np.column_stack((old.row/120, old.col/120, old.frame*NEAR/WIN))
    on = np.array([len(n) for n in cKDTree(X).query_ball_point(X[missed], NEAR, p=np.inf)])
    add = old[missed][on <= MAX_ON]
    # not within BORDER px of the edges (as the new pre-loc: the 13 x 13 patch of the fit)
    add = add[(add.row/120 >= BORDER) & (add.row/120 < SHAPE[0] - BORDER) & (add.col/120 >= BORDER) & (add.col/120 < SHAPE[1] - BORDER)]
    out = pd.concat([new.assign(source='new'), add.assign(source='old')]).sort_values(['frame', 'source'], kind='stable')
    keep = np.ones(len(out), bool); vals = out[['row', 'col']].values/120
    for f, i in out.groupby('frame').indices.items():  # new first in each frame: an old one closer than MATCH is dropped
        if len(i) > 1:
            for a, b in cKDTree(vals[i]).query_pairs(MATCH):
                keep[i[max(a, b)]] = False
    out = out[keep].sort_values(['frame']); out[['frame', 'row', 'col', 'source']].to_csv(out_csv, index=False)
    print(f'old {len(old)}, new {len(new)}; old missed by new {missed.sum()} ({missed.mean()*100:.1f} %), of which fixed '
          f'(> {MAX_ON} frames on within +-{WIN}) {(on > MAX_ON).sum()}, added {len(add)}; combined {len(out)} '
          f'(+{(len(out) - len(new))/len(new)*100:.1f} % vs new) -> {out_csv}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    thr = sys.argv[sys.argv.index('--thresholds') + 1] if '--thresholds' in sys.argv else None
    if thr: args.remove(thr)
    raw, calib = Path(args[0]), Path(args[1])
    if '--skip-reconstruction' not in sys.argv:
        reconstruct(raw, calib)
    rec = raw / 'reconstruction'
    for f in rec.glob('*.csv'):  # the fit reads the only csv of reconstruction/: earlier lists moved aside
        (rec / 'preloc_previous').mkdir(exist_ok=True); shutil.move(str(f), str(rec / 'preloc_previous' / f.name))
    deltaz = float(np.load(raw / 'calibr' / 'calibration.npz')['deltaz'])
    name = f'pre_loc_deltaz_{int(deltaz)}.csv'
    preloc_old(raw, deltaz, rec / 'preloc_old' / name)
    frames = frame_list(rec); bg = Background(rec, frames)
    thr = Path(thr) if thr else preloc_sum_suggest(rec, frames, bg)
    preloc_sum_run(rec, frames, bg, thr, rec / 'preloc_sum' / name)
    combine(rec / 'preloc_old' / name, rec / 'preloc_sum' / name, rec / name)
    print('pre-localisation done:', rec / name)
