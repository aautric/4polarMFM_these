# 2026-11-10 (Claude): whole-stack projection registration (part 1 of
# D:\AMAURY\DATA\2026_11_04_calibration\registration_visual\projections\projection_centres.py) for every calibration with
# a registration_z.npz: centre of each channel from the MAX (MIP) and SUM projections of the bead z-stacks (all frames),
# gaussian fit (fit_bead of reconstruction_juliette.py) and centroid (15 x 15 px, border-median background), vs channel 1.
# Same reconstruction (homographies of the calibration of the registration_z file) and bead selection as
# measure_registration_z. Frame 0 dropped only for 2026-11-04 (beads in focus before the stage moved, as its registration).
# Outputs here: whole_stack_<calib>.png, projection_whole_stack_all.txt, summary_all_calibrations.png, .npz.
import os, sys, numpy as np, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # root of the 4polarMFM_these repo
from reconstruction_juliette import subcanals, extract_raw_ome, measure_baseline, fit_bead
OUT = os.environ.get('OUT_DIR', os.getcwd()).rstrip('/') + '/'  # outputs: $OUT_DIR, else the current folder
os.makedirs(OUT, exist_ok=True)
PX, HALF = 120., 7
CALIBS = {  # name: (registration_z.npz, frames dropped at the start)
    '2026-02-19 SLB': ('/mnt/d/AMAURY/DATA/0_papier/SLB/calibr/registration_z.npz', 0),
    '2026-02-19 cricket': ('/home/amaury/data_local/1_nuclei/cricket/sm1/RAW_DATA/calibr/registration_z.npz', 0),
    '2026-05-29 mouse': ('/home/amaury/data_local/1_nuclei/mouse/zone1_SM1/RAW_DATA/calibr/registration_z.npz', 0),
    '2026-09-18 actin': ('/mnt/d/AMAURY/DATA/0_papier/actin2/calibr/registration_z.npz', 0),
    '2026-11-04 RPE1': ('/mnt/d/AMAURY/DATA/2026_11_04_calibration/registration_z.npz', 1),
}
if len(sys.argv) > 1: CALIBS = {k: v for k, v in CALIBS.items() if any(a in k for a in sys.argv[1:])}

def centroid(img, x0, y0):
    sub = img[y0-HALF:y0+HALF+1, x0-HALF:x0+HALF+1].astype(float)
    bg = np.median(np.concatenate([sub[0], sub[-1], sub[1:-1, 0], sub[1:-1, -1]])); w = np.clip(sub - bg, 0, None)
    yy, xx = np.mgrid[y0-HALF:y0+HALF+1, x0-HALF:x0+HALF+1]
    return np.array([(w*xx).sum()/w.sum(), (w*yy).sum()/w.sum()]) if w.sum() > 0 else np.full(2, np.nan)
def gauss(img, x0, y0):
    p = fit_bead(img, x0, y0); return np.array(p[1:3]) if p is not None else np.full(2, np.nan)
METH = {'gauss': gauss, 'centroid': centroid}
PROJ = {'MIP': lambda s: s.max(0), 'SUM': lambda s: s.sum(0)}
CHS = [0, 3, 4, 2, 5]; COL = {0: 'tab:blue', 3: 'tab:cyan', 4: 'tab:gray', 2: 'tab:red', 5: 'tab:orange'}
q = lambda v: f'{np.nanmedian(v):+6.1f} [{np.nanpercentile(v, 25):+6.1f},{np.nanpercentile(v, 75):+6.1f}]'
lines, summary = [], {}
for name, (regf, drop) in CALIBS.items():
    reg = np.load(regf, allow_pickle=True)
    cal = np.load(str(reg['calibration_file']), allow_pickle=True)
    X, Y, vec = cal['X'], cal['Y'], cal['vecteurs'].tolist()
    M = [cal[f'M_{i}'] for i in range(len([k for k in cal.files if k.startswith('M_')]))]
    C, nfr = [], []
    for bf in reg['bead_files']:
        raw = extract_raw_ome(str(bf)).astype(np.float32)[drop:]; raw -= measure_baseline(raw)
        rec = np.array([[cv2.warpPerspective(ch[i], M[i], ch[i].shape[:2][::-1], flags=cv2.INTER_CUBIC) for i in range(6)]
                        for ch in (subcanals(fr, X, Y, vec) for fr in raw)])
        nfr.append(len(rec))
        mp = rec[:, 1].max(0)
        peaks = (mp == ndimage.maximum_filter(mp, 9)) & (mp > np.percentile(mp, 99.5))
        by, bx = np.where(peaks); keep = (bx > 12) & (bx < mp.shape[1] - 12) & (by > 12) & (by < mp.shape[0] - 12)
        bx, by = bx[keep], by[keep]
        projs = [pf(rec) for pf in PROJ.values()]      # (6, H, W) each
        for i in range(len(bx)):
            if np.min(np.hypot(np.delete(bx, i) - bx[i], np.delete(by, i) - by[i]), initial=99) <= 12: continue
            x0, y0 = int(bx[i]), int(by[i])
            C.append([[[mf(pr[c], x0, y0) for c in range(6)] for mf in METH.values()] for pr in projs])
    C = np.array(C)                        # (bead, proj, meth, ch, xy) pixels
    off = (C - C[:, :, :, 1:2, :])*PX
    roff, rdz = reg['offset'], reg['dz']
    roff = roff*PX if np.nanmax(np.abs(roff)) < 3 else roff
    summary[name] = dict(off=off, reg0=np.array([[np.interp(0, rdz, roff[c, :, j]) for j in range(2)] for c in range(6)]),
                         regm=np.array([[np.nanmean(roff[c, np.abs(rdz) <= 500, j]) for j in range(2)] for c in range(6)]))
    lines.append(f'\n=== {name}: {len(C)} isolated beads, {len(reg["bead_files"])} stacks of {nfr} frames (first {drop} dropped); '
                 f'planes in focus at dz {", ".join(f"{v:+.0f}" for v in reg["focus_dz"])} nm; channel - channel 1, nm, median [p25, p75]')
    for pi, pn in enumerate(PROJ):
        for mi, mn in enumerate(METH):
            lines.append(f'  {pn} {mn}:')
            for c in CHS:
                lines.append(f'    ch {c}: x {q(off[:, pi, mi, c, 0])}  y {q(off[:, pi, mi, c, 1])}')
    lines.append('  registration_z.npz (gaussian per frame): at dz 0 | mean over dz -500..+500')
    for c in CHS:
        r0, rm = summary[name]['reg0'][c], summary[name]['regm'][c]
        lines.append(f'    ch {c}: x {r0[0]:+6.1f} y {r0[1]:+6.1f} | x {rm[0]:+6.1f} y {rm[1]:+6.1f}')
    print('\n'.join(lines[-40:]), flush=True)
    fig, ax = plt.subplots(2, 4, figsize=(17, 8), sharey='row')
    for k, (pi, mi) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        for xy in range(2):
            a = ax[xy, k]; data = [off[:, pi, mi, c, xy] for c in CHS]; data = [d[np.isfinite(d)] for d in data]
            a.boxplot(data, labels=[f'ch {c}' for c in CHS], showfliers=False)
            for j, d in enumerate(data):
                a.plot(np.full(len(d), j + 1) + np.random.default_rng(j).uniform(-.15, .15, len(d)), d, '.', ms=4, color=COL[CHS[j]], alpha=.6)
                a.plot(j + 1, summary[name]['reg0'][CHS[j], xy], 'k_', ms=18, mew=2)
            a.axhline(0, color='0.5', lw=.8); a.set_ylim(-100, 100); a.grid(alpha=.3)
            a.set_title(f'{list(PROJ)[pi]}, {list(METH)[mi]}: {"x (column)" if xy == 0 else "y (row)"}', fontsize=10)
            if k == 0: a.set_ylabel('centre - centre of channel 1 (nm)')
    fig.suptitle(f'Calibration {name}, {len(C)} beads: channel centres from the projections of the whole z-stack, vs channel 1\n'
                 'ch 0, 3 = plane 0; ch 1, 4 = plane 1 (middle); ch 2, 5 = plane 2.  black bar = registration_z.npz (gaussian per frame) at dz = 0', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + f'whole_stack_{name.replace(" ", "_")}.png', dpi=90); plt.close(fig)

# summary: median per calibration, 4 methods (markers) + registration_z at dz 0 (black)
names = list(summary)
fig, ax = plt.subplots(2, len(CHS), figsize=(4*len(CHS), 8), sharey=True)
mk = {(0, 0): ('o', 'MIP gauss'), (0, 1): ('s', 'MIP centroid'), (1, 0): ('^', 'SUM gauss'), (1, 1): ('D', 'SUM centroid')}
for j, c in enumerate(CHS):
    for xy in range(2):
        a = ax[xy, j]
        for (pi, mi), (m, lab) in mk.items():
            v = [np.nanmedian(summary[n]['off'][:, pi, mi, c, xy]) for n in names]
            a.plot(np.arange(len(names)) + 0.08*(2*pi + mi - 1.5), v, m, color=COL[c], label=lab, ms=7, mfc='none' if mi else COL[c])
        a.plot(np.arange(len(names)), [summary[n]['reg0'][c, xy] for n in names], 'k_', ms=16, mew=2, label='registration_z at dz 0')
        a.axhline(0, color='0.5', lw=.8); a.grid(alpha=.3); a.set_xticks(range(len(names)), names, rotation=35, ha='right', fontsize=8)
        a.set_title(f'ch {c}: {"x" if xy == 0 else "y"} vs ch 1 (nm)', fontsize=10)
        if j == 0 and xy == 0: a.legend(fontsize=7)
fig.suptitle('Whole-stack projection registration (median over the beads) for every calibration; black = registration_z.npz at dz = 0', fontsize=11)
fig.tight_layout(); fig.savefig(OUT + 'summary_all_calibrations.png', dpi=90); plt.close(fig)
np.savez(OUT + 'projection_whole_stack_all.npz', **{n.replace(' ', '_'): summary[n]['off'] for n in names})
open(OUT + 'projection_whole_stack_all.txt', 'w').write('Whole-stack projection registration (see the script header)\n' + '\n'.join(lines) + '\n')
