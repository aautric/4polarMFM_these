# 2026-11-10 (Claude): SLB analysis REDONE with the corrected bead circle fit (outliers outside the bead removed, Amaury).
# Replaces plot_slb_bias.py, slb_rho_bias.py (run_summary), eta_boundary.py, ellipse_check.py of the parent folder, whose
# circle fit kept the scattered points outside the bead (r up to 1.3 um) in the ring: centre of the central bead ~110 nm off.
#
# Selection (analyse_SGD.py): deviance < 1.5, 50 < delta < 150, 50 < N < 10000, 0 < z < 1800 nm, disk of 1.3 um.
# Bead fit (slb_circle_rho_th.py): radial histogram (25 nm bins) around the current centre, bead edge = first bin beyond
# the peak where the count < 10 % of the peak, points with r > edge + 50 nm = outliers; fit_circle (Kasa) on the bead
# points -> centre; R_bead = r p95; ring 0.7 R_bead < r (< edge + 50); refit on the ring; all iterated 3 times.
# rho_th = azimuth around the final centre (mod 180), drho = rho - rho_th folded to +-90, |drho| < 40 kept.
# Outputs (this folder): table_drift_linked.txt, table_raw.txt, harmonics_all_runs.png, beads_<run>.png,
# eta_boundary.txt, ellipse_check.txt.
# usage: python slb_analysis_v2.py
import glob, os, re, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/mnt/d/AMAURY/DATA/0_papier/SLB/'
OUT = os.environ.get('OUT_DIR', os.getcwd()).rstrip('/') + '/'  # outputs: $OUT_DIR, else the current folder
os.makedirs(OUT, exist_ok=True)
BEADS = {'central': (8800, -15400), 'top': (10100, -4300)}
R_FRAC = 0.7

def fit_circle(xx, yy):
    A = np.c_[2*xx, 2*yy, np.ones_like(xx)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, xx**2 + yy**2, rcond=None)
    return np.array([np.sqrt(c + cx**2 + cy**2), cx, cy])

def bead_edge(r):
    h, b = np.histogram(r, bins=np.arange(0, 1500, 25)); ip = np.argmax(h)
    return b[ip + np.argmax(h[ip:] < 0.1*h[ip])] + 50

def bead_fit(x, y, mask1, c0):
    """returns p = (R, cx, cy) of the ring fit, bead mask (outliers removed), ring mask, outlier mask, R_bead, edge"""
    disk = mask1 & (np.hypot(x - c0[0], y - c0[1]) < 1300)
    p = fit_circle(x[disk], y[disk])
    for _ in range(3):
        r = np.hypot(x - p[1], y - p[2]); edge = bead_edge(r[disk]); bead = disk & (r < edge); p = fit_circle(x[bead], y[bead])
    for _ in range(4):
        r = np.hypot(x - p[1], y - p[2]); edge = bead_edge(r[disk]); bead = disk & (r < edge)
        R_bead = np.percentile(r[bead], 95); ring = bead & (r > R_FRAC*R_bead); p = fit_circle(x[ring], y[ring])
    r = np.hypot(x - p[1], y - p[2]); edge = bead_edge(r[disk]); bead = disk & (r < edge)
    R_bead = np.percentile(r[bead], 95); ring = bead & (r > R_FRAC*R_bead)
    return p, bead, ring, disk & ~bead, R_bead, edge

def harm(t, v, nb=200):
    T = np.radians(t); A = np.c_[np.ones_like(T), np.cos(2*T), np.sin(2*T), np.cos(4*T), np.sin(4*T)]
    c = np.linalg.lstsq(A, v, rcond=None)[0]; rg = np.random.default_rng(0)
    e = np.std([np.linalg.lstsq(A[i], v[i], rcond=None)[0] for i in (rg.integers(0, len(T), len(T)) for _ in range(nb))], 0)
    return c, e

def curve(c, t):
    T = np.radians(t); return c[0] + c[1]*np.cos(2*T) + c[2]*np.sin(2*T) + c[3]*np.cos(4*T) + c[4]*np.sin(4*T)

def load(run, kind):
    f = D + (f'fit_{run}_drift_linked.csv' if kind == 'drift_linked' else f'fit_{run}.csv')
    d = pd.read_csv(f, sep=';', comment='#')
    cols = [d[k].to_numpy() for k in ('x', 'y', 'z', 'rho', 'eta', 'delta', 'N_photon', 'deviance')]
    x, y, z, rho, eta, delta, N, q = cols
    mask1 = (q < 1.5) & (delta < 150) & (delta > 50) & (N > 50) & (N < 10000) & (z < 1800) & (z > 0)
    return x, y, z, rho, eta, delta, mask1

# run descriptions from runs_summary.txt (rotation line + "tested")
desc = {}
txt = open(D + 'runs_summary.txt').read()
for blk in txt.split('\nNPZ_')[1:]:
    run = blk.split()[0]; rot = re.search(r'rotation .*', blk); t = re.search(r'tested: (.*)', blk)
    desc[run] = (rot.group(0) if rot else '') + ' | ' + (t.group(1)[:150] if t else '')
runs = sorted(os.path.basename(f)[4:-17] for f in glob.glob(D + 'fit_*_drift_linked.csv'))

# ---------------- 1. harmonics table, all runs, drift-corrected + linked and raw ----------------
res = {}
for kind in ('drift_linked', 'raw'):
    lines = [f'# SLB rho bias / delta vs rho_th, corrected bead fit (outliers removed), {kind}; harmonics c0 c2 s2 c4 s4 '
             f'(bootstrap sd); rms(2-4) = sqrt(sum c^2/2) of the periodic part; halves = rms of the median drho difference '
             f'between phi < 180 and phi > 180 at the same rho_th (30 deg bins): 0 if the centre is right and the bias depends only on rho']
    for run in runs:
        x, y, z, rho, eta, delta, mask1 = load(run, kind)
        for name, c0 in BEADS.items():
            p, bead, ring, out, R_bead, edge = bead_fit(x, y, mask1, c0)
            phi = np.degrees(np.arctan2(y[ring] - p[2], x[ring] - p[1])) % 360; rth = phi % 180
            dr = (rho[ring] - rth + 90) % 180 - 90; k = np.abs(dr) < 40
            cr, er = harm(rth[k], dr[k]); cd, ed = harm(rth, delta[ring])
            rms = np.sqrt(np.sum(cr[1:]**2)/2)
            hd = []
            for lo in range(0, 180, 30):
                s = k & (rth >= lo) & (rth < lo + 30); u, dn = s & (phi < 180), s & (phi >= 180)
                if u.sum() > 20 and dn.sum() > 20: hd.append(np.median(dr[u]) - np.median(dr[dn]))
            hrms = np.sqrt(np.mean(np.square(hd)))
            res[(kind, run, name)] = dict(c=cr, e=er, rms=rms, d=cd, ed=ed, halves=hrms, n=k.sum(), out=out.sum(), R=p[0], cx=p[1], cy=p[2])
            f2 = lambda c, e: ' '.join(f'{a:+5.1f}({b:.1f})' for a, b in zip(c, e))
            lines.append(f'{name:8s}{run:18s}n {k.sum():5d} outliers {out.sum():4d} R {p[0]:4.0f} centre ({p[1]:6.0f},{p[2]:7.0f}) | '
                         f'drho c0 c2 s2 c4 s4: {f2(cr, er)} | rms(2-4) {rms:4.2f} | halves {hrms:4.1f} | delta: {f2(cd, ed)}')
            print(lines[-1], flush=True)
    open(OUT + f'table_{kind}.txt', 'w').write('\n'.join(lines) + '\n\nruns:\n' + '\n'.join(f'{r}: {desc.get(r, "?")}' for r in runs) + '\n')

# summary plot: c0, rms(2-4), c2, s2 per run, both beads, drift vs raw
fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
xi = np.arange(len(runs))
for j, (lab, fn) in enumerate((('c0 = mean rho bias (deg)', lambda r: r['c'][0]), ('rms of the periodic part (deg)', lambda r: r['rms']),
                               ('c2: cos 2rho_th term (deg)', lambda r: r['c'][1]), ('s2: sin 2rho_th term (deg)', lambda r: r['c'][2]))):
    for name, col in (('central', 'tab:blue'), ('top', 'tab:red')):
        for kind, mk, ls in (('drift_linked', 'o', '-'), ('raw', 'x', ':')):
            v = [fn(res[(kind, r, name)]) for r in runs]
            ax[j].plot(xi, v, marker=mk, ls=ls, color=col, label=f'{name}, {kind}' if j == 0 else None)
    ax[j].axhline(0, color='0.5', lw=0.8); ax[j].set_ylabel(lab, fontsize=9); ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=8, ncol=4)
ax[-1].set_xticks(xi, runs, rotation=60, fontsize=8)
fig.suptitle('SLB rho bias harmonics per run, corrected bead fit (outliers removed); runs: see table_drift_linked.txt', fontsize=11)
fig.tight_layout(); fig.savefig(OUT + 'harmonics_all_runs.png', dpi=100); plt.close(fig)

# ---------------- 2. per-run bead figure (best run on both beads + 11h46) ----------------
score = {r: np.hypot(res[('drift_linked', r, 'central')]['rms'], res[('drift_linked', r, 'top')]['rms']) for r in runs}
best = min(score, key=score.get)
print(f'best run (sqrt(rms_central^2 + rms_top^2), drift-corrected): {best} {score[best]:.2f}')
for run in sorted({best, '2026-11-03_11h46'}):
    x, y, z, rho, eta, delta, mask1 = load(run, 'drift_linked')
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.5), gridspec_kw=dict(width_ratios=[1, 1, 1.5, 1.5]))
    tt = np.linspace(0, 180, 361); bins = np.arange(0, 181, 15); th = np.linspace(0, 2*np.pi, 200)
    for i, (name, c0) in enumerate(BEADS.items()):
        p, bead, ring, out, R_bead, edge = bead_fit(x, y, mask1, c0)
        X, Y = (x - p[1])/1000, (y - p[2])/1000
        phi = np.degrees(np.arctan2(y[ring] - p[2], x[ring] - p[1])) % 360; rth = phi % 180
        dr = (rho[ring] - rth + 90) % 180 - 90; k = np.abs(dr) < 40
        r_ = res[('drift_linked', run, name)]
        a = ax[i, 0]
        a.scatter(X[bead & ~ring], Y[bead & ~ring], s=2, c='0.8', rasterized=True)
        a.scatter(X[ring], Y[ring], s=2, c='0.3', rasterized=True)
        a.scatter(X[out], Y[out], s=8, c='r', marker='x', label=f'{out.sum()} outliers')
        a.plot(p[0]/1000*np.cos(th), p[0]/1000*np.sin(th), 'g-', lw=1.2, label=f'ring fit R {p[0]:.0f} nm')
        a.plot(edge/1000*np.cos(th), edge/1000*np.sin(th), 'm-', lw=0.8, label='edge + 50 nm')
        a.set_aspect('equal'); a.legend(fontsize=7, loc='lower left'); a.set_title(f'{name}: bead fit', fontsize=10)
        a = ax[i, 1]
        a.scatter(X[ring], Y[ring], c=rho[ring] % 180, cmap='hsv', vmin=0, vmax=180, s=3, rasterized=True)
        a.scatter((p[0]/1000 + 0.2)*np.cos(th), (p[0]/1000 + 0.2)*np.sin(th), c=np.degrees(th) % 180, cmap='hsv', vmin=0, vmax=180, s=6)
        a.set_aspect('equal'); a.set_title('ring by rho (outer: rho_th)', fontsize=10)
        a = ax[i, 2]
        up = phi < 180
        a.plot(rth[k & up], dr[k & up], '.', ms=2, color='tab:orange', alpha=0.4, label='phi < 180 (upper half of the image ring)')
        a.plot(rth[k & ~up], dr[k & ~up], '.', ms=2, color='tab:cyan', alpha=0.4, label='phi > 180')
        ii = np.digitize(rth[k], bins) - 1
        med = [np.median(dr[k][ii == j]) if np.sum(ii == j) > 10 else np.nan for j in range(len(bins) - 1)]
        a.plot((bins[:-1] + bins[1:])/2, med, 'ko', label='median per 15 deg')
        a.plot(tt, curve(r_['c'], tt), 'r-', lw=2, label=f"c0 {r_['c'][0]:+.1f}, c2 {r_['c'][1]:+.1f}, s2 {r_['c'][2]:+.1f}, rms {r_['rms']:.2f}")
        a.axhline(0, color='0.3', lw=0.8); a.set_ylim(-20, 20); a.set_xlim(0, 180); a.legend(fontsize=7, loc='upper left')
        a.set_xlabel('rho_th (deg)'); a.set_ylabel('rho - rho_th (deg)'); a.set_title(f'{name}: rho bias ({k.sum()} points)', fontsize=10)
        a = ax[i, 3]
        a.plot(rth, delta[ring], '.', ms=2, color='0.65', alpha=0.5)
        ii = np.digitize(rth, bins) - 1
        med = [np.median(delta[ring][ii == j]) if np.sum(ii == j) > 10 else np.nan for j in range(len(bins) - 1)]
        a.plot((bins[:-1] + bins[1:])/2, med, 'ko'); a.plot(tt, curve(r_['d'], tt), '-', color='tab:blue', lw=2, label=f"mean {r_['d'][0]:.1f}")
        a.set_ylim(50, 150); a.set_xlim(0, 180); a.set_xlabel('rho_th (deg)'); a.set_ylabel('delta (deg)'); a.legend(fontsize=8)
        a.set_title(f'{name}: delta on the ring', fontsize=10)
    fig.suptitle(f'SLB run {run}, drift-corrected + linked, corrected bead fit\n{desc.get(run, "")}', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + f'beads_{run}.png', dpi=100); plt.close(fig)

# ---------------- 3. eta boundary vs rho = 0 line (Wollaston / analyser frame vs image) ----------------
def boundary(phi, up):
    grid = np.arange(-45, 45.01, 0.5)
    S = np.sign(np.sin(np.radians(phi[None, :] - grid[:, None])))
    agree = (S == up[None, :]).sum(1) / len(up)
    s = np.argmax(agree if agree.max() >= 0.5 else 1 - agree)
    return grid[s], max(agree[s], 1 - agree[s])
rng = np.random.default_rng(0); lines = ['# eta sign boundary (tilted |eta-90| > 15, inside the ring fit radius, upper/lower half at the median z) '
                                         'vs the line rho = 0 (phi = -c0); offset = phi0 - (-c0), deg']
offs = {n: [] for n in BEADS}
for run in runs:
    x, y, z, rho, eta, delta, mask1 = load(run, 'drift_linked')
    for name, c0 in BEADS.items():
        p, bead, ring, out, R_bead, edge = bead_fit(x, y, mask1, c0)
        r = np.hypot(x - p[1], y - p[2]); phi = np.degrees(np.arctan2(y - p[2], x - p[1]))
        inside = bead & (r < p[0]) & (np.abs(eta - 90) > 15); zc = np.median(z[bead & (r < p[0])])
        rho0 = -res[('drift_linked', run, name)]['c'][0]; out_ = []
        for half, hm in (('upper', inside & (z > zc)), ('lower', inside & (z <= zc))):
            up = np.where(eta[hm] < 90, 1, -1); g, a = boundary(phi[hm], up)
            bs = [boundary(phi[hm][i], up[i])[0] for i in (rng.integers(0, hm.sum(), hm.sum()) for _ in range(100))]
            out_.append(f'{half} phi0 {g:+5.1f} +- {np.std(bs):.1f} (agr {100*a:.0f} %, n {hm.sum()}) offset {g - rho0:+5.1f}')
            offs[name].append((g - rho0, np.std(bs)))
        lines.append(f'{run} {name:8s} rho = 0 at phi {rho0:+.1f} | ' + ' | '.join(out_)); print(lines[-1], flush=True)
for name in BEADS:
    v, e = np.array(offs[name]).T; w = 1/e**2
    lines.append(f'{name}: weighted mean offset {np.sum(w*v)/np.sum(w):+.1f} +- {1/np.sqrt(np.sum(w)):.1f} deg (all runs, both halves; '
                 f'runs are refits of the same data, NOT independent), median {np.median(v):+.1f}, spread sd {np.std(v):.1f}')
    print(lines[-1])
open(OUT + 'eta_boundary.txt', 'w').write('\n'.join(lines) + '\n')

# ---------------- 4. ellipse check (is the 2rho term a shape effect?) ----------------
lines = ['# ellipse fitted to the ring (corrected fit); rho_th = circle azimuth vs ellipse normal']
for run in runs:
    x, y, z, rho, eta, delta, mask1 = load(run, 'drift_linked')
    for name, c0 in BEADS.items():
        p, bead, ring, out, R_bead, edge = bead_fit(x, y, mask1, c0)
        X, Y = (x[ring] - p[1])/1000, (y[ring] - p[2])/1000
        a_, b_, c_, d_, e_ = np.linalg.lstsq(np.c_[X**2, X*Y, Y**2, X, Y], np.ones_like(X), rcond=None)[0]
        ev, evec = np.linalg.eigh(np.array([[a_, b_/2], [b_/2, c_]])); ratio = np.sqrt(ev.max()/ev.min())
        ang = np.degrees(np.arctan2(evec[1, 0], evec[0, 0])) % 180
        gx, gy = 2*a_*X + b_*Y + d_, b_*X + 2*c_*Y + e_; s = []
        for lab, rt in (('circle', np.degrees(np.arctan2(Y, X)) % 180), ('ellipse', np.degrees(np.arctan2(gy, gx)) % 180)):
            dr = (rho[ring] - rt + 90) % 180 - 90; k = np.abs(dr) < 40; h = harm(rt[k], dr[k], nb=1)[0]
            s.append(f'{lab}: c2 {h[1]:+.1f} s2 {h[2]:+.1f} rms {np.sqrt(np.sum(h[1:]**2)/2):.2f}')
        lines.append(f'{run} {name:8s} axis ratio {ratio:.3f} long axis {ang:5.1f} | ' + ' | '.join(s))
open(OUT + 'ellipse_check.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines[-6:]))
