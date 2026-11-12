# 2026-11-10 (Claude): the SLB bead analysis of analyse_SGD.py ("bead fit", "plot of the bead fit", "rho bias" cells),
# for the 2 beads of one run, with the steps of the circle fit shown.
# Selection (as plot_slb_bias.py): deviance < 1.5, 50 < delta < 150, 50 < N < 10000, 0 < z < 1800 nm, disk of 1.3 um
# around the bead. Circle fit = fit_circle of analyse_SGD.py (closed-form Kasa: least squares of (r^2 - R^2)):
#   0. OUTLIERS (2026-11-10, Amaury: points of the disk that are not on the bead): radial histogram (25 nm bins) around
#      the current centre; bead edge = first bin beyond the peak where the count < 10 % of the peak; points with
#      r > edge + 50 nm removed. Iterated with the fits (3 times), so the centre is never pulled by them.
#   1. fit on the points of the bead (projection = filled disk) -> centre 1; R_bead = 95th percentile of r
#   2. keep the ring 0.7 R_bead < r < edge (equator), fit again -> centre 2, ring and outliers re-selected around it
#   rho_th = azimuth of the point around centre 2 (mod 180); Delta rho = rho - rho_th folded to +-90.
# Localisations RCC drift-corrected and linked (fit_<run>_drift_linked.csv).
# usage: python slb_circle_rho_th.py [run]
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
run = sys.argv[1] if len(sys.argv) > 1 else '2026-11-03_11h46'
D = '/mnt/d/AMAURY/DATA/0_papier/SLB/'
OUT = os.environ.get('OUT_DIR', os.getcwd()).rstrip('/') + '/'  # outputs: $OUT_DIR, else the current folder
os.makedirs(OUT, exist_ok=True)
beads = {'central': (8800, -15400), 'top': (10100, -4300)}
R_FRAC = 0.7
def fit_circle(xx, yy):
    A = np.c_[2*xx, 2*yy, np.ones_like(xx)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, xx**2 + yy**2, rcond=None)
    return np.array([np.sqrt(c + cx**2 + cy**2), cx, cy])
def hsv(v):  # 0-180 deg -> rgb, as in analyse_SGD.py
    h = (np.asarray(v) % 180)/180; return hsv_to_rgb(np.stack((h, np.ones_like(h), np.ones_like(h)), axis=1))
d = pd.read_csv(D + f'fit_{run}_drift_linked.csv', sep=None, engine='python', comment='#')
x, y, z, rho, delta, N, q = (d[k].to_numpy() for k in ('x', 'y', 'z', 'rho', 'delta', 'N_photon', 'deviance'))
mask1 = (q < 1.5) & (delta < 150) & (delta > 50) & (N > 50) & (N < 10000) & (z < 1800) & (z > 0)
th = np.linspace(0, 2*np.pi, 200)
for name, c0 in beads.items():
    disk = mask1 & (np.hypot(x - c0[0], y - c0[1]) < 1300)
    def bead_edge(r):
        h, b = np.histogram(r, bins=np.arange(0, 1500, 25)); ip = np.argmax(h)
        j = ip + np.argmax(h[ip:] < 0.1*h[ip]); return b[j] + 50
    p1 = fit_circle(x[disk], y[disk])
    for it in range(3):  # outliers <-> centre, iterated
        r1 = np.hypot(x - p1[1], y - p1[2]); edge = bead_edge(r1[disk]); mask = disk & (r1 < edge)
        p1 = fit_circle(x[mask], y[mask])
    r1 = np.hypot(x - p1[1], y - p1[2]); R_bead = np.percentile(r1[mask], 95)
    ring1 = mask & (r1 > R_FRAC*R_bead)
    p2 = fit_circle(x[ring1], y[ring1])
    for it in range(3):
        r2 = np.hypot(x - p2[1], y - p2[2]); edge = bead_edge(r2[disk]); mask = disk & (r2 < edge)
        R_bead = np.percentile(r2[mask], 95); ring = mask & (r2 > R_FRAC*R_bead); p2 = fit_circle(x[ring], y[ring])
    r2 = np.hypot(x - p2[1], y - p2[2]); edge = bead_edge(r2[disk]); mask = disk & (r2 < edge)
    R_bead = np.percentile(r2[mask], 95); ring = mask & (r2 > R_FRAC*R_bead)
    out = disk & ~mask
    rho_th = np.degrees(np.arctan2(y[ring] - p2[2], x[ring] - p2[1])) % 180
    drho = (rho[ring] - rho_th + 90) % 180 - 90
    X, Y = (x - p2[1])/1000, (y - p2[2])/1000  # um, centred on the final centre
    cx1, cy1 = (p1[1] - p2[1])/1000, (p1[2] - p2[2])/1000
    Rout = (200 + p2[0])/1000  # reference ring of analyse_SGD.py (R + 200 nm), coloured by its own rho_th
    lineL = np.degrees(np.arctan2(np.sin(th), np.cos(th))) % 180
    fig = plt.figure(figsize=(17, 10)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    # 1. circle fit, step 1
    a = fig.add_subplot(gs[0, 0])
    a.scatter(X[mask], Y[mask], s=2, c='0.5', rasterized=True)
    a.scatter(X[out], Y[out], s=6, c='r', marker='x', rasterized=True, label=f'outliers removed: {out.sum()} (r > {edge:.0f} nm)')
    a.plot(edge/1000*np.cos(th), edge/1000*np.sin(th), 'm-', lw=0.8, label='bead edge + 50 nm (final centre)')
    a.plot(cx1 + p1[0]/1000*np.cos(th), cy1 + p1[0]/1000*np.sin(th), 'b-', lw=1.2, label=f'fit 1 (all): R {p1[0]:.0f} nm')
    a.plot(cx1 + R_bead/1000*np.cos(th), cy1 + R_bead/1000*np.sin(th), 'k:', lw=1, label=f'R_bead = r p95 {R_bead:.0f} nm')
    a.plot(cx1 + R_FRAC*R_bead/1000*np.cos(th), cy1 + R_FRAC*R_bead/1000*np.sin(th), 'r--', lw=1, label=f'{R_FRAC} R_bead: ring cut')
    a.plot(cx1, cy1, 'bx', ms=9)
    a.set_aspect('equal'); a.legend(fontsize=7, loc='lower left'); a.set_title(f'1. outliers removed, fit on the {mask.sum()} bead points', fontsize=10)
    a.set_xlabel('x (um)'); a.set_ylabel('y (um)')
    # 2. step 2
    a = fig.add_subplot(gs[0, 1])
    a.scatter(X[mask & ~ring], Y[mask & ~ring], s=2, c='0.8', rasterized=True, label='removed (inside)')
    a.scatter(X[out], Y[out], s=6, c='r', marker='x', rasterized=True, label='outliers')
    a.scatter(X[ring], Y[ring], s=2, c='0.3', rasterized=True, label=f'ring: {ring.sum()} points')
    a.plot(p2[0]/1000*np.cos(th), p2[0]/1000*np.sin(th), 'g-', lw=1.5, label=f'fit 2 (ring): R {p2[0]:.0f} nm')
    a.plot(R_FRAC*R_bead/1000*np.cos(th), R_FRAC*R_bead/1000*np.sin(th), 'r--', lw=1)
    a.plot(0, 0, 'gx', ms=9); a.plot(cx1, cy1, 'bx', ms=7)
    a.set_aspect('equal'); a.legend(fontsize=7, loc='lower left')
    a.set_title(f'2. refit on the ring; centre moved {np.hypot(cx1, cy1)*1000:.0f} nm (x blue -> green)', fontsize=10)
    a.set_xlabel('x (um)')
    # 3. ring coloured by rho, outer ring coloured by rho_th (as analyse_SGD.py)
    for j, (vals, ttl) in enumerate(((rho[ring], 'ring coloured by fitted rho'), (rho_th, 'ring coloured by rho_th'))):
        a = fig.add_subplot(gs[0, 2 + j]); a.set_facecolor('white')
        a.scatter(X[ring], Y[ring], c=hsv(vals), s=6, rasterized=True)
        a.scatter(Rout*np.cos(th), Rout*np.sin(th), c=hsv(lineL), s=8)
        a.plot(0, 0, 'kx', ms=8); a.axhline(0, color='tab:blue', lw=0.8); a.axvline(0, color='tab:blue', lw=0.8)
        a.set_aspect('equal'); a.set_title(f'3{"ab"[j]}. {ttl}\n(outer ring = rho_th, hsv 0-180)', fontsize=10); a.set_xlabel('x (um)')
    # 4. rho bias: scatter + IQR boxes (analyse_SGD.py "rho bias" cell)
    a = fig.add_subplot(gs[1, 0:2])
    a.scatter(rho_th, drho, s=4, c=hsv(rho_th), rasterized=True)
    a.set_ylabel('Delta rho (deg)'); a.set_ylim(-60, 60); a.set_xlim(0, 180); a.grid()
    a.set_title(f'{name}: Delta rho = rho - rho_th vs rho_th ({ring.sum()} points), colour = rho_th', fontsize=10)
    a.set_xlabel('rho_theory (deg)'); a.set_xticks(np.linspace(0, 180, 7))
    a = fig.add_subplot(gs[1, 2:4])
    rr = np.linspace(0, 180, 37); boxes, pos = [], []
    for i in range(36):
        v = drho[(rho_th > rr[i]) & (rho_th < rr[i+1])]; v = v[np.abs(v) < 40]
        if len(v): boxes.append(v); pos.append(rr[i] + 2.5)
    a.boxplot(boxes, positions=pos, widths=4, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor='white', edgecolor='black', alpha=0.5, linewidth=1.2),
              medianprops=dict(color='black', linewidth=2), whiskerprops=dict(color='none'), capprops=dict(color='none'))
    a.axhline(0., c='r'); a.set_ylim(-10, 10); a.set_xlim(0, 180)
    a.set_xticks(np.linspace(0, 180, 7), ['0', '30', '60', '90', '120', '150', '180']); a.grid()
    a.set_xlabel('rho_theory (deg)'); a.set_ylabel('Delta rho (deg)')
    med = np.median(drho[np.abs(drho) < 40])
    a.set_title(f'IQR per 5 deg (|Delta rho| < 40), overall median {med:+.1f} deg', fontsize=10)
    fig.suptitle(f'SLB {name} bead, run {run} (J of A, sign (-, +), y_gain_J2 0.95, rho_offset -6.2, CNN + LM), '
                 f'RCC drift-corrected + linked; final centre ({p2[1]:.0f}, {p2[2]:.0f}) nm, R {p2[0]:.0f} nm', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT + f'slb_circle_rho_th_{name}_{run}_outliers_removed.png', dpi=100); plt.close(fig)
    phi = np.degrees(np.arctan2(y[ring] - p2[2], x[ring] - p2[1])) % 360; kk = np.abs(drho) < 40
    for lo in range(0, 180, 30):
        sl = kk & (rho_th >= lo) & (rho_th < lo + 30); up, dn = sl & (phi < 180), sl & (phi >= 180)
        print(f'   rho_th {lo:3d}-{lo+30:3d}: upper half {np.median(drho[up]):+5.1f} (n {up.sum():4d}), lower half {np.median(drho[dn]):+5.1f} (n {dn.sum():4d})')
    print('   points per 45 deg of phi:', np.histogram(phi, bins=np.arange(0, 361, 45))[0])
    print(f'{name}: outliers {out.sum()}, edge {edge:.0f} nm; centre1 ({p1[1]:.0f}, {p1[2]:.0f}) R1 {p1[0]:.0f}; R_bead {R_bead:.0f}; centre2 ({p2[1]:.0f}, {p2[2]:.0f}) '
          f'R2 {p2[0]:.0f}; shift {np.hypot(cx1, cy1)*1000:.0f} nm; ring {ring.sum()}/{mask.sum()}; median drho {med:+.2f}')
