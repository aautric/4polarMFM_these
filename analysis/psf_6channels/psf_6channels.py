# 2026-11-10 (Claude): noiseless 6-channel PSF of the 4polarMFM model (SGD_jax.py through simu_precision/SGD_precision.py) with
# an ideal polarisation calibration (J1 = identity, J2 = rot(45): projections 0 / 90 in planes 0 and 2,
# 45 / 135 in plane 1), next to the dipole and its wobbling cone drawn as in PSF_modulation/animation/anim_dipole_only.py
# (cone of full angle delta, faint dipoles spread over it, dashed projection on the coverslip).
# Runs on the CPU (JAX_PLATFORMS=cpu): small computation, the GPU stays free.
# usage: python psf_6channels.py rho eta delta [z_um] [middle_plane]   (z default: focus of the middle plane)
# 2026-11-12: moved into git (was D:\Amaury\figures_claude\psf_6channels\), imports the repo it is in; J2 kept at
# rot(45): with it a dipole at rho 45 lights the channel labelled 45 deg (with rot(-45): the 135 deg one), see
# review/simulation_precision_fisher_2026-11-12 item 18
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu'); os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
REPO = str(Path(__file__).resolve().parents[2])  # root of the 4polarMFM_these repo
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'simu_precision')); import SGD_precision as P
HERE = Path(os.environ.get('OUT_DIR', os.getcwd()))  # outputs: $OUT_DIR, else the current folder
HERE.mkdir(parents=True, exist_ok=True)

rho, eta, delta = (float(v) for v in sys.argv[1:4])
z_um = float(sys.argv[4]) if len(sys.argv) > 4 else None
middle_plane = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
HALF = 3  # patch of 2*HALF + 1 pixels of 120 nm (7 x 7, centre of the 13 x 13 patch of the fit)

# ---------------------------------------------------------------- model (ideal calibration, no aberration, no noise)
a = P.parse_arguments(['--preset', 'poisson', '--d', str(-middle_plane)])
overrides = {'J1': 'np.eye(2)', 'J2': 'rot(45)', 'rotation': '0', 'rotation2': '0', 'rho_offset': '0',
             'unitary_J': 'False', 'y_gain_J1': '1.', 'y_gain_J2': '1.', 'loss2': 'huber', 'middle_plane': repr(middle_plane)}
tmp = tempfile.mkdtemp()
ns, _, _ = P.build_pipeline(overrides, tmp, 'huber', False, a)
n = int(ns['NPSF'])

def psf(zs):
    # one PSF per z (the model works on whole batches of NPSF)
    zs = np.atleast_1d(zs); out = []
    for s in range(0, len(zs), n):
        zz = np.full(n, zs[min(s, len(zs) - 1)]); zz[:len(zs[s:s + n])] = zs[s:s + n]
        p = dict(rho=np.full(n, rho), eta=np.full(n, eta), delta=np.full(n, delta), N=np.full(n, 1e4),
                 x=np.zeros(n), y=np.zeros(n), z=zz)
        out.append(P.model_psf(ns, slice(0, n), p, half=HALF)[:len(zs[s:s + n])])
    return np.concatenate(out)

if z_um is None:  # focus of the middle plane: highest peak of plane 1 (sum of its 2 polarisations)
    zs = np.linspace(0.05, 2.0, 40)
    peak = psf(zs)[:, 1].sum(1).max(axis=(-2, -1))
    z_um = float(zs[np.argmax(peak)])
h = psf([z_um])[0]  # (3 planes, 2 polarisations, side, side)

# ---------------------------------------------------------------- figure
proj = np.array(ns['polar_projections'], float)
fig = plt.figure(figsize=(6.6, 2.75), facecolor='white')
# dipole and wobbling cone (as anim_dipole_only.py)
ink, col_dip, col_arrow = '#1f2328', '#2a6fdb', '#173f8a'
ax3d = fig.add_axes((-0.03, 0.0, 0.36, 0.92), projection='3d'); ax3d.computed_zorder = False
rh, et = np.deg2rad(rho), np.deg2rad(eta)
mu = np.array([np.sin(et)*np.cos(rh), np.sin(et)*np.sin(rh), np.cos(et)])
e1 = np.cross(mu, [0, 0, 1] if abs(mu[2]) < 0.9 else [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(mu, e1)
half_angle = np.deg2rad(delta)/2; m = 400; u = (np.arange(m) + 0.5)/m
cth = 1 - u*(1 - np.cos(half_angle)); sth = np.sqrt(1 - cth**2); ph = np.arange(m)*np.pi*(3 - np.sqrt(5))
ghosts = cth[:, None]*mu + sth[:, None]*(np.cos(ph)[:, None]*e1 + np.sin(ph)[:, None]*e2)
ss = np.linspace(-1, 1, 2); X, Y = np.meshgrid(ss, ss)
ax3d.plot_surface(X, Y, np.full_like(X, -1), color='#9ab', alpha=0.2, linewidth=0, zorder=0)  # coverslip
for v, lab in zip(np.eye(3), ('x', 'y', 'z')):
    # axes with an arrow head on the positive side; the z label a bit lower than the tip (title above it)
    ax3d.quiver(*(-1.1*v), *(2.3*v), color='#999', lw=0.8, arrow_length_ratio=0.06, zorder=1)
    ax3d.text(*(v*(1.08 if lab == 'z' else 1.3) + (np.array([0.12, 0, 0]) if lab == 'z' else 0)), lab, color='#666', fontsize=9)
ax3d.add_collection3d(Line3DCollection(np.stack([-ghosts, ghosts], axis=1), colors=col_dip,
                                       alpha=0.025*min(1, delta/30), linewidths=5, zorder=2))
ax3d.quiver(0, 0, 0, *mu, color=col_arrow, lw=2.2, arrow_length_ratio=0.2, zorder=5)
ax3d.quiver(0, 0, 0, *(-mu), color=col_arrow, lw=2.2, arrow_length_ratio=0.2, zorder=5)
ax3d.plot([0, mu[0]], [0, mu[1]], [-1, -1], ls='--', color=col_dip, lw=1, zorder=1)
ax3d.scatter(0, 0, 0, color=ink, s=10, zorder=6)
ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1); ax3d.set_box_aspect((1, 1, 1))
ax3d.view_init(elev=20, azim=-60); ax3d.set_axis_off()
fig.text(0.16, 0.87, f'ρ = {rho:.0f}°, η = {eta:.0f}°, δ = {delta:.0f}°', ha='center', fontsize=10, color=ink)
# 6 channels: columns = planes (defocus), rows = the 2 polarisations of each plane, one colour scale for all
vmax = h.max(); w, gap, x0, y0, hh = 0.185, 0.01, 0.33, 0.04, 0.41
# analyser of each channel: J = rot(theta) of the plane (theta = polar_projections), channel 0 = x output of rot(theta)
# = direction -theta, channel 1 = 90 - theta (plane 2 of the model: 135 / 45). Top row 0 / 45 / 0, bottom 90 / 135 / 90
for p in range(3):
    for q in range(2):
        angle = (-proj[p] + 90*q) % 180
        row = 0 if angle < 90 else 1
        ax = fig.add_axes((x0 + p*(w + gap), y0 + (1 - row)*(hh + gap*2), w, hh))
        im = ax.imshow(h[p, q], cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.04, 0.96, f'{angle:.0f}°', transform=ax.transAxes, color='white', fontsize=8, va='top')
        if row == 0:
            ax.set_title(f'plane {p + 1}', fontsize=9, pad=3)
cax = fig.add_axes((x0 + 3*(w + gap) - gap + 0.006, y0, 0.012, 2*hh + 2*gap))
cb = fig.colorbar(im, cax=cax); cb.set_ticks([]); cb.set_label('Intensity', fontsize=8)
out = HERE / f'psf6_rho{rho:g}_eta{eta:g}_delta{delta:g}.png'
fig.savefig(out, dpi=200)
# parameters of the figure in a text file next to it (not written on the figure)
sums = h.sum((-2, -1))
out.with_suffix('.txt').write_text(
    f'{out.name}\n'
    f'dipole: rho {rho:g} deg, eta {eta:g} deg, delta {delta:g} deg (full angle of the wobbling cone)\n'
    f'emitter z = {1000*z_um:.0f} nm above the coverslip' + (' (focus of the middle plane)' if len(sys.argv) <= 4 else '') + '\n'
    f'planes at {middle_plane - 0.354:.3f} / {middle_plane:.3f} / {middle_plane + 0.354:.3f} um (middle_plane {middle_plane:g}, interplane 0.354)\n'
    f'channels: {2*HALF + 1} x {2*HALF + 1} pixels of 120 nm, centre of the patch; grayscale, one common scale; no noise\n'
    f'ideal projections (J1 = identity, J2 = rot(45)), no aberration, N = 1e4 photons (scale irrelevant)\n'
    f'photons in each displayed {2*HALF + 1} x {2*HALF + 1} crop (of N = 1e4 for the whole molecule):\n'
    + '\n'.join(f'  plane {p + 1}: ' + '  '.join(f'{(-proj[p] + 90*q) % 180:3.0f} deg {sums[p, q]:7.0f}' for q in range(2)) for p in range(3)) + '\n')
print('saved', out, 'and', out.with_suffix('.txt'))
