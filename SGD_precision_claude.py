# -*- coding: utf-8 -*-
"""
Precision and bias of the SGD pipeline of SGD_jax.py across rho and eta, on simulated data.

The pipeline is the one of SGD_jax.py, executed cell by cell (as SGD_batch.py does): SGD1, SGD2, losses, noise model,
score, saving of the npz files. Only the data are simulated: the reading of the frames in load_batch is replaced by
simulated PSF (the end of load_batch, which builds the batch for the descent, is taken from SGD_jax.py itself), and
the noise model of each batch is measured by noise_model of SGD_jax.py on simulated background frames.

Simulation (in photons):
    PSF of the model of SGD_jax.py, ideal polarisation calibration (J1 = identity, J2 = 45 deg rotation), delta = 90
    background b + sigma_b * N(0,1) per pixel and frame (fluctuation of the background beyond the Poisson noise),
    camera noise of simu_PSF_polarMFM_JAX.py: emccd_jax (photoelectrons Poisson(QE x light), EM register Gamma(e, EM),
    read noise sigma_r, ADC: variance 2 x light/QE + sigma_r^2) or, with --camera poisson, noise_jax (Poisson and read
    noise only). No camera offset: the reconstruction removes the baseline of the raw frames, so the data are 0
    without light
    crazy pixels (optional): a fraction of the pixels of each PSF get extra photons (exponential law), to mimic the
    irregular background of dense samples
    second emitter (optional): a fraction of the PSF get a neighbour (random orientation, N and z) a few pixels away;
    the truth is the central emitter
    reconstruction warp (optional, --warp calibration.npz): each channel is interpolated as by the cubic warp of
    reconstruction_juliette.py (noise added before, as on the camera); plane 1 x, the reference, is not interpolated
    model mismatch of the data (optional, the fit keeps the model of SGD_jax.py):
        --data-n2: refractive index of the sample of the data (critical angle, Fresnel coefficients, defocus phase)
        --data-saf-dz: the evanescent (SAF) part of the field of the data decays as for an emitter dz further from the
            coverslip (dz < 0: closer, more SAF), the propagating part unchanged: too much / too little SAF only
        --data-shift plane pol dx dy: the channel (plane 0-2, pol 0 x / 1 y / 2 both) of the data shifted laterally by
            dx, dy nm, as a registration error of the reconstruction (repeatable)
        --data-zernike file.npz: aberrations of the data (zx, zy: (3 planes, 15 Noll modes) in rad, for the x and y
            channels), the fit keeps its own (none, or fitted in SGD2 with --set fit_zernike=True)
        --data-zshift table.npz: each channel of the data shifted laterally by an amount depending on the z of the
            emitter (z_em (nz,) um, shift (3, 2, nz, 2) nm in the coordinates of the fit, planes and pol of SGD_jax.py),
            e.g. the residual registration measured on bead z-stacks after the reconstruction
Orientations: --rho-values / --eta-values (lists), or a regular grid n_rho x n_eta, rho and eta at the centres of bins of [0, 180[ (20 x 20: 4.5, 13.5, ...,
175.5 deg), per_point PSF each (80 by default; the batches of the SGD, NPSF = 100, are filled with consecutive PSF), random x, y, z and noise.
The losses given (poisson, lms, huber) are run on exactly the same simulated data, each in a fresh pipeline.

Presets (--preset; any option given explicitly replaces the value of the preset). Values measured on the real runs
(median N, background above the offset, offset, in photons). N = photons of the whole molecule (3 planes): the
values measured before 2026-10-28 were per plane (PSF_jax normalised each plane) and are multiplied by 3 here:
    poisson      Poisson noise only, N and background of actin (no EM excess, no read noise, no offset, no fluctuation)
    slb          SLB 02-02: N 1170 (390 per plane), background 3.4, offset 3.5
    actin        RPE1 actin 09-18 / 09-23: N 3570 / 1965 (1190 / 655 per plane), background 20 / 35, offset 3.8 / 4.3
                 -> N 3000, background 25
    actin_dense  actin + 3 % crazy pixels + a second emitter on 20 % of the PSF

usage (python of the environment, e.g. /home/amaury/exit/envs/env_linux/bin/python):
    python SGD_precision_claude.py [options] [--set name=value ...]
    e.g. python SGD_precision_claude.py --preset actin_dense --losses poisson lms huber --out /mnt/d/Amaury/simu_claude
options: see python SGD_precision_claude.py --help
    python SGD_precision_claude.py --recompute <result folder> [...]: statistics, figures and summary of finished runs
    again from their simulation.npz and results_<loss>.npz (no fit)
--set name=value changes a parameter of SGD_jax.py (as in SGD_batch.py), e.g. --set huber_c=1.5 num_epochs_max2=150
--cnn runs the pipeline of SGD_jax_CNN.py (CNN start and per-PSF fit) instead of SGD_jax.py (its --set parameters)

outputs in <out>/<date>_<tag>/:
    simulation.npz          simulated data and true parameters
    NPZ_<loss>/             npz files of the pipeline (as a real run) and its config.txt
    results_<loss>.npz      true and found parameters of each PSF, errors
    summary.csv             bias (median of the error) and precision (robust std) per loss and orientation
    figures                 polar maps of bias and precision of rho, eta, delta; profiles vs eta; comparison of
                            the losses
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import argparse
import textwrap
import time
from pathlib import Path
from datetime import datetime
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import SGD_batch # helpers to read and change SGD_jax.py and run its cells (not modified)

SGD_SCRIPT = HERE / 'SGD_jax.py'
log = SGD_batch.log


# ---------------------------------------------------------------- parameters

PRESETS = {
    'poisson':     dict(camera='poisson', N=3000., background=25., sigma_b=0., sigma_r=0.,
                        crazy_fraction=0., two_emitter_fraction=0.),
    'slb':         dict(camera='emccd', N=1170., background=3.4, sigma_b=0.5, sigma_r=0.4,
                        crazy_fraction=0., two_emitter_fraction=0.),
    'actin':       dict(camera='emccd', N=3000., background=25., sigma_b=3., sigma_r=0.4,
                        crazy_fraction=0., two_emitter_fraction=0.),
    'actin_dense': dict(camera='emccd', N=3000., background=25., sigma_b=3., sigma_r=0.4,
                        crazy_fraction=0.03, two_emitter_fraction=0.2),
}


def parse_arguments(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default='actin', choices=list(PRESETS), help='noise and photon conditions (see above)')
    p.add_argument('--losses', nargs='+', default=['poisson', 'lms', 'huber'], choices=['poisson', 'lms', 'huber'])
    p.add_argument('--n-rho', type=int, default=20, help='number of rho bins in [0, 180[')
    p.add_argument('--n-eta', type=int, default=20, help='number of eta bins in [0, 180[ (even: the bins split at 90)')
    p.add_argument('--per-point', type=int, default=80, help='number of PSF per orientation')
    p.add_argument('--delta', type=float, default=90., help='true wobbling cone (deg)')
    # None: value of the preset
    p.add_argument('--camera', default=None, choices=['emccd', 'poisson'], help='emccd: Poisson + EM excess + read noise + offset; poisson: Poisson noise only')
    p.add_argument('--N', type=float, default=None, help='photons of each PSF (whole molecule, 3 planes)')
    p.add_argument('--background', type=float, default=None, help='background b (photons/pixel)')
    p.add_argument('--sigma-b', type=float, default=None, help='fluctuation of the background per pixel and frame (photons)')
    p.add_argument('--sigma-r', type=float, default=None, help='read noise (photons)')
    p.add_argument('--QE', type=float, default=0.92)
    p.add_argument('--cic', type=float, default=0., help='clock induced charge (electrons per pixel and frame, emccd camera)')
    p.add_argument('--EM', type=float, default=200.)
    p.add_argument('--interplane', type=float, default=0.354, help='distance between the planes (um)')
    p.add_argument('--d', type=float, default=-1.0, help='d_ of compute_M_jax (um), d_ = -middle_plane in SGD_jax.py')
    p.add_argument('--z-range', type=float, nargs=2, default=[0.2, 1.2], help='range of the true z (um)')
    p.add_argument('--xy-range', type=float, default=0.2, help='true x, y uniform in [-xy_range, xy_range] (um)')
    p.add_argument('--crazy-fraction', type=float, default=None, help='fraction of crazy pixels in each PSF (0: off)')
    p.add_argument('--crazy-amplitude', type=float, default=100., help='mean extra photons of a crazy pixel')
    p.add_argument('--two-emitter-fraction', type=float, default=None, help='fraction of the PSF with a second emitter (0: off)')
    p.add_argument('--neighbour-distance', type=float, nargs=2, default=[3., 5.], help='distance of the second emitter (pixels of 120 nm)')
    p.add_argument('--neighbour-N', type=float, nargs=2, default=[0.5, 1.5], help='photons of the second emitter, relative to N (uniform)')
    p.add_argument('--warp', default=None, help='calibration.npz of reconstruction_juliette.py: the simulated raw '
                   'channels go through its cubic warps (blur and noise smoothing of the reconstruction), see warp_maps')
    p.add_argument('--data-y-gain-J2', type=float, default=1., help='y channel of plane 1 of the simulated data x this '
                   'factor (real x/y intensity error, the fit keeps its own y_gain_J2)')
    p.add_argument('--calibration', default=None, help='config.txt of a run: its J1, J2, rotation, rotation2, unitary_J, '
                   'y_gain_J1/J2 for the data and the fit (rho_offset stays 0: rho of the model) instead of the ideal calibration')
    p.add_argument('--rho-values', type=float, nargs='+', default=None, help='list of true rho (replaces the grid of n_rho)')
    p.add_argument('--eta-values', type=float, nargs='+', default=None, help='list of true eta (replaces the grid of n_eta)')
    p.add_argument('--data-n2', type=float, default=None, help='refractive index of the sample in the data (model: n2 of simu_PSF_polarMFM_JAX.py)')
    p.add_argument('--data-saf-dz', type=float, default=0., help='SAF of the data as for an emitter dz um further from the coverslip')
    p.add_argument('--data-shift', type=float, nargs=4, action='append', default=[], metavar=('PLANE', 'POL', 'DX', 'DY'),
                   help='lateral shift (nm) of a channel of the data: plane 0-2, pol 0 (x), 1 (y) or 2 (both)')
    p.add_argument('--data-zernike', default=None, help='npz (zx, zy (3, 15) rad): aberrations of the simulated data only')
    p.add_argument('--data-zshift', default=None, help='npz table of a lateral shift of each channel vs the z of the emitter')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='/mnt/d/Amaury/simu_precision_claude', help='folder of the results')
    p.add_argument('--tag', default='', help='added to the name of the result folder (default: the preset)')
    p.add_argument('--save-figures', action='store_true', help='also save the figures of the pipeline (first batches)')
    p.add_argument('--set', nargs='*', default=[], help='name=value parameters of SGD_jax.py')
    p.add_argument('--cnn', action='store_true', help='pipeline of SGD_jax_CNN.py (CNN start and per-PSF fit) instead of SGD_jax.py')
    a = p.parse_args(argv)
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if not a.tag:
        a.tag = a.preset + ('_warp' if a.warp else '') + (f'_ygain{a.data_y_gain_J2:g}' if a.data_y_gain_J2 != 1 else '')
        a.tag += (f'_n2{a.data_n2:g}' if a.data_n2 else '') + (f'_safdz{a.data_saf_dz:g}' if a.data_saf_dz else '')
        a.tag += ''.join(f'_shift{int(p)}{"xyb"[int(q)]}{dx:g},{dy:g}' for p, q, dx, dy in a.data_shift)
        a.tag += '_zshift' if a.data_zshift else ''
        a.tag += '_zernike' if a.data_zernike else ''
    if (a.data_shift or a.data_zshift) and a.warp:
        raise SystemExit('--data-shift / --data-zshift are not implemented with --warp')
    a.warps = load_warps(a.warp) if a.warp else None
    return a


# ---------------------------------------------------------------- camera noise (as SGD_simu_precision2.ipynb)

def camera(rng, expected, a):
    # expected photons per pixel (without background) -> pixel values in photons, without camera baseline (as after
    # the reconstruction), with the noise functions of simu_PSF_polarMFM_JAX.py: emccd_jax (photoelectrons, EM register,
    # read noise, ADC; excess factor 2, variance 2*(light)/QE + sigma_r^2) or noise_jax (Poisson and read noise only).
    # The JAX key is drawn from rng, so that a run is reproduced by its seed
    import jax, jax.numpy as jnp
    from simu_PSF_polarMFM_JAX import noise_jax, emccd_jax
    key = jax.random.PRNGKey(int(rng.integers(2**31 - 1)))
    expected = jnp.asarray(expected, dtype=jnp.float32)
    if a.camera == 'poisson':
        return np.asarray(noise_jax(key, expected, b=a.background, sigma_b=a.sigma_b, sigma_r=a.sigma_r), dtype=float)
    # sigma_r is given in photons: read noise in electrons at the output = sigma_r*QE*EM
    return np.asarray(emccd_jax(key, expected, QE=a.QE, EM=a.EM, b=a.background, sigma_b=a.sigma_b,
                                sigma_read_e=a.sigma_r*a.QE*a.EM, sensitivity=4.4, cic=a.cic), dtype=float)


def crazy_pixels(rng, data, a):
    if a.crazy_fraction <= 0:
        return data
    hit = rng.random(data.shape) < a.crazy_fraction
    return data + hit*rng.exponential(a.crazy_amplitude, data.shape)


# ---------------------------------------------------------------- pipeline of SGD_jax.py

def build_pipeline(overrides, out_folder, loss, save_figures, a):
    """executes the cells of SGD_jax.py up to the main loop; returns the namespace and the text of the main loop"""
    source = SGD_batch.read_script(SGD_SCRIPT)
    source = SGD_batch.apply_overrides(source, overrides)
    cells = SGD_batch.split_cells(source)
    ns = {'__name__': '__main__', '__file__': str(SGD_SCRIPT)}
    dialog = SGD_batch.FakeDialog(out_folder)
    show_factory = lambda n: SGD_batch.make_show(n, save_figures)
    main_loop = None
    for first_line, header, text in cells:
        if 'OPTIONAL' in header:
            continue
        if 'extracting positions' in header:
            # no localisation file: interplane of the simulation, and the variance without light of the channels
            # (dark_variance.txt of a reconstruction): the read noise of the simulated camera, sigma_r^2 (ADU^2)
            import jax
            ns['interplane'] = jax.device_put(a.interplane)
            ns['dark_variance_adu'] = np.full(6, a.sigma_r**2)/(ns['sensitivity']/(ns['QE']*ns['EM']))**2
            ns['camera_metadata'] = {} # no raw file: EM of the parameter cell
            continue
        if 'gradient descent' in header:
            marker = '################# main loop'
            split = text.index(marker)
            head, loop = text[:split], text[split:]
            SGD_batch.run_cells(SGD_SCRIPT, [(first_line, header, head)], ns, [], dialog, show_factory)
            main_loop = (first_line + head.count('\n'), loop)
            continue
        SGD_batch.run_cells(SGD_SCRIPT, [(first_line, header, text)], ns, [], dialog, show_factory)
        if 'PARAMETERS TO BE DEFINED' in header:
            # one npz folder per loss instead of NPZ_<date>
            created = Path(ns['save_folder'])
            ns['save_folder'] = Path(out_folder) / f'NPZ_{loss}'
            ns['save_folder'].mkdir(parents=True, exist_ok=True)
            if created.is_dir() and not any(created.iterdir()):
                created.rmdir()
    if main_loop is None:
        raise RuntimeError('gradient descent cell not found in SGD_jax.py')
    return ns, main_loop, source


def batch_builder(source):
    # end of load_batch of SGD_jax.py: builds the batch (result) from the PSF and their noise
    start = source.index("    result['buffer'] = buffer")
    end = source.index('\n', source.index("    result['last_frame_processed']", start))
    return compile(textwrap.dedent(source[start:end]), str(SGD_SCRIPT), 'exec')


def simulated_load_batch(ns, sim, builder, rng_noise, a):
    # replaces load_batch: same signature, the PSF of the next batch are taken in the simulation; the noise model is
    # measured by noise_model of SGD_jax.py on Nframe simulated background frames, as on a chunk of real frames
    counter = [0]
    def load_batch(last_frame_processed, buffer, psf_buffer, NPSF, result, n_skip=0):
        b = counter[0]; counter[0] += 1
        sl = slice(b*NPSF, (b+1)*NPSF)
        noisy_psf = sim['data'][sl]
        if a.warps is None:
            frames = camera(rng_noise, np.zeros((ns['Nframe'], 6, 64, 64)), a)
        else: # raw frames, then the warp of each channel (one region per chunk), as in the reconstruction
            raw = camera(rng_noise, np.zeros((ns['Nframe'], 6, 64 + 2*MARGIN, 64 + 2*MARGIN)), a)
            frames = np.zeros((ns['Nframe'], 6, 64, 64))
            for ch in range(6): # channel 2*plane + pol of raw in load_batch, planes reversed for the PSF
                plane, pol = 2 - ch//2, ch % 2
                mx, my, _ = warp_maps(rng_noise, a.warps, WARP_CANAL[plane][pol], 1, 64 + 2*MARGIN)
                frames[:, ch] = remap(raw[:, ch], np.repeat(mx, len(raw), 0), np.repeat(my, len(raw), 0))[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
        if a.camera == 'poisson':
            # integer values: the levels of noise_model (percentiles of the median over time) would coincide and be
            # empty (gain nan); a uniform dither of 1 makes them continuous and adds only 1/12 to the variance
            frames = frames + rng_noise.uniform(-0.5, 0.5, frames.shape)
        gain, shift, level = ns['noise_model'](frames, ns['dark_variance_adu']*(ns['sensitivity']/(ns['QE']*ns['EM']))**2)
        chunk_noise = np.stack((gain, shift, level))[:, np.arange(6).reshape(3, 2)[::-1]] # as in load_batch
        psf_noise = np.tile(chunk_noise, (NPSF, 1, 1, 1))
        x = np.zeros(NPSF); y = np.zeros(NPSF) # the true position is the offset in um, the patch is centred on 0
        index_frame = np.arange(sl.start, sl.stop).astype(float) # PSF index, saved as 'frame'
        last_frame_processed = last_frame_processed + 1
        local = dict(result=result, buffer=buffer, psf_buffer=psf_buffer, noisy_psf=noisy_psf, x=x, y=y, NPSF=NPSF,
                     psf_noise=psf_noise, index_frame=index_frame, last_frame_processed=last_frame_processed)
        exec(builder, ns, local)
    return load_batch


# ---------------------------------------------------------------- simulation

def grid(n_rho, n_eta, rho_values=None, eta_values=None):
    # regular grid: rho and eta at the centres of n_rho and n_eta bins of [0, 180[, one orientation per (rho, eta),
    # or the lists of values given
    rho_bins = np.array(rho_values, float) if rho_values else (np.arange(n_rho) + 0.5)*180/n_rho
    eta_bins = np.array(eta_values, float) if eta_values else (np.arange(n_eta) + 0.5)*180/n_eta
    rho, eta = np.meshgrid(rho_bins, eta_bins, indexing='ij')
    return rho.ravel(), eta.ravel()


def model_psf(ns, sl, p, half=6, fields=None, zernike=None):
    # expected photons of the 6 channels ((2 half + 1)^2 patch, 13 x 13 by default) for the parameters p[sl], model of
    # SGD_jax.py; fields: pupil fields replacing the ones of ns (data with another sample index)
    import jax.numpy as jnp
    f32 = lambda v: jnp.array(v[sl], dtype=jnp.float32)
    F = {k: ns[k] for k in ('xx', 'yy', 'th1', 'phi', 'Ex0', 'Ex1', 'Ex2', 'Ey0', 'Ey1', 'Ey2')}
    F.update(fields or {})
    M = ns['compute_M_jax'](xp=f32(p['x']), yp=f32(p['y']), zp=f32(p['z']), d=ns['d_'], x=F['xx'], y=F['yy'],
                            th1=F['th1'], phi=F['phi'], Ex0=F['Ex0'], Ex1=F['Ex1'], Ex2=F['Ex2'], Ey0=F['Ey0'],
                            Ey1=F['Ey1'], Ey2=F['Ey2'], u=ns['u'], v=ns['v'], phase_maskx=ns['phase_mask'],
                            phase_masky=ns['phase_mask'], zernike_base=ns['zernike_base'],
                            zernike_coefs_x=ns['zernike_coefs_x'] if zernike is None else zernike[0],
                            zernike_coefs_y=ns['zernike_coefs_y'] if zernike is None else zernike[1],
                            second_plane=ns['second_plane'], polar_projections=ns['polar_projections'],
                            lambd=ns['lambd'], f_tube=ns['f_tube'])
    h = ns['PSF_jax'](rho=f32(p['rho']), eta=f32(p['eta']), delta=f32(p['delta']), M=M, N_photons=f32(p['N']))
    dim = int(ns['dim_simu'])
    return np.array(h * ns['channel_gain']()[None, :, :, None, None])[:, :, :, dim-half:dim+half+1, dim-half:dim+half+1]


# ---------------------------------------------------------------- warp of the reconstruction

WARP_CANAL = [[2, 5], [1, 4], [0, 3]] # channel of subcanals (reconstruction_juliette.py) of each (plane, polarisation)
MARGIN = 2 # pixels around the patch for the cubic kernel


def load_warps(path):
    cal = np.load(path, allow_pickle=True)
    X, Y = cal['X'], cal['Y']
    return dict(inv=[np.linalg.inv(cal[f'M_{i}']) for i in range(6)], size=(int(Y[1] - Y[0]), int(X[3] - X[2])))


def warp_maps(rng, warps, canal, n, side):
    # maps of cv2.remap reproducing cv2.warpPerspective(INTER_CUBIC) of the reconstruction on side x side regions at n
    # random places of the channel: dst pixel -> src = M^-1 dst (same kernel, same sub-pixel position per pixel). The
    # integer part of the displacement at the centre is removed; shift = remaining (row, col) displacement at the
    # centre: the content at src p appears at dst p - shift
    H, W = warps['size']
    r0 = rng.integers(0, H - side, n)[:, None, None]; c0 = rng.integers(0, W - side, n)[:, None, None]
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    X, Y = c0 + jj, r0 + ii
    m = warps['inv'][canal]
    den = m[2, 0]*X + m[2, 1]*Y + m[2, 2]
    dx = (m[0, 0]*X + m[0, 1]*Y + m[0, 2])/den - X; dy = (m[1, 0]*X + m[1, 1]*Y + m[1, 2])/den - Y
    c = side//2
    ox, oy = np.round(dx[:, c, c])[:, None, None], np.round(dy[:, c, c])[:, None, None]
    map_x = (jj + dx - ox).astype(np.float32); map_y = (ii + dy - oy).astype(np.float32)
    return map_x, map_y, np.stack([(dy - oy)[:, c, c], (dx - ox)[:, c, c]], axis=1)


def remap(images, map_x, map_y):
    import cv2
    return np.stack([cv2.remap(np.ascontiguousarray(im, dtype=np.float32), mx, my, cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101) for im, mx, my in zip(images, map_x, map_y)])


def pixels_per_um(ns, a):
    # displacement (row, col) of the PSF in each channel of the patch for 1 um of x and of y (centroids)
    n = int(ns['NPSF']) # the model works on whole batches; PSF 0, 1, 2: reference, +0.06 um in x, +0.06 um in y
    p = dict(rho=np.zeros(n), eta=np.zeros(n), delta=np.full(n, a.delta), N=np.full(n, 1e4), x=np.zeros(n),
             y=np.zeros(n), z=np.full(n, 0.6))
    p['x'][1] = 0.06; p['y'][2] = 0.06
    h = model_psf(ns, slice(0, n), p, half=10)[:3]
    ii, jj = np.meshgrid(np.arange(h.shape[-2]), np.arange(h.shape[-1]), indexing='ij')
    cen = np.stack([(h*ii).sum((-2, -1))/h.sum((-2, -1)), (h*jj).sum((-2, -1))/h.sum((-2, -1))], axis=-1) # (3, 3, 2, 2)
    return np.stack([cen[1] - cen[0], cen[2] - cen[0]], axis=-1)/0.06 # (plane, pol, (row, col), (x, y))


def warped_data(ns, a, rng, rng_warp, sl, truth, second, neighbour, A):
    # expected raw patches of the 6 channels, each at the position that the warp brings back to the truth, camera
    # noise and crazy pixels on the raw patches, cubic warp of the reconstruction, 13 x 13 crop
    n = sl.stop - sl.start; side = 13 + 2*MARGIN
    out = np.zeros((n, 3, 2, 13, 13))
    for plane in range(3):
        for pol in range(2):
            map_x, map_y, shift = warp_maps(rng_warp, a.warps, WARP_CANAL[plane][pol], n, side)
            move = np.linalg.solve(A[plane, pol], shift.T).T # um of (x, y) that move the content by shift
            def placed(p):
                q = dict(p); q['x'] = p['x'].copy(); q['y'] = p['y'].copy()
                q['x'][sl] += move[:, 0]; q['y'][sl] += move[:, 1]
                return q
            h = model_psf(ns, sl, placed(truth), half=6 + MARGIN)[:, plane, pol]
            if neighbour[sl].any():
                h = h + neighbour[sl][:, None, None]*model_psf(ns, sl, placed(second), half=6 + MARGIN)[:, plane, pol]
            if (plane, pol) == (1, 1):
                h = h*a.data_y_gain_J2
            raw = crazy_pixels(rng, camera(rng, h, a), a)
            out[:, plane, pol] = remap(raw, map_x, map_y)[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
    return out


class data_physics:
    ''' context in which model_psf computes the data with the mismatch of --data-n2 / --data-saf-dz: the globals of
    simu_PSF_polarMFM_JAX.py (n2, psi_z_jit) are changed and the compiled functions cleared, then restored (and
    cleared again) for the fit. self.fields: pupil fields of the data, recomputed with the sample index of the data '''
    def __init__(self, ns, a):
        self.ns, self.a, self.fields = ns, a, None
        self.active = a.data_n2 is not None or a.data_saf_dz != 0
    def __enter__(self):
        if not self.active:
            return self
        import jax, jax.numpy as jnp
        import simu_PSF_polarMFM_JAX as SP
        ns, a = self.ns, self.a
        self.saved = (SP.n2, SP.psi_z_jit)
        if a.data_n2 is not None:
            SP.n2 = a.data_n2
        if a.data_saf_dz:
            dz = a.data_saf_dz
            def psi_z_saf(theta, z, lambd=0.617, cos_th2=None):
                sq = jnp.sqrt(1 - (SP.n1*jnp.sin(theta)/SP.n2)**2) # imaginary in the SAF region: exp(-k kappa z)
                factor = 2*jnp.pi*SP.n2/lambd
                out = jnp.einsum('a, bc -> abc', z, factor*sq)
                return out + dz*factor*1j*jnp.imag(sq)[None] # evanescent part only, as at z + dz
            SP.psi_z_jit = jax.jit(psi_z_saf)
        jax.clear_caches()
        if a.data_n2 is not None:
            out = SP.vectorial_BFP_perfect_focus_jax(int(ns['N']), NA=ns['NA'], mag=ns['mag'], lambd_nm=ns['lambd'],
                                                     f_tube_mm=ns['f_tube'], J_dichroic=ns['J_dichroic'], SAF=ns['SAF'])
            xx, yy, th1, phi, (Ex0, Ex1, Ex2), (Ey0, Ey1, Ey2) = out[:6]
            pad = lambda f: SP.pad_jax(f, ns['Npadding']).astype(jnp.complex64)
            self.fields = dict(xx=pad(xx), yy=pad(yy), th1=pad(th1), phi=pad(phi), Ex0=pad(Ex0), Ex1=pad(Ex1),
                               Ex2=pad(Ex2), Ey0=pad(Ey0), Ey1=pad(Ey1), Ey2=pad(Ey2))
        return self
    def __exit__(self, *exc):
        if self.active:
            import jax
            import simu_PSF_polarMFM_JAX as SP
            SP.n2, SP.psi_z_jit = self.saved
            jax.clear_caches()
        return False


def data_psf(ns, sl, p, a, phys, half=6):
    # model_psf for the data: mismatch of the sample (phys.fields), aberrations (--data-zernike) and lateral shift of
    # channels (--data-shift, --data-zshift)
    zern = None
    if a.data_zernike:
        import jax.numpy as jnp
        zf = np.load(a.data_zernike)
        zern = (jnp.array(zf['zx'], dtype=jnp.complex64), jnp.array(zf['zy'], dtype=jnp.complex64))
    h = model_psf(ns, sl, p, half, phys.fields, zern)
    for plane, pol, dx, dy in a.data_shift:
        q = dict(p); q['x'] = p['x'] + dx/1000; q['y'] = p['y'] + dy/1000
        hs = model_psf(ns, sl, q, half, phys.fields, zern)
        pols = (0, 1) if int(pol) == 2 else (int(pol),)
        for k in pols:
            h[:, int(plane), k] = hs[:, int(plane), k]
    if a.data_zshift:
        table = np.load(a.data_zshift)
        for plane in range(3):
            for pol in range(2):
                sh = table['shift'][plane, pol]
                if not np.any(sh):
                    continue
                q = dict(p); q['x'] = p['x'].copy(); q['y'] = p['y'].copy()
                q['x'][sl] = p['x'][sl] + np.interp(p['z'][sl], table['z_em'], sh[:, 0])/1000
                q['y'][sl] = p['y'][sl] + np.interp(p['z'][sl], table['z_em'], sh[:, 1])/1000
                h[:, plane, pol] = model_psf(ns, sl, q, half, phys.fields, zern)[:, plane, pol]
    return h


def simulate(ns, a, rng):
    NPSF = int(ns['NPSF'])
    rho_list, eta_list = grid(a.n_rho, a.n_eta, a.rho_values, a.eta_values)
    point = np.repeat(np.arange(len(rho_list)), a.per_point)
    n_total = int(np.ceil(len(point)/NPSF))*NPSF # whole batches: the last PSF are repeated points
    point = np.concatenate([point, point[:n_total-len(point)]])
    truth = dict(rho=rho_list[point], eta=eta_list[point], delta=np.full(n_total, a.delta),
                 N=np.full(n_total, a.N), x=rng.uniform(-a.xy_range, a.xy_range, n_total),
                 y=rng.uniform(-a.xy_range, a.xy_range, n_total), z=rng.uniform(*a.z_range, n_total))
    # second emitter: random orientation over the half-sphere, N, z, at neighbour_distance pixels in a random direction
    neighbour = rng.random(n_total) < a.two_emitter_fraction
    distance = 0.12*rng.uniform(*a.neighbour_distance, n_total); angle = rng.uniform(0, 2*np.pi, n_total)
    second = dict(rho=rng.uniform(0, 180, n_total), eta=np.rad2deg(np.arccos(rng.uniform(-1, 1, n_total))),
                  delta=np.full(n_total, a.delta), N=a.N*rng.uniform(*a.neighbour_N, n_total),
                  x=truth['x'] + distance*np.cos(angle), y=truth['y'] + distance*np.sin(angle),
                  z=rng.uniform(*a.z_range, n_total))
    data = []
    if a.warps is not None:
        A = pixels_per_um(ns, a); rng_warp = np.random.default_rng(a.seed + 2) # truth unchanged by the warp draws
    with data_physics(ns, a) as phys:
        for b in range(n_total//NPSF):
            sl = slice(b*NPSF, (b+1)*NPSF)
            if a.warps is not None:
                data.append(warped_data(ns, a, rng, rng_warp, sl, truth, second, neighbour, A))
                continue
            h = data_psf(ns, sl, truth, a, phys)
            if neighbour[sl].any():
                h = h + neighbour[sl][:, None, None, None, None]*data_psf(ns, sl, second, a, phys)
            h[:, 1, 1] *= a.data_y_gain_J2 # y of plane 1 (1: no change)
            data.append(crazy_pixels(rng, camera(rng, h, a), a))
    return dict(data=np.concatenate(data).astype(np.float32), point=point, rho_list=rho_list, eta_list=eta_list,
                neighbour=neighbour, **{'true_'+k: v for k, v in truth.items()}, **{'second_'+k: v for k, v in second.items()})


# ---------------------------------------------------------------- errors and statistics

def wrap(v, period=180):
    return (v + period/2) % period - period/2


def collect(npz_folder, sim):
    files = sorted([f for f in Path(npz_folder).glob('*.npz') if f.stem.isdigit()], key=lambda f: int(f.stem))
    found = {k: np.concatenate([np.load(f)[k] for f in files]) for k in ('frame', 'x', 'y', 'z', 'rho', 'eta', 'delta', 'N_photons', 'deviance')}
    idx = found['frame'].astype(int)
    t = {k: sim['true_'+k][idx] for k in ('rho', 'eta', 'delta', 'N', 'x', 'y', 'z')}
    correction_dft = 0.9952298 # as in SGD_jax.py: saved x = 1000*x_found/correction_dft (patch centred on 0)
    err = {}
    raw = found['rho'] - t['rho']
    err['rho'] = wrap(raw)
    eta = found['eta'].copy()
    crossed = np.abs(err['rho'] - raw) > 1 # rho brought back across 0/180: equivalent orientation (rho+180, 180-eta)
    eta[crossed] = 180 - eta[crossed]
    err['eta'] = wrap(eta - t['eta'])
    err['delta'] = found['delta'] - t['delta']
    err['x'] = found['x']*correction_dft - 1000*t['x'] # nm
    err['y'] = found['y']*correction_dft - 1000*t['y']
    err['z'] = found['z'] - 1000*t['z']
    err['N'] = (found['N_photons'] - t['N'])/t['N'] # relative
    return idx, found, t, err


def robust_std(v):
    return 1.4826*np.median(np.abs(v - np.median(v)))


def statistics(idx, err, sim, subset=None):
    # bias and precision per orientation point, on all the PSF or on a subset (boolean per found PSF)
    g = sim['point'][idx]
    rows = []
    for i, (rho, eta) in enumerate(zip(sim['rho_list'], sim['eta_list'])):
        sel = g == i
        if subset is not None:
            sel = sel & subset
        failed = sel & ~np.isfinite(err['eta']) # fits that diverged (nan): counted, left out of the statistics
        row = dict(point=i, rho=rho, eta=eta, n=int(sel.sum()), n_failed=int(failed.sum()))
        for k, v in err.items():
            ok = sel & np.isfinite(v)
            row['bias_'+k] = np.median(v[ok]) if ok.any() else np.nan
            row['sigma_'+k] = robust_std(v[ok]) if ok.any() else np.nan
        rows.append(row)
    return rows


# ---------------------------------------------------------------- figures

def disc_map(ax, rows, values, cmap, vmin, vmax):
    # binned disc of the regular grid, one bin per simulated (rho, eta), no interpolation. Polar plot: angle rho,
    # radius eta <= 90; (rho, eta) and (rho+180, 180-eta) are the same orientation, so the bins eta > 90 are shown at
    # rho+180, 180-eta (as in SGD_simu_precision2.ipynb)
    rho_bins = np.unique([r['rho'] for r in rows]); eta_bins = np.unique([r['eta'] for r in rows])
    n_rho, n_eta = len(rho_bins), len(eta_bins)
    if n_eta % 2:
        raise ValueError('n_eta must be even for the polar maps (bins split at eta = 90)')
    table = np.full((n_rho, n_eta), np.nan)
    for r, v in zip(rows, values):
        table[np.searchsorted(rho_bins, r['rho']), np.searchsorted(eta_bins, r['eta'])] = v
    half = n_eta//2
    # (radius bin, angle bin): angles 0-180 eta < 90, angles 180-360 eta > 90 with radius 180 - eta
    shown = np.concatenate([table[:, :half].T, table[:, half:][:, ::-1].T], axis=1)
    th_edges = np.deg2rad(np.linspace(0, 360, 2*n_rho + 1)); r_edges = np.linspace(0, 90, half + 1)
    im = ax.pcolormesh(th_edges, r_edges, shown, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.set_rlim(0, 90); ax.set_rticks([30, 60]); ax.set_yticklabels([]); ax.grid(color='k', alpha=0.25, lw=0.5)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(['0', '90', '180', '270'], fontsize=7)
    ax.tick_params(pad=0)
    return im


def case_figure(all_rows, folder, case):
    # one figure per case: bias (3 rows: losses) and precision (3 rows) of rho, eta, delta, N, on the half-sphere
    import matplotlib.pyplot as plt
    params = [('rho', 'deg', 1), ('eta', 'deg', 1), ('delta', 'deg', 1), ('N', '%', 100)]
    losses = list(all_rows)
    blocks = (('bias', 'bias (median of the error)', 'coolwarm'), ('sigma', 'precision (robust std)', 'viridis'))
    fig = plt.figure(figsize=(3.2*len(params), 2*3.0*len(losses) + 1.6), layout='constrained')
    fig.suptitle(f'{case}: bias and precision, one bin per simulated (rho, eta) of the grid\n'
                 f'angle = rho, radius = eta (circles at 30, 60 deg; eta > 90 shown at rho+180, 180-eta), same color scale for the 3 losses within a block', fontsize=12)
    for subfig, (kind, title, cmap) in zip(fig.subfigures(2, 1, hspace=0.04), blocks):
        subfig.suptitle(title, fontsize=12, fontweight='bold')
        axs = np.atleast_2d(subfig.subplots(len(losses), len(params), subplot_kw=dict(projection='polar')))
        for c, (k, unit, scale) in enumerate(params):
            values = {l: scale*np.array([r[f'{kind}_{k}'] for r in all_rows[l]], dtype=float) for l in losses}
            allv = np.concatenate(list(values.values())); allv = allv[np.isfinite(allv)]
            if kind == 'bias':
                lim = np.percentile(np.abs(allv), 95) if allv.size else 1; vmin, vmax = -lim, lim
            else:
                vmin, vmax = 0, (np.percentile(allv, 95) if allv.size else 1)
            for li, loss in enumerate(losses):
                ax = axs[li, c]
                im = disc_map(ax, all_rows[loss], values[loss], cmap, vmin, vmax)
                med = np.nanmedian(np.abs(values[loss])) if kind == 'bias' else np.nanmedian(values[loss])
                ax.set_title(f'{k}, {"median |bias|" if kind == "bias" else "median"} {med:.1f} {unit}', fontsize=9)
                if c == 0:
                    ax.text(-0.18, 0.5, loss, transform=ax.transAxes, rotation=90, ha='center', va='center', fontsize=11, fontweight='bold')
            cb = subfig.colorbar(im, ax=axs[:, c], location='bottom', shrink=0.85, aspect=20)
            cb.set_label(f'{k} {"bias" if kind == "bias" else "precision"} ({unit})', fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(Path(folder) / f'polar_summary_{case}.png', dpi=100); plt.close(fig)


def compare_losses(all_rows, folder):
    import matplotlib.pyplot as plt
    params = [('rho', 'deg'), ('eta', 'deg'), ('delta', 'deg'), ('z', 'nm'), ('x', 'nm'), ('N', 'relative')]
    fig, axs = plt.subplots(2, len(params), figsize=(4*len(params), 7))
    for loss, rows in all_rows.items():
        eta_values = np.unique([r['eta'] for r in rows])
        for c, (k, unit) in enumerate(params):
            for r_, kind in enumerate(('bias', 'sigma')):
                prof = [np.nanmedian([r[f'{kind}_{k}'] for r in rows if r['eta'] == e]) for e in eta_values]
                axs[r_, c].plot(eta_values, prof, 'o-', label=loss)
                axs[r_, c].set(title=f'{k} {kind} ({unit})', xlabel='true eta (deg)')
    for ax in axs[0]:
        ax.axhline(0, color='k', lw=0.8)
    axs[0, 0].legend()
    fig.suptitle('median over rho of the bias (top) and of the precision (bottom), versus the true eta', fontsize=11)
    fig.tight_layout(); fig.savefig(Path(folder) / 'compare_losses_vs_eta.png', dpi=110); plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv):
    global SGD_SCRIPT
    a = parse_arguments(argv)
    if a.cnn:
        SGD_SCRIPT = HERE / 'SGD_jax_CNN.py'
    run_folder = Path(a.out) / (f'{datetime.now():%Y-%m-%d_%Hh%M}' + (f'_{a.tag}' if a.tag else ''))
    run_folder.mkdir(parents=True, exist_ok=True)
    log(f'results in {run_folder}')
    log('parameters: ' + ', '.join(f'{k}={v}' for k, v in vars(a).items() if k != 'warps'))
    extra = SGD_batch.parse_overrides(a.set)
    calibration = {'J1': 'np.eye(2)', 'J2': 'rot(45)', 'rotation': '0', 'rotation2': '0', 'rho_offset': '0',
                   'unitary_J': 'False', 'y_gain_J1': '1.', 'y_gain_J2': '1.'}
    if a.calibration: # calibration of a real run (same for the data and the fit), rho of the model (no rho_offset)
        from density_experimental import run_parameters
        real = run_parameters(a.calibration)[0]
        calibration.update({k: v for k, v in real.items() if k in ('J1', 'J2', 'rotation', 'rotation2', 'unitary_J', 'y_gain_J1', 'y_gain_J2')})
        log('calibration of ' + a.calibration + ': ' + ', '.join(f'{k}={calibration[k]}' for k in ('rotation', 'rotation2', 'unitary_J', 'y_gain_J1', 'y_gain_J2')))
    rng = np.random.default_rng(a.seed)
    sim = None
    all_rows = {}
    for loss in a.losses:
        start = time.time()
        overrides = dict(calibration, loss2=loss, middle_plane=repr(-a.d), **extra) # d_ = -middle_plane
        ns, (loop_first_line, loop_text), source = build_pipeline(overrides, run_folder, loss, a.save_figures, a)
        if sim is None: # same simulated data for all the losses
            log('simulating the data')
            sim = simulate(ns, a, rng)
            np.savez_compressed(run_folder / 'simulation.npz', **sim)
            log(f'{len(sim["data"])} PSF ({a.n_rho} rho x {a.n_eta} eta x {a.per_point}), '
                f'{len(sim["data"])//int(ns["NPSF"])} batches, {int(sim["neighbour"].sum())} with a second emitter')
        ns['batch_nb'] = len(sim['data'])//int(ns['NPSF'])
        ns['first_batch'] = 0
        ns['load_batch'] = simulated_load_batch(ns, sim, batch_builder(source), np.random.default_rng(a.seed + 1), a)
        log(f'pipeline of SGD_jax.py with loss2 = {loss}')
        SGD_batch.run_cells(SGD_SCRIPT, [(loop_first_line, 'main loop', loop_text)], ns, [],
                            SGD_batch.FakeDialog(run_folder), lambda n: SGD_batch.make_show(n, a.save_figures))
        idx, found, truth, err = collect(ns['save_folder'], sim)
        np.savez_compressed(run_folder / f'results_{loss}.npz', index=idx, neighbour=sim['neighbour'][idx],
                            **{'found_'+k: v for k, v in found.items()}, **{'true_'+k: v for k, v in truth.items()},
                            **{'error_'+k: v for k, v in err.items()})
        all_rows[loss] = loss_statistics(loss, idx, err, sim, run_folder, a)
        log(f'{loss} done in {(time.time()-start)/60:.1f} min')
        del ns
        import jax
        if hasattr(jax, 'clear_caches'): # compiled functions of this loss are not needed any more
            jax.clear_caches()
    summary(all_rows, run_folder)
    log(f'done, results in {run_folder}')


def median_line(rows, keys=('rho', 'eta', 'delta', 'x', 'y', 'z', 'N')):
    return '  '.join(f'{k} {np.nanmedian([abs(r["bias_"+k]) for r in rows]):6.2f}/{np.nanmedian([r["sigma_"+k] for r in rows]):6.2f}'
                     for k in keys)


def loss_statistics(loss, idx, err, sim, run_folder, a):
    # statistics per orientation, polar maps and log of one loss
    rows = statistics(idx, err, sim)
    failed = ~np.isfinite(err['eta'])
    log(f'{loss}: {int(failed.sum())} of {len(failed)} fits diverged (nan), left out; median over the orientations of '
        f'|bias| / precision: {median_line(rows)}')
    if sim['neighbour'].any(): # isolated PSF and PSF with a second emitter separately
        for name, subset in (('isolated', ~sim['neighbour'][idx]), ('neighbour', sim['neighbour'][idx])):
            log(f'  {loss} {name:9s} ({int(subset.sum())} PSF, {int((subset & failed).sum())} nan): '
                f'{median_line(statistics(idx, err, sim, subset))}')
    return rows


def summary(all_rows, run_folder):
    import pandas as pd
    pd.concat([pd.DataFrame(r).assign(loss=l) for l, r in all_rows.items()]).to_csv(run_folder / 'summary.csv', index=False)
    compare_losses(all_rows, run_folder)
    try: # polar maps: only for a regular grid with an even number of eta bins
        case_figure(all_rows, run_folder, run_folder.name.split('_', 2)[-1] if run_folder.name.count('_') >= 2 else run_folder.name)
    except ValueError as e:
        log(f'no polar maps: {e}')
    log('summary per loss (median over the orientations): |bias| / precision, angles in deg, x and z in nm, N relative')
    for loss, rows in all_rows.items():
        log(f'  {loss:8s} {median_line(rows)}  (nan fits: {sum(r["n_failed"] for r in rows)})')


def recompute(folder):
    # statistics, figures and summary of a finished run again, from its simulation.npz and results_<loss>.npz
    folder = Path(folder)
    sim = dict(np.load(folder / 'simulation.npz'))
    a = None
    all_rows = {}
    for loss in ('poisson', 'lms', 'huber'):
        f = folder / f'results_{loss}.npz'
        if f.exists():
            r = np.load(f)
            err = {k[6:]: r[k] for k in r.files if k.startswith('error_')}
            all_rows[loss] = loss_statistics(loss, r['index'], err, sim, folder, a)
    summary(all_rows, folder)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--recompute']: # python SGD_precision_claude.py --recompute <result folder> ...
        for folder in sys.argv[2:]:
            recompute(folder)
    else:
        main(sys.argv[1:])
