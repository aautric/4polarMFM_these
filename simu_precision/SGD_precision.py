# -*- coding: utf-8 -*-
"""
Precision and bias of the SGD pipeline of SGD_jax.py across rho and eta, on simulated data.

The pipeline is the one of SGD_jax.py, executed cell by cell (as SGD_batch.py does): SGD1, SGD2, losses, noise model,
score, saving of the npz files. Only the data are simulated: the reading of the frames in load_batch is replaced by
simulated PSF (the end of load_batch, which builds the batch for the descent, is taken from SGD_jax.py itself), and
the noise model of each batch is measured by noise_model of SGD_jax.py on simulated background frames.

Simulation (in photons):
    PSF of the model of SGD_jax.py (or SGD_jax_CNN.py with --cnn: same model), ideal polarisation calibration
    (J1 = identity, J2 = rot(45): a dipole at rho 45 deg lights the x channel of the middle plane; the torch model of the
    CRB needs polar_projections 0/-45/0 for the same convention, review item 18; rotations 0, rho_offset 0, y gains 1) unless --calibration; the same model
    (and calibration) for the data and the fit unless a --data-* mismatch option is given
    true delta = --delta (90 by default) for every PSF, or uniform in --delta-range per PSF
    light of a pixel = s (PSF, photons) + b + sigma_b * N(0,1), clipped at 0: b = mean background (photons/pixel),
        sigma_b = fluctuation of the background BEYOND its Poisson noise (0: the background has only its Poisson noise).
        b = 0: no background at all (nothing but the molecule)
    camera noise of simu_PSF_polarMFM_JAX.py on that light:
        --camera emccd: emccd_jax (photoelectrons Poisson(QE x light), EM register Gamma(e, EM), read noise sigma_r,
            ADC rounding), back in photons: variance 2 x light/QE + sigma_r^2 (EM excess factor 2, QE thinning)
        --camera poisson: noise_jax, Poisson(light) + read noise sigma_r: variance light + sigma_r^2 (NO QE, no excess:
            the variance ratio emccd / poisson is 2/QE = 2.17, precision ratio sqrt(2/QE) = 1.47)
    No camera offset: the reconstruction removes the baseline of the raw frames, so the data are 0 without light
    noise model of the fit (var = gain x model + V, see noise_model of SGD_jax.py): measured by noise_model on simulated
        background frames, as on real data (correct for b >= 5: review 2026-11-12), or, with b = 0 and sigma_b = 0
        (nothing to measure), the true noise of the simulation (gain 1 or 2/QE, V = sigma_r^2)
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
                 WARNING: background 25 included (Poisson noise of the PSF AND of a background of 25 photons/pixel);
                 for "Poisson only, no background" add --background 0
    slb          SLB 02-02: N 1170 (390 per plane), background 3.4, offset 3.5
    actin        RPE1 actin 09-18 / 09-23: N 3570 / 1965 (1190 / 655 per plane), background 20 / 35, offset 3.8 / 4.3
                 -> N 3000, background 25, sigma_b 3 (extra fluctuation), read noise 0.4 photons, EMCCD
    actin_dense  actin + 3 % crazy pixels + a second emitter on 20 % of the PSF
    ("offset" above = background level measured above the camera baseline in the real data; not simulated)
    Noise types of the user (2026-11-12): "Poisson" = --preset poisson --background 0; "Poisson + read + excess" =
    --preset actin --background 0 --sigma-b 0; "+ background" = --preset actin --sigma-b 0 (b 25 with its Poisson noise)

usage (python of the environment, e.g. /home/amaury/exit/envs/env_linux/bin/python):
    python SGD_precision.py [options] [--set name=value ...]
    e.g. python SGD_precision.py --preset actin_dense --losses poisson lms huber --out /mnt/d/Amaury/simu_claude
options: see python SGD_precision.py --help
    python SGD_precision.py --recompute <result folder> [...]: statistics, figures and summary of finished runs
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

HERE = Path(__file__).resolve().parents[1]  # root of the repo (this file: simu_precision/SGD_precision.py, was SGD_precision_claude.py at the root until 2026-11-12)
sys.path.insert(0, str(HERE))
import SGD_batch # helpers to read and change SGD_jax.py and run its cells (not modified)

SGD_SCRIPT = HERE / 'SGD_jax.py'
log = SGD_batch.log


# ---------------------------------------------------------------- parameters

# Values used for every option left at None on the command line (see parse_arguments). N = photons of the whole
# molecule; background, sigma_b, sigma_r in photons per pixel (sigma_r: read noise in photons, i.e. sigma_r*QE*EM
# electrons at the EM output). Every preset has a background (25 or 3.4): --background 0 for none.
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


def noise_description(a):
    # Purpose: the noise of the simulated data in plain words, in the vocabulary of the user (review item 2):
    # "Poisson" = no background, perfect detector; "Poisson + read + excess" = EMCCD; "+ background" = b with its
    # Poisson noise (and its extra fluctuation sigma_b if > 0)
    parts = ['Poisson (perfect detector, no QE)' if a.camera == 'poisson' else f'Poisson + excess (EMCCD, QE {a.QE:g}, EM {a.EM:g})']
    if a.sigma_r: parts.append(f'read noise {a.sigma_r:g} photons')
    if a.cic: parts.append(f'clock induced charge {a.cic:g} e-')
    parts.append(f'background {a.background:g} photons/pixel with its Poisson noise' if a.background else 'NO background')
    if a.sigma_b: parts.append(f'background fluctuation {a.sigma_b:g} photons')
    if a.crazy_fraction: parts.append(f'crazy pixels {100*a.crazy_fraction:g} %')
    if a.two_emitter_fraction: parts.append(f'second emitter on {100*a.two_emitter_fraction:g} % of the PSF')
    return f'N {a.N:g} photons per molecule; noise: ' + ', '.join(parts)


def parse_arguments(argv):
    # Reads the command line (list of words argv) and returns one object `a` with a field per option (a.N, a.background,
    # ...), used by the rest of the script. Steps: (1) declaration of the accepted options (name, type, number of
    # values, allowed values, default, help); (2) reading of argv (unknown option or wrong type: stop before any
    # computation); (3) options left at None and not typed take the value of the preset (what is typed wins, what is
    # not typed comes from the preset: all the presets have a background); (4) default name of the run (--tag) and
    # consistency check. The optical parameters (wavelength, NA, magnifications, pixel, SAF) are options since
    # 2026-11-12 (review item 8): None = the value of the parameter cell of SGD_jax*.py; the values used are logged
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default='actin', choices=list(PRESETS), help='noise and photon conditions (see above)')
    p.add_argument('--losses', nargs='+', default=['poisson', 'lms', 'huber'], choices=['poisson', 'lms', 'huber'])
    p.add_argument('--n-rho', type=int, default=20, help='number of rho bins in [0, 180[')
    p.add_argument('--n-eta', type=int, default=20, help='number of eta bins in [0, 180[ (even: the bins split at 90)')
    p.add_argument('--per-point', type=int, default=80, help='number of PSF per orientation')
    p.add_argument('--delta', type=float, default=90., help='true wobbling cone (deg)')
    p.add_argument('--delta-range', type=float, nargs=2, default=None, metavar=('LO', 'HI'), help='true delta drawn '
                   'per PSF, uniform in [LO, HI] (deg), instead of --delta: a start of the fit near one common true delta '
                   'is not mistaken for precision (slope of found vs true delta in the log)')
    # None: value of the preset
    p.add_argument('--camera', default=None, choices=['emccd', 'poisson'], help='emccd: Poisson (after QE) + EM excess + read noise; poisson: Poisson noise (no QE) + read noise (no camera offset in either)')
    p.add_argument('--N', type=float, default=None, help='photons of each PSF (whole molecule, 3 planes)')
    p.add_argument('--background', type=float, default=None, help='background b (photons/pixel)')
    p.add_argument('--sigma-b', type=float, default=None, help='fluctuation of the background per pixel and frame (photons)')
    p.add_argument('--sigma-r', type=float, default=None, help='read noise (photons)')
    p.add_argument('--QE', type=float, default=0.92)
    p.add_argument('--cic', type=float, default=0., help='clock induced charge (electrons per pixel and frame, emccd camera)')
    p.add_argument('--EM', type=float, default=200.)
    p.add_argument('--interplane', type=float, default=0.354, help='distance between the planes (um)')
    p.add_argument('--d', type=float, default=-1.0, help='d_ of compute_M_jax (um), d_ = -middle_plane in SGD_jax.py (same as --middle-plane)')
    p.add_argument('--middle-plane', type=float, default=None, help='focus of the middle plane above the coverslip (um); '
                   'sets --d = -middle_plane (review item 14, clearer than --d)')
    # optics of the model (data and fit), review item 8: None = the value of the parameter cell of SGD_jax*.py, written in the log
    p.add_argument('--lambda-emission', type=float, default=None, help='emission wavelength (nm), default: SGD_jax.py (620)')
    p.add_argument('--NA', type=float, default=None, help='numerical aperture, default: SGD_jax.py (1.4)')
    p.add_argument('--mag', type=float, default=None, help='objective magnification, default: SGD_jax.py (100)')
    p.add_argument('--MAG', type=float, default=None, help='extra magnification after the objective, default: SGD_jax.py (200/150)')
    p.add_argument('--l-pixel', type=float, default=None, help='camera pixel (um), default: SGD_jax.py (16)')
    p.add_argument('--SAF', default=None, choices=['True', 'False'], help='supercritical angle fluorescence in the model, default: SGD_jax.py (True)')
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
    for k, v in PRESETS[a.preset].items(): # options not typed (None) <- value of the preset
        if getattr(a, k) is None:
            setattr(a, k, v)
    if not a.tag: # default name of the run: preset + the special options
        a.tag = a.preset + ('_warp' if a.warp else '') + (f'_ygain{a.data_y_gain_J2:g}' if a.data_y_gain_J2 != 1 else '')
        a.tag += (f'_n2{a.data_n2:g}' if a.data_n2 else '') + (f'_safdz{a.data_saf_dz:g}' if a.data_saf_dz else '')
        a.tag += ''.join(f'_shift{int(p)}{"xyb"[int(q)]}{dx:g},{dy:g}' for p, q, dx, dy in a.data_shift)
        a.tag += '_zshift' if a.data_zshift else ''
        a.tag += '_zernike' if a.data_zernike else ''
        a.tag += f'_delta{a.delta_range[0]:g}-{a.delta_range[1]:g}' if a.delta_range else ''
    if (a.data_shift or a.data_zshift) and a.warp:
        raise SystemExit('--data-shift / --data-zshift are not implemented with --warp')
    if (a.data_zernike or a.data_n2 is not None) and a.warp: # review item 13: warped_data uses model_psf, not data_psf
        raise SystemExit('--data-zernike / --data-n2 are not implemented with --warp')
    if a.middle_plane is not None:
        a.d = -a.middle_plane
    a.warps = load_warps(a.warp) if a.warp else None
    return a


# ---------------------------------------------------------------- camera noise (as SGD_simu_precision2.ipynb, removed 2026-11-12, docs/removed_files.md)

def camera(rng, expected, a):
    # expected photons per pixel (without background) -> pixel values in photons, without camera baseline (as after
    # the reconstruction), with the noise functions of simu_PSF_polarMFM_JAX.py: emccd_jax (photoelectrons, EM register,
    # read noise, ADC; excess factor 2, variance 2*(light)/QE + sigma_r^2) or noise_jax (Poisson and read noise only).
    # The JAX key is drawn from rng, so that a run is reproduced by its seed.
    # Both functions add the background inside (light = expected + b + sigma_b N(0,1), clipped at 0), so `expected`
    # is the PSF alone. Mean of the output = light in both cases. emccd_jax: sensitivity 4.4 e-/ADU only sets the ADC
    # rounding (variance (4.4/(QE EM))^2/12 ~ 5e-5 photons^2, negligible); the read noise is given here in photons
    # (sigma_r) and converted to electrons at the EM output (sigma_r*QE*EM, 73.6 e- for 0.4 photons)
    # Purpose: expected image of the PSF (no background, no noise) -> noisy image as the camera data after the
    # reconstruction. Poisson camera = perfect detector, no QE (kept on purpose, user 2026-11-12).
    import jax, jax.numpy as jnp
    from simu_PSF_polarMFM_JAX import noise_jax, emccd_jax
    key = jax.random.PRNGKey(int(rng.integers(2**31 - 1))) # JAX key from the generator of the run: same seed, same noise
    expected = jnp.asarray(expected, dtype=jnp.float32)
    if a.camera == 'poisson':
        return np.asarray(noise_jax(key, expected, b=a.background, sigma_b=a.sigma_b, sigma_r=a.sigma_r), dtype=float)
    # sigma_r is given in photons: read noise in electrons at the output = sigma_r*QE*EM
    return np.asarray(emccd_jax(key, expected, QE=a.QE, EM=a.EM, b=a.background, sigma_b=a.sigma_b,
                                sigma_read_e=a.sigma_r*a.QE*a.EM, sensitivity=4.4, cic=a.cic), dtype=float)


def crazy_pixels(rng, data, a):
    # extra photons on a random fraction of the pixels, exponential law of mean crazy_amplitude, added AFTER the camera
    # noise (no Poisson / EM noise of their own): irregular background of dense samples (preset actin_dense only)
    if a.crazy_fraction <= 0:
        return data
    hit = rng.random(data.shape) < a.crazy_fraction
    return data + hit*rng.exponential(a.crazy_amplitude, data.shape)


# ---------------------------------------------------------------- pipeline of SGD_jax.py

def build_pipeline(overrides, out_folder, loss, save_figures, a):
    """executes the cells of SGD_jax.py up to the main loop; returns the namespace and the text of the main loop

    Purpose: prepare the real fit pipeline (SGD_jax.py, or SGD_jax_CNN.py with --cnn) as it starts on real data, and
    stop just before the loop that fits the batches. ns ("namespace") is a dictionary that plays the role of the
    memory of a Python session: each cell is executed with exec(cell, ns) and everything it creates (variables,
    functions: ns['NPSF'], ns['d_'], ns['compute_M_jax'], ns['PSF_jax'], ns['noise_model'], ...) goes into ns. The
    simulation then builds its data with the same objects (model_psf), and replaces ns['load_batch'] before the
    main loop is run inside ns. A new ns for each loss.
    The cells are found by the text of their header (review item 9). The source is read when the run starts:
    editing SGD_jax.py during a queue changes the later runs."""
    source = SGD_batch.read_script(SGD_SCRIPT)
    # each top-level 'name = ...' of the script that is in overrides becomes 'name = value' (ideal calibration,
    # loss2, middle_plane, --set); a name absent from the script stops with an error (no silent typo)
    source = SGD_batch.apply_overrides(source, overrides)
    cells = SGD_batch.split_cells(source) # cut at the '#%%' markers: (first line, header, text) of each cell
    # review item 9: the cells are found by the text of their header; each must be found exactly once, else a renamed
    # header of SGD_jax*.py would change the simulation silently
    for key in ('PARAMETERS TO BE DEFINED', 'extracting positions', 'gradient descent'):
        n = sum(key in header for _, header, _ in cells)
        if n != 1:
            raise RuntimeError(f'{SGD_SCRIPT.name}: {n} cells with "{key}" in their header (1 expected): SGD_precision.py must be adapted')
    for key in ('################# main loop', "    result['buffer'] = buffer", "    result['last_frame_processed']"):
        if source.count(key) != 1:
            raise RuntimeError(f'{SGD_SCRIPT.name}: "{key.strip()}" found {source.count(key)} times (1 expected): SGD_precision.py must be adapted')
    ns = {'__name__': '__main__', '__file__': str(SGD_SCRIPT)} # empty memory, filled cell by cell
    dialog = SGD_batch.FakeDialog(out_folder) # answers the folder dialogs of the pipeline with the run folder
    show_factory = lambda n: SGD_batch.make_show(n, save_figures) # figures of the pipeline not shown (saved with --save-figures)
    main_loop = None
    for first_line, header, text in cells:
        if 'OPTIONAL' in header: # reloading the parameters of an old run / resuming a stopped run: not used
            continue
        if 'extracting positions' in header:
            # no localisation file: interplane of the simulation, and the variance without light of the channels
            # (dark_variance.txt of a reconstruction): the read noise of the simulated camera, sigma_r^2 (ADU^2).
            # The pipeline converts it back to photons^2 (x (sensitivity/(QE EM))^2): V of the noise model = sigma_r^2
            import jax
            ns['interplane'] = jax.device_put(a.interplane)
            ns['dark_variance_adu'] = np.full(6, a.sigma_r**2)/(ns['sensitivity']/(ns['QE']*ns['EM']))**2
            ns['camera_metadata'] = {} # no raw file: EM of the parameter cell
            continue
        if 'gradient descent' in header:
            # only the definitions of the fit functions (text before the marker) are run now; the main loop is kept
            # as text and run by main() once the simulated data exist and load_batch is replaced
            marker = '################# main loop'
            split = text.index(marker)
            head, loop = text[:split], text[split:]
            SGD_batch.run_cells(SGD_SCRIPT, [(first_line, header, head)], ns, [], dialog, show_factory)
            main_loop = (first_line + head.count('\n'), loop)
            continue
        SGD_batch.run_cells(SGD_SCRIPT, [(first_line, header, text)], ns, [], dialog, show_factory) # all the other cells, unchanged
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
    # end of load_batch of SGD_jax.py: builds the batch (result) from the PSF and their noise.
    # Purpose: the real load_batch (1) reads frames, measures the noise, cuts and rotates the patches, (2) packs the
    # batch into the dictionary result: noisy_psf (100, 3, 2, 13, 13) photons, x, y, Nstart (fixed start of N of
    # SGD_jax.py), background_array (start of the 6 backgrounds = level of the noise model), noise (gain, shift of the
    # 6 channels), frame, buffer / psf_buffer (leftovers), last_frame_processed. Part (2) is copied here from the text
    # of the pipeline itself (between the two markers below), so the batch is packed exactly as for real data;
    # simulated_load_batch replaces part (1)
    start = source.index("    result['buffer'] = buffer")
    end = source.index('\n', source.index("    result['last_frame_processed']", start))
    return compile(textwrap.dedent(source[start:end]), str(SGD_SCRIPT), 'exec')


def simulated_load_batch(ns, sim, builder, rng_noise, a):
    # Purpose: returns a replacement of load_batch (same inputs), installed as ns['load_batch'], that does part (1)
    # of the real load_batch (see batch_builder) with the simulation, then the copied part (2).
    # replaces load_batch: same signature, the PSF of the next batch are taken in the simulation; the noise model is
    # measured by noise_model of SGD_jax.py on Nframe simulated background frames, as on a chunk of real frames
    # (new frames for every batch: the measured gain fluctuates by ~1 % from batch to batch). The simulated
    # background is uniform, so all pixels have the same level: noise_model gives (variance - V)/level with V fixed
    # at sigma_r^2, correct for b >= 5 (review 2026-11-12: Poisson 1.001 at b 25, 1.023 at b 5, 1.06 at b 2 because
    # of the dither; EMCCD 2.149 vs 2/QE = 2.174 at b 25). b = 0: true noise instead (below).
    # x = y = 0: each simulated patch is centred on 0, the true x, y is the offset of the emitter from the centre.
    # The PSF index is saved as 'frame' (collect() matches found and true values with it)
    # Not as on real data: the noise is measured on clean uniform background frames (real data: the frames
    # themselves, with molecules and an uneven background); no pre-localisation (every simulated PSF is fitted);
    # n_skip (resuming a stopped run) ignored.
    counter = [0] # number of the next batch (a list so that the inner function can change it)
    def load_batch(last_frame_processed, buffer, psf_buffer, NPSF, result, n_skip=0):
        b = counter[0]; counter[0] += 1
        sl = slice(b*NPSF, (b+1)*NPSF) # PSF b*100 ... (b+1)*100 - 1 of the simulation
        noisy_psf = sim['data'][sl] # already in the orientation of the model (no rotation as in the real load_batch)
        if a.warps is None: # Nframe (20) background frames, 6 channels of 64 x 64, same camera and background as the data
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
        if a.background == 0 and a.sigma_b == 0:
            # 2026-11-12 (review): no background, the background frames hold no light and noise_model cannot measure
            # the gain (slope of the variance vs a level that is 0 everywhere: gain -0.12..-0.16 measured, deviance
            # -8). The noise of the simulation is known: var = gain*s + V with gain 1 (noise_jax, Poisson) or 2/QE
            # (emccd_jax, EM excess factor 2 on QE*s photoelectrons, back in photons), V = sigma_r^2 (read noise in
            # photons; the ADC rounding, (4.4/(QE EM))^2/12 ~ 5e-5, is neglected)
            g = 1. if a.camera == 'poisson' else 2./a.QE
            gain, shift, level = np.full(6, g), np.full(6, a.sigma_r**2/g), np.zeros(6)
            if b == 0:
                log(f'background 0: noise model of the fit = the true noise of the simulation, gain {g:.3f}, V {a.sigma_r**2:g}')
        # (3, 3 planes, 2 pol), planes reversed as in load_batch (no effect here: the 6 channels have the same noise)
        chunk_noise = np.stack((gain, shift, level))[:, np.arange(6).reshape(3, 2)[::-1]] # as in load_batch
        psf_noise = np.tile(chunk_noise, (NPSF, 1, 1, 1)) # the same noise model for the 100 PSF of the batch
        x = np.zeros(NPSF); y = np.zeros(NPSF) # the true position is the offset in um, the patch is centred on 0
        index_frame = np.arange(sl.start, sl.stop).astype(float) # PSF index, saved as 'frame'
        last_frame_processed = last_frame_processed + 1 # counter only (no frame file)
        # part (2) of the real load_batch, run with these local values: fills result
        local = dict(result=result, buffer=buffer, psf_buffer=psf_buffer, noisy_psf=noisy_psf, x=x, y=y, NPSF=NPSF,
                     psf_noise=psf_noise, index_frame=index_frame, last_frame_processed=last_frame_processed)
        exec(builder, ns, local)
    return load_batch


# ---------------------------------------------------------------- simulation

def grid(n_rho, n_eta, rho_values=None, eta_values=None):
    # Purpose: list of the simulated orientations. Regular grid: rho and eta at the centres of n_rho and n_eta bins of
    # [0, 180[ (20 x 20: 4.5, 13.5 ... 175.5 deg), every (rho, eta) pair (400 points), or the lists of values given.
    # Returns rho and eta of each point (flat arrays)
    rho_bins = np.array(rho_values, float) if rho_values else (np.arange(n_rho) + 0.5)*180/n_rho
    eta_bins = np.array(eta_values, float) if eta_values else (np.arange(n_eta) + 0.5)*180/n_eta
    rho, eta = np.meshgrid(rho_bins, eta_bins, indexing='ij')
    return rho.ravel(), eta.ravel()


def model_psf(ns, sl, p, half=6, fields=None, zernike=None):
    # expected photons of the 6 channels ((2 half + 1)^2 patch, 13 x 13 by default) for the parameters p[sl], model of
    # SGD_jax.py; fields: pupil fields replacing the ones of ns (data with another sample index)
    # Purpose: the noiseless image of each molecule, with the pipeline's own functions taken from ns, so that data and
    # fit use exactly the same model (optics, plane positions d_, calibration J, aberrations, channel gains).
    # N = photons of the whole molecule in the 6 channels: PSF_jax normalises the full image (3 planes x 2 pol) to N
    # whatever the orientation and z, so N is the light reaching the camera, not the emitted light (an axial dipole
    # gives as many photons as an in-plane one): the total brightness carries no orientation information, in the fit
    # and in the CRB alike (modelling choice, review 2026-11-12). The 13 x 13 crop holds a bit less than N.
    import jax.numpy as jnp
    f32 = lambda v: jnp.array(v[sl], dtype=jnp.float32) # the PSF of the batch, float32 for JAX
    F = {k: ns[k] for k in ('xx', 'yy', 'th1', 'phi', 'Ex0', 'Ex1', 'Ex2', 'Ey0', 'Ey1', 'Ey2')} # pupil fields
    F.update(fields or {})
    # basis images M of the 3 planes x 2 polarisations at the x, y, z of each PSF (x, y, z in um)
    M = ns['compute_M_jax'](xp=f32(p['x']), yp=f32(p['y']), zp=f32(p['z']), d=ns['d_'], x=F['xx'], y=F['yy'],
                            th1=F['th1'], phi=F['phi'], Ex0=F['Ex0'], Ex1=F['Ex1'], Ex2=F['Ex2'], Ey0=F['Ey0'],
                            Ey1=F['Ey1'], Ey2=F['Ey2'], u=ns['u'], v=ns['v'], phase_maskx=ns['phase_mask'],
                            phase_masky=ns['phase_mask'], zernike_base=ns['zernike_base'],
                            zernike_coefs_x=ns['zernike_coefs_x'] if zernike is None else zernike[0],
                            zernike_coefs_y=ns['zernike_coefs_y'] if zernike is None else zernike[1],
                            second_plane=ns['second_plane'], polar_projections=ns['polar_projections'],
                            lambd=ns['lambd'], f_tube=ns['f_tube'])
    # orientation (rho, eta) and wobbling cone delta (eigenvalues) combined with M, normalised to N
    h = ns['PSF_jax'](rho=f32(p['rho']), eta=f32(p['eta']), delta=f32(p['delta']), M=M, N_photons=f32(p['N']))
    dim = int(ns['dim_simu']) # centre pixel of the computed image
    # x y gains of the channels (y_gain_J1/J2, 1 here), then the (2 half + 1)^2 crop around the centre
    return np.array(h * ns['channel_gain']()[None, :, :, None, None])[:, :, :, dim-half:dim+half+1, dim-half:dim+half+1]


# ---------------------------------------------------------------- warp of the reconstruction
# Only with --warp (reviewed 2026-11-12, see warped_data; not used in the runs of 2026-11-12). In real data each
# of the 6 channels is cut from the camera image and resampled by the reconstruction (reconstruction_juliette.py:
# perspective transform M_i, cubic interpolation) to register it on the reference: slight blur and smoothing of the
# noise (correlated neighbouring pixels) that the fit does not model. This block reproduces it on the simulated raw
# patches. The reference channel (plane 1, x) has the identity transform: no interpolation.

WARP_CANAL = [[2, 5], [1, 4], [0, 3]] # channel of subcanals (reconstruction_juliette.py) of each (plane, polarisation)
MARGIN = 2 # pixels around the patch for the cubic kernel


def load_warps(path):
    # Purpose: reads the calibration of the reconstruction: inverse transform of each of the 6 channels and size
    # (rows, columns) of a channel
    cal = np.load(path, allow_pickle=True)
    X, Y = cal['X'], cal['Y']
    return dict(inv=[np.linalg.inv(cal[f'M_{i}']) for i in range(6)], size=(int(Y[1] - Y[0]), int(X[3] - X[2])))


def warp_maps(rng, warps, canal, n, side):
    # maps of cv2.remap reproducing cv2.warpPerspective(INTER_CUBIC) of the reconstruction on side x side regions at n
    # random places of the channel: dst pixel -> src = M^-1 dst (same kernel, same sub-pixel position per pixel). The
    # integer part of the displacement at the centre is removed; shift = remaining (row, col) displacement at the
    # centre: the content at src p appears at dst p - shift
    # Purpose: what the reconstruction does to a small region of channel `canal`, written as cv2.remap maps on a local
    # side x side patch. warpPerspective(src, M) computes dst(X, Y) = src(M^-1 (X, Y)) (cubic interpolation), so each
    # output pixel reads the raw image at a non-integer place. The region is drawn at a random place of the channel:
    # each PSF gets the sub-pixel phase of its own place (uniform, as molecules anywhere in the field).
    # Returns map_x, map_y (n, side, side): local raw coordinates read by each output pixel, and shift (n, 2): the
    # sub-pixel (row, col) displacement at the centre, in -0.5..0.5 px (the integer part only moves the region and is
    # irrelevant, the patch is cut around the molecule anyway). Reference channel (canal 1, M = identity): maps =
    # integer pixels, shift 0, the cubic kernel then returns the raw pixels unchanged.
    H, W = warps['size']
    r0 = rng.integers(0, H - side, n)[:, None, None]; c0 = rng.integers(0, W - side, n)[:, None, None] # random region
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing='ij') # local (row, col) of the patch
    X, Y = c0 + jj, r0 + ii # output (dst) pixels in channel coordinates
    m = warps['inv'][canal]
    den = m[2, 0]*X + m[2, 1]*Y + m[2, 2] # perspective division
    dx = (m[0, 0]*X + m[0, 1]*Y + m[0, 2])/den - X; dy = (m[1, 0]*X + m[1, 1]*Y + m[1, 2])/den - Y # src - dst (px)
    c = side//2
    ox, oy = np.round(dx[:, c, c])[:, None, None], np.round(dy[:, c, c])[:, None, None] # integer part at the centre
    # raw local pixel (u, v) = channel pixel (r0 + oy + u, c0 + ox + v): the output (ii, jj) reads raw (ii + dy - oy, jj + dx - ox)
    map_x = (jj + dx - ox).astype(np.float32); map_y = (ii + dy - oy).astype(np.float32)
    return map_x, map_y, np.stack([(dy - oy)[:, c, c], (dx - ox)[:, c, c]], axis=1)


def remap(images, map_x, map_y):
    # Purpose: applies the maps of warp_maps to each image with the cubic interpolation of OpenCV (as the reconstruction)
    import cv2
    return np.stack([cv2.remap(np.ascontiguousarray(im, dtype=np.float32), mx, my, cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101) for im, mx, my in zip(images, map_x, map_y)])


def pixels_per_um(ns, a):
    # displacement (row, col) of the PSF in each channel of the patch for 1 um of x and of y (centroids)
    # Purpose: converts the sub-pixel shift of the warp (pixels) into a shift of the emitter (um), see warped_data.
    # Measured on the model itself (noise-free PSF of a dipole along z, eta 0, at z 0.6 um, 21 x 21 px): centroid of
    # the reference PSF and of the PSF moved by 0.06 um in x, then in y -> A[plane, pol] = 2 x 2 matrix
    # (row, col) per um of (x, y), ~ 1/0.12 = 8.3 px/um with the signs / swaps of each channel of the model.
    # The centroid moves exactly with x, y (translation of the image), so the orientation chosen does not matter.
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
    # Purpose: data of one batch with --warp (replaces data_psf + camera in simulate). The noise is added before the
    # interpolation, as on the camera.
    # Steps for each of the 6 channels (plane, pol of the model):
    #  1. warp_maps: a random region of the camera channel WARP_CANAL[plane][pol] -> maps + sub-pixel shift (px);
    #  2. the warp moves the content by -shift, so the raw PSF is drawn at truth + shift: move = A^-1 shift (um), with
    #     A from pixels_per_um; after the warp the PSF is back at the true x, y (linear at the centre of the patch, the
    #     perspective varies by ~1e-3 px per px: negligible over 13 px);
    #  3. expected raw patch 17 x 17 (13 + 2 MARGIN) with model_psf (+ the second emitter, + the y gain of the data);
    #  4. camera noise + background (camera) and crazy pixels on the raw patch, as on the camera;
    #  5. cubic remap (blur + correlated noise), then the central 13 x 13 crop. MARGIN 2 = reach of the cubic kernel
    #     (pixels floor(u) - 1 ... floor(u) + 2) for |shift| <= 0.5: the border reflection is never read.
    # Same physics as real data: the reconstruction warps the raw channels (baseline removed, as camera() returns).
    # Not reproduced: the division by the lamp normalisation of each channel (normalisation[i] of the reconstruction,
    # a gain per channel), and the row flip of load_batch (the warp is applied to the patch in the orientation of
    # the model, a mirrored sub-pixel pattern: same statistics, shift and A are taken in the same orientation).
    # Uses model_psf, not data_psf: --data-zernike and the pupil fields of --data-n2 are ignored with --warp
    # (review item 13); --data-shift / --data-zshift are refused in parse_arguments.
    n = sl.stop - sl.start; side = 13 + 2*MARGIN
    out = np.zeros((n, 3, 2, 13, 13))
    for plane in range(3):
        for pol in range(2):
            # canal of the camera: planes reversed in the model (plane 0 = canals 2/5 of the reconstruction)
            map_x, map_y, shift = warp_maps(rng_warp, a.warps, WARP_CANAL[plane][pol], n, side)
            move = np.linalg.solve(A[plane, pol], shift.T).T # um of (x, y) that move the content by shift
            # copy of the truth with x, y moved for this channel only (the other channels keep the truth)
            def placed(p):
                q = dict(p); q['x'] = p['x'].copy(); q['y'] = p['y'].copy()
                q['x'][sl] += move[:, 0]; q['y'][sl] += move[:, 1]
                return q
            # all 6 channels are computed, only this one is kept (6 x the model time: --warp runs are slower)
            h = model_psf(ns, sl, placed(truth), half=6 + MARGIN)[:, plane, pol]
            if neighbour[sl].any(): # second emitter moved by the same amount (same warp region)
                h = h + neighbour[sl][:, None, None]*model_psf(ns, sl, placed(second), half=6 + MARGIN)[:, plane, pol]
            if (plane, pol) == (1, 1):
                h = h*a.data_y_gain_J2
            raw = crazy_pixels(rng, camera(rng, h, a), a)
            out[:, plane, pol] = remap(raw, map_x, map_y)[:, MARGIN:-MARGIN, MARGIN:-MARGIN]
    return out


class data_physics:
    ''' context in which model_psf computes the data with the mismatch of --data-n2 / --data-saf-dz: the globals of
    simu_PSF_polarMFM_JAX.py (n2, psi_z_jit) are changed and the compiled functions cleared, then restored (and
    cleared again) for the fit. self.fields: pupil fields of the data, recomputed with the sample index of the data
    Purpose: only for the mismatch options --data-n2 / --data-saf-dz (data with another sample than the fit); does
    nothing otherwise. Used as `with data_physics(ns, a) as phys:` in simulate: changes on entering, restores on
    leaving, so that the fit sees the normal model (reviewed 2026-11-12) '''
    def __init__(self, ns, a):
        self.ns, self.a, self.fields = ns, a, None
        self.active = a.data_n2 is not None or a.data_saf_dz != 0 # does nothing without these options
    def __enter__(self):
        if not self.active:
            return self
        import jax, jax.numpy as jnp
        import simu_PSF_polarMFM_JAX as SP
        ns, a = self.ns, self.a
        self.saved = (SP.n2, SP.psi_z_jit) # sample index and axial phase of the model, put back by __exit__
        if a.data_n2 is not None:
            SP.n2 = a.data_n2
        if a.data_saf_dz:
            # axial phase k n2 sqrt(1 - (n1 sin(theta)/n2)^2) z: the sqrt is imaginary beyond the critical angle
            # (SAF), exp(i k sqrt z) is then a decay. dz is added to the decay only: the evanescent light decays as for
            # an emitter dz further from the glass, the propagating light is unchanged. cos_th2 is ignored here, as
            # the model never passes it (None)
            dz = a.data_saf_dz
            def psi_z_saf(theta, z, lambd=0.617, cos_th2=None):
                sq = jnp.sqrt(1 - (SP.n1*jnp.sin(theta)/SP.n2)**2) # imaginary in the SAF region: exp(-k kappa z)
                factor = 2*jnp.pi*SP.n2/lambd
                out = jnp.einsum('a, bc -> abc', z, factor*sq)
                return out + dz*factor*1j*jnp.imag(sq)[None] # evanescent part only, as at z + dz
            SP.psi_z_jit = jax.jit(psi_z_saf)
        jax.clear_caches() # the compiled model functions keep the old values: recompiled with the new ones
        if a.data_n2 is not None: # new sample index: pupil fields (Fresnel coefficients) of the data recomputed
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
            jax.clear_caches() # the fit is compiled again with the normal model
        return False


def data_psf(ns, sl, p, a, phys, half=6):
    # model_psf for the data: mismatch of the sample (phys.fields), aberrations (--data-zernike) and lateral shift of
    # channels (--data-shift, --data-zshift)
    # Purpose: expected images of the DATA. Without any --data-* option it is exactly model_psf (the fit's model)
    zern = None
    if a.data_zernike: # aberrations of the data only (re-read for every batch: small waste, not a bug)
        import jax.numpy as jnp
        zf = np.load(a.data_zernike)
        zern = (jnp.array(zf['zx'], dtype=jnp.complex64), jnp.array(zf['zy'], dtype=jnp.complex64))
    h = model_psf(ns, sl, p, half, phys.fields, zern)
    # --data-shift: the image of one channel (pol 2 = both) recomputed with the emitter moved by dx, dy nm replaces
    # that channel (registration error of the channel)
    for plane, pol, dx, dy in a.data_shift:
        q = dict(p); q['x'] = p['x'] + dx/1000; q['y'] = p['y'] + dy/1000
        hs = model_psf(ns, sl, q, half, phys.fields, zern)
        pols = (0, 1) if int(pol) == 2 else (int(pol),)
        for k in pols:
            h[:, int(plane), k] = hs[:, int(plane), k]
    # --data-zshift: same with a shift of each channel that depends on the z of the emitter (table interpolated; end
    # values outside the table), added to the --data-shift of the channel if any (review item 11, fixed 2026-11-12:
    # was recomputed from the original position, the fixed shift was lost)
    fixed = {}
    for plane, pol, dx, dy in a.data_shift:
        for k in ((0, 1) if int(pol) == 2 else (int(pol),)):
            fixed[int(plane), k] = (dx/1000, dy/1000)
    if a.data_zshift:
        table = np.load(a.data_zshift)
        for plane in range(3):
            for pol in range(2):
                sh = table['shift'][plane, pol]
                if not np.any(sh):
                    continue
                q = dict(p); q['x'] = p['x'].copy(); q['y'] = p['y'].copy()
                fx, fy = fixed.get((plane, pol), (0., 0.))
                q['x'][sl] = p['x'][sl] + fx + np.interp(p['z'][sl], table['z_em'], sh[:, 0])/1000
                q['y'][sl] = p['y'][sl] + fy + np.interp(p['z'][sl], table['z_em'], sh[:, 1])/1000
                h[:, plane, pol] = model_psf(ns, sl, q, half, phys.fields, zern)[:, plane, pol]
    return h


def simulate(ns, a, rng):
    # true parameters and noisy data of all the PSF. Draws from rng in a fixed order (x, y, z, second emitter, then the
    # camera noise batch by batch): the same seed gives the same truth in every run (any noise type, SGD or CNN);
    # delta of --delta-range from its own generator (seed + 3), so x, y, z are those of a run with one delta.
    # rho, eta: grid point of each PSF (per_point consecutive PSF per point); N: same for all (molecule, 3 planes);
    # x, y uniform +-xy_range um around the patch centre, z uniform in z_range um above the coverslip.
    # 20 x 20 x 80 = 32000 = 320 whole batches: no repeated PSF
    # Purpose: the truth of every PSF, then its noisy image (data_psf -> second emitter -> y gain of the data ->
    # camera -> crazy pixels), batch by batch. Returns everything saved in simulation.npz
    NPSF = int(ns['NPSF'])
    rho_list, eta_list = grid(a.n_rho, a.n_eta, a.rho_values, a.eta_values)
    point = np.repeat(np.arange(len(rho_list)), a.per_point) # grid point of each PSF: 0,0,...(80x),1,1,...
    n_total = int(np.ceil(len(point)/NPSF))*NPSF # whole batches: the last PSF are repeated points
    point = np.concatenate([point, point[:n_total-len(point)]])
    truth = dict(rho=rho_list[point], eta=eta_list[point], delta=np.full(n_total, a.delta),
                 N=np.full(n_total, a.N), x=rng.uniform(-a.xy_range, a.xy_range, n_total),
                 y=rng.uniform(-a.xy_range, a.xy_range, n_total), z=rng.uniform(*a.z_range, n_total))
    # with --delta-range each grid point (rho, eta) holds 80 PSF of different delta (60-120 in the runs of 2026-11-12):
    # the bias and precision of a point are averages over delta (the precision degrades with delta)
    if a.delta_range: # own generator: the other draws (x, y, z, noise) are those of a run with one delta
        truth['delta'] = np.random.default_rng(a.seed + 3).uniform(*a.delta_range, n_total)
    # second emitter (preset actin_dense only, two_emitter_fraction > 0): rho uniform, eta = arccos(uniform(-1, 1))
    # (uniform over the sphere), N x uniform(neighbour_N), own z, at neighbour_distance pixels (x 0.12 um) of the
    # first in a random direction. It is not fitted: it is a perturbation of the patch
    # second emitter: random orientation over the half-sphere, N, z, at neighbour_distance pixels in a random direction
    # (drawn even when two_emitter_fraction = 0, so that the order of the draws, hence the noise, never changes)
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
            h = data_psf(ns, sl, truth, a, phys) # expected images (100, 3, 2, 13, 13), photons, no background
            if neighbour[sl].any():
                h = h + neighbour[sl][:, None, None, None, None]*data_psf(ns, sl, second, a, phys)
            h[:, 1, 1] *= a.data_y_gain_J2 # y of plane 1 (1: no change)
            data.append(crazy_pixels(rng, camera(rng, h, a), a)) # background + noise, then crazy pixels (off: no change)
    return dict(data=np.concatenate(data).astype(np.float32), point=point, rho_list=rho_list, eta_list=eta_list,
                neighbour=neighbour, **{'true_'+k: v for k, v in truth.items()}, **{'second_'+k: v for k, v in second.items()})


# ---------------------------------------------------------------- errors and statistics

def wrap(v, period=180):
    # Purpose: angle difference brought back into [-period/2, period/2[ (e.g. 175 - 5 = 170 -> -10)
    return (v + period/2) % period - period/2


def collect(npz_folder, sim):
    # found values of every PSF (npz of the pipeline) matched to the truth by the PSF index ('frame'), and errors:
    # rho: wrapped to [-90, 90[; when the wrap moves rho by 180, the found orientation is written (rho+180, 180-eta),
    # so eta is flipped before its error (assumes found and true rho in [0, 180], checked 2026-11-12: always, the
    # pipelines fold rho, eta to [0, 180]; ~8.5 % of the PSF cross, those near rho 0/180); eta error wrapped too.
    # x, y in nm (saved x = 1000 x/correction_dft, as SGD_jax.py writes the csv), z in nm, N relative to the truth.
    # Diverged fits (nan) are kept here and left out of the statistics
    # Purpose: found vs true, PSF by PSF. One fit per simulated PSF (no pre-localisation), so no matching is needed:
    # 'frame' is the PSF index set by simulated_load_batch.
    # Example of the eta flip: truth (175, 60), fit (2, 120) = (182, 60) folded: raw rho -173 -> wrapped +7, crossed,
    # eta 180 - 120 = 60, error 0
    files = sorted([f for f in Path(npz_folder).glob('*.npz') if f.stem.isdigit()], key=lambda f: int(f.stem)) # batch npz, in order
    found = {k: np.concatenate([np.load(f)[k] for f in files]) for k in ('frame', 'x', 'y', 'z', 'rho', 'eta', 'delta', 'N_photons', 'deviance')}
    idx = found['frame'].astype(int) # PSF index of each fit
    t = {k: sim['true_'+k][idx] for k in ('rho', 'eta', 'delta', 'N', 'x', 'y', 'z')} # truth of each fit
    # SGD_jax.py and SGD_jax_CNN.py save x = int(x_preloc/120)*120 + 1000*x_found/correction_dft (nm); x_preloc = 0 here
    # (patch centred on 0) -> x_found (um) = saved*correction_dft/1000. Checked 2026-11-12 on nobg_poisson_cnn and
    # nobg_emccd_cnn: 32000 fits, 32000 different PSF, no nan, median x, y error 0.00 nm
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
    # Purpose: std that ignores the outliers: 1.4826 x median absolute deviation (= std for a Gaussian)
    return 1.4826*np.median(np.abs(v - np.median(v)))


def statistics(idx, err, sim, subset=None):
    # bias and precision per orientation point, on all the PSF or on a subset (boolean per found PSF):
    # bias = median of the error, precision = robust std (1.4826 x MAD) of the ~80 PSF of the point: both ignore the
    # outliers (e.g. fits stuck at the delta bound). NOT the std: the CRB comparison (crb_vs_unbiased_sim.py) uses
    # the std, see review item 6
    # Purpose: one row per grid point (summary.csv, maps). Statistical noise per point with 80 PSF (review item 10):
    # median +-1.25 sigma/sqrt(80) = +-0.14 sigma (delta, sigma ~20 deg: +-2.8 deg; eta, sigma ~9: +-1.3 deg); robust
    # std +-13 % (plain std +-8 %). Point-to-point patterns of that size in the maps are noise. rho near the axis
    # (eta 4.5, 175.5) is almost undefined: its large precision there is expected
    g = sim['point'][idx] # grid point of each fit
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
            # statistical noise of this point (review item 10, Gaussian errors): standard error of the median
            # 1.2533 sigma/sqrt(n), relative standard error of the robust std ~1.17/sqrt(n) (13 % for 80 PSF)
            row['bias_err_'+k] = 1.2533*row['sigma_'+k]/np.sqrt(ok.sum()) if ok.any() else np.nan
            row['sigma_relerr_'+k] = 1.17/np.sqrt(ok.sum()) if ok.any() else np.nan
        rows.append(row)
    return rows


# ---------------------------------------------------------------- figures

def disc_map(ax, rows, values, cmap, vmin, vmax):
    # Purpose: draw one value per grid point (rows = the grid points of statistics(), values = one number each) on a
    # polar disc = the orientation hemisphere seen from above the coverslip. No interpolation: one coloured cell per
    # simulated (rho, eta).
    # Polar plot: angle rho, radius eta <= 90 (centre = dipole along the optical axis z, edge = in the plane).
    # (rho, eta) and (rho+180, 180-eta) are the same dipole (vector -u), so the bins eta > 90 are drawn at rho+180,
    # radius 180-eta (as in SGD_simu_precision2.ipynb, removed): the half 0-180 deg of the disc holds eta < 90, the half
    # 180-360 holds eta > 90.
    # eta BIAS: in the coordinates of the truth a positive eta error moves the dipole towards the centre of the disc
    # on the half 180-360 (eta > 90) and towards the edge on the half 0-180; the callers flip it on the half eta > 90
    # with eta_bias_on_disc (review item 12): on the maps + = towards the edge everywhere. rho bias: same on both halves.
    # Needs a regular grid with rho bins covering 0-180 evenly and an even number of eta bins symmetric about 90.
    rho_bins = np.unique([r['rho'] for r in rows]); eta_bins = np.unique([r['eta'] for r in rows])
    n_rho, n_eta = len(rho_bins), len(eta_bins)
    if n_eta % 2:
        raise ValueError('n_eta must be even for the polar maps (bins split at eta = 90)')
    table = np.full((n_rho, n_eta), np.nan) # table[rho bin, eta bin]; NaN = grid point without statistics (white)
    for r, v in zip(rows, values):
        table[np.searchsorted(rho_bins, r['rho']), np.searchsorted(eta_bins, r['eta'])] = v
    half = n_eta//2
    # (radius bin, angle bin): angles 0-180 eta < 90, angles 180-360 eta > 90 with radius 180 - eta
    # (eta bins of the second half reversed: 175.5 -> radius 4.5 first, 94.5 -> radius 85.5 last)
    shown = np.concatenate([table[:, :half].T, table[:, half:][:, ::-1].T], axis=1)
    # cell edges: 2 n_rho angular cells of 180/n_rho deg over 0-360, half radial cells over 0-90
    th_edges = np.deg2rad(np.linspace(0, 360, 2*n_rho + 1)); r_edges = np.linspace(0, 90, half + 1)
    im = ax.pcolormesh(th_edges, r_edges, shown, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    ax.set_rlim(0, 90); ax.set_rticks([30, 60]); ax.set_yticklabels([]); ax.grid(color='k', alpha=0.25, lw=0.5)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(['0', '90', '180', '270'], fontsize=7)
    ax.tick_params(pad=0)
    return im


def eta_bias_on_disc(rows, values):
    # Purpose (review item 12): on the polar disc the grid points eta > 90 are drawn at (rho + 180, 180 - eta), where a
    # positive eta error moves the dipole towards the CENTRE of the disc, while for eta < 90 it moves it towards the
    # EDGE. The sign of the eta bias of the points eta > 90 is flipped for the maps, so that everywhere on the disc
    # positive = towards the edge (more in-plane), negative = towards the centre (towards the optical axis).
    # summary.csv keeps the bias in the coordinates of the truth.
    return np.array([-v if r['eta'] > 90 else v for r, v in zip(rows, values)], dtype=float)


def map_noise(rows, k, kind, scale=1):
    # Purpose (review item 10): statistical noise of one grid point, for the titles of the maps
    if kind == 'bias':
        return f'noise per point +-{scale*np.nanmedian([r.get("bias_err_"+k, np.nan) for r in rows]):.2g}'
    return f'noise per point +-{100*np.nanmedian([r.get("sigma_relerr_"+k, np.nan) for r in rows]):.0f} %'


def case_figure(all_rows, folder, case):
    # Purpose: the main figure of a run, polar_summary_<case>.png. all_rows = {loss: rows of statistics()}.
    # Top block: bias (median of the error) of rho, eta, delta, N; bottom block: precision (robust std); one row of
    # discs per loss (1 row in the runs of 2026-11-12, --losses huber), one column per parameter.
    # Colour scales, shared by the losses of a block: bias symmetric +-(95th percentile of |bias|), precision
    # 0 - (95th percentile); the 5 % most extreme grid points saturate.
    # Panel titles: median over the grid of |bias| (bias block) or of the precision. Note: median |bias| has a noise
    # floor of about 0.1 sigma even for an unbiased fit (80 PSF per point, review item 10).
    import matplotlib.pyplot as plt
    params = [('rho', 'deg', 1), ('eta', 'deg', 1), ('delta', 'deg', 1), ('N', '%', 100)] # N error is relative -> %
    losses = list(all_rows)
    blocks = (('bias', 'bias (median of the error)', 'coolwarm'), ('sigma', 'precision (robust std)', 'viridis'))
    fig = plt.figure(figsize=(3.2*len(params), 2*3.0*len(losses) + 1.6), layout='constrained')
    fig.suptitle(f'{case}: bias and precision, one bin per simulated (rho, eta) of the grid; eta bias: + = towards the edge of the disc (more in-plane)\n'
                 f'angle = rho, radius = eta (circles at 30, 60 deg; eta > 90 shown at rho+180, 180-eta), same color scale for the 3 losses within a block', fontsize=12)
    for subfig, (kind, title, cmap) in zip(fig.subfigures(2, 1, hspace=0.04), blocks):
        subfig.suptitle(title, fontsize=12, fontweight='bold')
        axs = np.atleast_2d(subfig.subplots(len(losses), len(params), subplot_kw=dict(projection='polar')))
        for c, (k, unit, scale) in enumerate(params):
            values = {l: scale*np.array([r[f'{kind}_{k}'] for r in all_rows[l]], dtype=float) for l in losses}
            if (kind, k) == ('bias', 'eta'):
                values = {l: eta_bias_on_disc(all_rows[l], v) for l, v in values.items()}
            allv = np.concatenate(list(values.values())); allv = allv[np.isfinite(allv)]
            if kind == 'bias':
                lim = np.percentile(np.abs(allv), 95) if allv.size else 1; vmin, vmax = -lim, lim
            else:
                vmin, vmax = 0, (np.percentile(allv, 95) if allv.size else 1)
            for li, loss in enumerate(losses):
                ax = axs[li, c]
                im = disc_map(ax, all_rows[loss], values[loss], cmap, vmin, vmax)
                med = np.nanmedian(np.abs(values[loss])) if kind == 'bias' else np.nanmedian(values[loss])
                ax.set_title(f'{k}, {"median |bias|" if kind == "bias" else "median"} {med:.1f} {unit}\n{map_noise(all_rows[loss], k, kind, scale)}', fontsize=8)
                if c == 0:
                    ax.text(-0.18, 0.5, loss, transform=ax.transAxes, rotation=90, ha='center', va='center', fontsize=11, fontweight='bold')
            cb = subfig.colorbar(im, ax=axs[:, c], location='bottom', shrink=0.85, aspect=20)
            cb.set_label(f'{k} {"bias" if kind == "bias" else "precision"} ({unit})', fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(Path(folder) / f'polar_summary_{case}.png', dpi=100); plt.close(fig)


def compare_losses(all_rows, folder):
    # Purpose: compare_losses_vs_eta.png, profiles versus the true eta (one curve per loss): for each eta of the grid,
    # the median over rho of the bias (top row) and of the precision (bottom row) of rho, eta, delta, z, x, N.
    # The median over rho of a signed bias can hide biases of opposite sign at different rho (see the polar maps).
    # Units: N is relative here (0.01 = 1 %), in % on the polar maps.
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
    # Purpose: one complete run. Options -> run folder <out>/<date>_<tag> -> calibration of the model -> for each loss:
    # pipeline of SGD_jax.py or SGD_jax_CNN.py prepared (build_pipeline), data simulated once (first loss), load_batch replaced,
    # main loop of the pipeline run unchanged, results collected -> statistics, maps, summary.
    global SGD_SCRIPT
    a = parse_arguments(argv)
    if a.cnn:
        SGD_SCRIPT = HERE / 'SGD_jax_CNN.py' # CNN first guess + per-PSF LM instead of SGD1/SGD2
    run_folder = Path(a.out) / (f'{datetime.now():%Y-%m-%d_%Hh%M}' + (f'_{a.tag}' if a.tag else ''))
    run_folder.mkdir(parents=True, exist_ok=True)
    log(f'results in {run_folder}')
    log('parameters: ' + ', '.join(f'{k}={v}' for k, v in vars(a).items() if k != 'warps'))
    log(noise_description(a)) # review item 2: the noise in plain words
    extra = SGD_batch.parse_overrides(a.set) # --set name=value: any parameter of the parameter cell of SGD_jax.py / SGD_jax_CNN.py
    # ideal polarisation calibration (J1 identity, J2 = rot(45): x channel of the middle plane = analyser at rho 45 deg,
    # review item 18), no rotation of the channels, y gains 1),
    # same for the data and the fit
    calibration = {'J1': 'np.eye(2)', 'J2': 'rot(45)', 'rotation': '0', 'rotation2': '0', 'rho_offset': '0',
                   'unitary_J': 'False', 'y_gain_J1': '1.', 'y_gain_J2': '1.'}
    if a.calibration: # calibration of a real run (same for the data and the fit), rho of the model (no rho_offset)
        from density_experimental import run_parameters
        real = run_parameters(a.calibration)[0]
        calibration.update({k: v for k, v in real.items() if k in ('J1', 'J2', 'rotation', 'rotation2', 'unitary_J', 'y_gain_J1', 'y_gain_J2')})
        log('calibration of ' + a.calibration + ': ' + ', '.join(f'{k}={calibration[k]}' for k in ('rotation', 'rotation2', 'unitary_J', 'y_gain_J1', 'y_gain_J2')))
    rng = np.random.default_rng(a.seed) # truth and data noise (simulate); seed + 1: background frames; + 2 warp; + 3 delta
    sim = None
    all_rows = {}
    for loss in a.losses:
        start = time.time()
        # overrides of the parameter cell: calibration, loss2 = the loss of this pass, middle_plane = -d (--d -1.0 or
        # --middle-plane 1.0 -> middle_plane 1.0 um), the optics given as options (item 8, same form as the parameter
        # cell), then --set, which wins over everything (item 14, fixed 2026-11-12: --set loss2= / middle_plane= crashed)
        optics = {'lambda_emission': (a.lambda_emission, 'jax.device_put({})'), 'NA': (a.NA, 'jax.device_put(jnp.array({}))'),
                  'mag': (a.mag, 'jax.device_put(jnp.array({}))'), 'MAG': (a.MAG, 'jax.device_put(jnp.array({}))'),
                  'l_pixel': (a.l_pixel, 'jax.device_put(jnp.array({}))'), 'SAF': (a.SAF, '{}')}
        overrides = {**calibration, 'loss2': repr(loss), 'middle_plane': repr(-a.d),
                     **{k: f.format(v) for k, (v, f) in optics.items() if v is not None}, **extra}
        ns, (loop_first_line, loop_text), source = build_pipeline(overrides, run_folder, loss, a.save_figures, a)
        if sim is None: # same simulated data for all the losses
            log('model (data and fit): ' + ', '.join(f'{k} {float(np.asarray(ns[k])):g}' for k in ('lambda_emission', 'NA', 'mag', 'MAG', 'l_pixel'))
                + f', SAF {ns["SAF"]}, middle_plane {float(np.asarray(ns["middle_plane"])):g} um, interplane {float(np.asarray(ns["interplane"])):g} um'
                + ' (values of the parameter cell of ' + SGD_SCRIPT.name + ' unless given as options)')
            log('simulating the data')
            sim = simulate(ns, a, rng)
            np.savez_compressed(run_folder / 'simulation.npz', **sim)
            log(f'{len(sim["data"])} PSF ({a.n_rho} rho x {a.n_eta} eta x {a.per_point}), '
                f'{len(sim["data"])//int(ns["NPSF"])} batches, {int(sim["neighbour"].sum())} with a second emitter')
        ns['batch_nb'] = len(sim['data'])//int(ns['NPSF']) # number of batches the main loop runs (320)
        ns['first_batch'] = 0
        # load_batch replaced; rng of the background frames re-created for each loss: every loss gets the same frames
        ns['load_batch'] = simulated_load_batch(ns, sim, batch_builder(source), np.random.default_rng(a.seed + 1), a)
        log(f'pipeline of SGD_jax.py with loss2 = {loss}')
        # the main loop of SGD_jax.py / SGD_jax_CNN.py (text kept by build_pipeline), executed in ns: fits + one npz per batch in
        # NPZ_<loss>, exactly as for real data
        SGD_batch.run_cells(SGD_SCRIPT, [(loop_first_line, 'main loop', loop_text)], ns, [],
                            SGD_batch.FakeDialog(run_folder), lambda n: SGD_batch.make_show(n, a.save_figures))
        idx, found, truth, err = collect(ns['save_folder'], sim) # found vs truth per PSF -> results_<loss>.npz
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
    # Purpose: one log line, for each parameter 'median over the grid points of |bias| / median of the precision'
    # (angles deg, x y z nm, N relative: 0.03 = 3 %). The |bias| has a noise floor ~0.1 sigma (review item 10)
    return '  '.join(f'{k} {np.nanmedian([abs(r["bias_"+k]) for r in rows]):6.2f}/{np.nanmedian([r["sigma_"+k] for r in rows]):6.2f}'
                     for k in keys)


def loss_statistics(loss, idx, err, sim, run_folder, a):
    # statistics per orientation, polar maps and log of one loss
    # Purpose: statistics() of one loss + log lines: number of diverged fits, median_line; with --delta-range, the
    # straight-line fit found delta = slope x true delta + offset (least squares, all PSF and eta 60-120 only):
    # slope < 1 = shrinkage of delta towards a common value (the fit does not follow delta), offset = global bias.
    # The noise is on the found delta only, so the slope is not diluted by the noise; the fits stuck at the delta
    # bound are included (least squares, not robust). With second emitters: isolated and neighbour PSF separately.
    rows = statistics(idx, err, sim)
    failed = ~np.isfinite(err['eta'])
    log(f'{loss}: {int(failed.sum())} of {len(failed)} fits diverged (nan), left out; median over the orientations of '
        f'|bias| / precision: {median_line(rows)}')
    true_delta = sim['true_delta'][idx]
    if np.ptp(true_delta) > 1: # true delta drawn per PSF: shrinkage of the found delta toward a common value
        found_delta = err['delta'] + true_delta
        eta_true = sim['true_eta'][idx]
        for name, m in (('all', np.isfinite(found_delta)), ('in plane, eta 60-120', np.isfinite(found_delta) & (np.abs(eta_true - 90) <= 30))):
            if m.sum() < 10: continue
            slope, offset = np.polyfit(true_delta[m], found_delta[m], 1)
            log(f'  {loss} delta found vs true ({name}, {int(m.sum())} PSF): slope {slope:.3f} (1 = no shrinkage), '
                f'offset {offset:.1f} deg, rmse {np.sqrt(np.mean(err["delta"][m]**2)):.1f} deg')
    if sim['neighbour'].any(): # isolated PSF and PSF with a second emitter separately
        for name, subset in (('isolated', ~sim['neighbour'][idx]), ('neighbour', sim['neighbour'][idx])):
            log(f'  {loss} {name:9s} ({int(subset.sum())} PSF, {int((subset & failed).sum())} nan): '
                f'{median_line(statistics(idx, err, sim, subset))}')
    return rows


def summary(all_rows, run_folder):
    # Purpose: end of a run (or of recompute): summary.csv (one row per grid point and loss: point, rho, eta, n,
    # n_failed, bias_<k>, sigma_<k>, loss), compare_losses_vs_eta.png, polar_summary_<case>.png (case = folder name
    # without the date), and one log line per loss. log = print with the time: it goes to the <tag>.log of the queue
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
    # Purpose: redo the statistics and figures after a change of statistics / figures, without fitting again:
    # python SGD_precision.py --recompute <run folder> [...]. Overwrites summary.csv and the png; the log
    # lines only go to the console (redirect them to keep them). The CNN-start maps are not made here
    # (simu_precision/polar_maps.py <run folder>)
    folder = Path(folder)
    sim = dict(np.load(folder / 'simulation.npz'))
    a = None # loss_statistics does not use the options
    all_rows = {}
    for loss in ('poisson', 'lms', 'huber'):
        f = folder / f'results_{loss}.npz'
        if f.exists():
            r = np.load(f)
            err = {k[6:]: r[k] for k in r.files if k.startswith('error_')}
            all_rows[loss] = loss_statistics(loss, r['index'], err, sim, folder, a)
    summary(all_rows, folder)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--recompute']: # python SGD_precision.py --recompute <result folder> ...
        for folder in sys.argv[2:]:
            recompute(folder)
    else:
        main(sys.argv[1:])
