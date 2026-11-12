# User manual: precision simulations (simu_precision/SGD_precision.py) and Cramer-Rao bound

Written during the code review of 2026-11-12 (branch simulation-review), function by function, in the order in which
the code runs. Sections marked "(not reviewed yet)" will be completed as the review goes on. Open issues and fixes:
review/simulation_precision_fisher_2026-11-12/simulation_precision_review.md. Exact chain of a real run: review/simulation_precision_fisher_2026-11-12/run_map_simulation_precision.md.

## 1. What the script does

`simu_precision/SGD_precision.py` measures the bias and the precision of the fit on simulated data. It simulates PSF with
the same model as the fit, adds camera noise, and runs the real fit pipeline (`SGD_jax.py`, or `SGD_jax_CNN.py` with
`--cnn`) on them exactly as on real data. Only the reading of the frames is replaced by the simulated PSF. It then
compares the found parameters with the truth, per orientation (rho, eta).

Typical command (unbiased simulation of 2026-11-12, "Poisson" without background, CNN + LM fit):

    python simu_precision/SGD_precision.py --preset poisson --N 2100 --delta-range 60 120 --losses huber \
        --background 0 --sigma-b 0 --out /mnt/d/Amaury/simu_precision_claude/2026-11-09_unbiased_sgd_vs_cnn \
        --tag nobg_poisson_cnn --cnn

Outputs in `<out>/<date>_<tag>/`: `simulation.npz` (data and truth), `NPZ_<loss>/` (npz of the pipeline, as a real
run, and its config.txt), `results_<loss>.npz` (found, true, error per PSF), `summary.csv` (bias and precision per
orientation), polar maps and profiles vs eta, and a log.

## 2. Command-line options: `parse_arguments` (line ~114)

"Parsing" = reading the words of the command line and turning them into named values (`a.N`, `a.background`, ...)
used by the rest of the script. Four steps:
1. the list of accepted options (`p.add_argument`): name, type, number of values, allowed values, default, help;
2. reading the command line (`p.parse_args`): an unknown option or a wrong type stops the script before any
   computation;
3. filling from the preset: every option whose default is `None` and that was not typed takes the value of the
   chosen preset. **Rule: what you type wins, what you do not type comes from the preset;**
4. the run name (`--tag`, built from the preset and the special options if not given) and a consistency check.

### 2.1 Run control
| option | default | meaning |
|---|---|---|
| `--preset` | actin | set of noise and photon values for the options left at None (2.3) |
| `--losses` | poisson lms huber | fit losses, each run on the same simulated data |
| `--cnn` | off | CNN start + per-PSF LM fit (`SGD_jax_CNN.py`) instead of `SGD_jax.py` |
| `--set name=value` | - | changes a parameter of SGD_jax.py (e.g. `huber_c=1.5`) |
| `--seed` | 0 | random seed: same seed = same truth and same noise draws |
| `--out`, `--tag` | | results folder and name |
| `--save-figures` | off | also the diagnostic figures of the pipeline |

### 2.2 The simulated molecules (truth)
| option | default | meaning |
|---|---|---|
| `--n-rho`, `--n-eta` | 20, 20 | orientation grid: rho, eta at the centres of 20 bins of [0, 180[ (4.5, 13.5 ... 175.5 deg); n_eta even for the polar maps |
| `--rho-values`, `--eta-values` | - | explicit lists instead of the grid |
| `--per-point` | 80 | PSF per (rho, eta) point (20 x 20 x 80 = 32000) |
| `--delta` | 90 | true delta, the same for all |
| `--delta-range LO HI` | - | true delta uniform in [LO, HI] per PSF (replaces --delta) |
| `--xy-range` | 0.2 | true x, y uniform in +-0.2 um around the patch centre |
| `--z-range` | 0.2 1.2 | true z uniform (um above the coverslip) |

### 2.3 Photons and noise (default None = from the preset)
| option | meaning | preset poisson | preset actin |
|---|---|---|---|
| `--camera` | `poisson` (Poisson + read noise, no QE) or `emccd` (QE, EM excess, read noise) | poisson | emccd |
| `--N` | photons of the whole molecule (3 planes) | 3000 | 3000 |
| `--background` | mean background, photons/pixel, with its Poisson noise | **25** | **25** |
| `--sigma-b` | extra background fluctuation beyond Poisson (photons, per pixel and frame) | 0 | 3 |
| `--sigma-r` | read noise (photons) | 0 | 0.4 |
| `--crazy-fraction` | fraction of pixels with extra photons | 0 | 0 |
| `--two-emitter-fraction` | fraction of PSF with a second emitter | 0 | 0 |
Presets slb (N 1170, background 3.4, sigma_b 0.5) and actin_dense (actin + 3 % crazy pixels + 20 % second emitters)
also exist. Fixed defaults (not from the presets): `--QE 0.92`, `--EM 200`, `--cic 0`, `--crazy-amplitude 100`,
`--neighbour-distance 3 5` (pixels of 120 nm), `--neighbour-N 0.5 1.5` (x N).

**Warning: every preset has a background (25, or 3.4 for slb), including `poisson`.** The noise types used in the
thesis (user's definitions, 2026-11-12; "background" always means a background with its Poisson fluctuations):
| noise type | options |
|---|---|
| Poisson | `--preset poisson --background 0` |
| Poisson + read + excess | `--preset actin --background 0 --sigma-b 0` |
| Poisson + read + excess + background | `--preset actin --sigma-b 0` (background 25) |

### 2.4 Geometry and calibration (same for data and fit)
| option | default | meaning |
|---|---|---|
| `--d` | -1.0 | middle focal plane = -d = 1.0 um |
| `--interplane` | 0.354 | distance between planes: planes at 0.646 / 1.0 / 1.354 um |
| `--calibration` | - | config.txt of a real run: its J1, J2, rotations, y gains for data and fit, instead of the ideal calibration (J1 identity, J2 45 deg rotation, rotations 0, rho_offset 0, y gains 1) |
Not options (taken from the parameter cell of SGD_jax.py, changeable only with `--set`): emission wavelength
(620 nm), NA 1.4, magnifications (100 x 4/3), camera pixel 16 um (120 nm in the sample), SAF, refractive indices
(review item 8: to be moved into the presets).

### 2.5 Deliberate mismatch (data only, the fit keeps the standard model)
`--data-y-gain-J2` (y channel of plane 1 x factor), `--data-n2` (sample index), `--data-saf-dz` (more / less SAF),
`--data-shift PLANE POL DX DY` (one channel shifted, nm), `--data-zshift` (shift vs z of the emitter, table),
`--data-zernike` (aberrations), `--warp` (interpolation of the reconstruction on the raw channels).

## 3. Camera noise: `camera` (line ~182) and the functions of simu_PSF_polarMFM_JAX.py

`camera(rng, expected, a)` takes the expected image of the PSF (photons per pixel, no background, no noise) and
returns the noisy image in photons, as the camera data after the reconstruction (no baseline). The JAX random key
is drawn from the generator of the run: same `--seed`, same noise.

**`--camera poisson` -> `noise_jax`:**

    background = max(b + sigma_b N(0,1), 0)
    value      = Poisson(PSF + background) + sigma_r N(0,1)

mean PSF + b, variance PSF + b (+ sigma_b^2) + sigma_r^2. Perfect detector (no QE): kept on purpose (user,
2026-11-12), "Poisson" = the ideal detector.

**`--camera emccd` -> `emccd_jax`:**

    light     = max(PSF + b + sigma_b N(0,1), 0)
    electrons = Poisson(QE light + cic)                       # 92 % of the photons give an electron
    amplified = Gamma(shape electrons, scale EM)  (0 if none) # EM register, high-gain approximation
    adu       = round((amplified + sigma_read_e N(0,1)) / 4.4)
    value     = adu x 4.4 / (QE EM)                           # back to photons

mean = light, variance = 2 light / QE + sigma_r^2 (+ ADC rounding ~5e-5, negligible). The 2 is the EM excess
noise; the read noise is given in photons (`--sigma-r`) and converted to electrons at the EM output
(sigma_r x QE x EM: 0.4 photons = 73.6 e-).

Variance per pixel of the three noise types (s = expected PSF photons in the pixel):
| noise type | variance |
|---|---|
| Poisson | s |
| Poisson + read + excess | 2.17 s + 0.16 |
| Poisson + read + excess + background 25 | 2.17 (s + 25) + 0.16 |
"Poisson" -> "Poisson + read + excess" multiplies the variance by 2/QE = 2.17 (not 2): expected precision ratio
sqrt(2/QE) = 1.47. Values can be negative in empty pixels (read noise, rounding), as real data after the baseline
subtraction.

## 4. Crazy pixels: `crazy_pixels`
Each pixel has a probability `--crazy-fraction` of receiving extra photons, drawn from an exponential law of mean
`--crazy-amplitude` (100). They are added after the camera noise (no Poisson / EM noise of their own). Mimics the
irregular background of dense samples; preset actin_dense only.

## 5. Preparing the fit pipeline: `build_pipeline`
Purpose: start the real fit pipeline (`SGD_jax.py`, or `SGD_jax_CNN.py` with `--cnn`) exactly as on real data, and
stop just before the loop that fits the batches.

**`ns` (namespace).** A Python dictionary that plays the role of the memory of a Python session. Each cell of the
pipeline is executed with `exec(cell, ns)`; everything it creates goes into `ns`: e.g. `ns['middle_plane']`,
`ns['d_']`, `ns['NPSF']` (100 PSF per batch), `ns['Nframe']` (20 frames for the noise model), `ns['lambd']`,
`ns['NA']`, `ns['SAF']`, `ns['J1']`, `ns['J2']`, `ns['compute_M_jax']`, `ns['PSF_jax']`, `ns['noise_model']`. Why it
matters: (1) the simulated data are built with the objects of this same `ns` (`model_psf`), so data and fit share
exactly one model; (2) a piece of the pipeline can be swapped before the fit (`ns['load_batch'] = ...`): the main
loop, run inside `ns`, then calls the simulated version; (3) a new `ns` for each loss, deleted afterwards.

Steps:
1. read the text of the pipeline (when the run starts: editing `SGD_jax.py` during a queue changes the later runs);
2. `apply_overrides` (SGD_batch.py): each top-level line `name = ...` named in the overrides becomes `name = value`:
   ideal calibration (J1, J2, rotation, rotation2, rho_offset, unitary_J, y gains), `loss2`, `middle_plane`, and
   `--set`. A name absent from the script stops with an error (no silent typo);
3. cut into cells at the `#%%` markers;
4. run the cells in order in `ns`:
   - "OPTIONAL" cells (reload an old run, resume): skipped;
   - "extracting positions" cell (reads the pre-localisations and camera files of a real acquisition): replaced by
     the interplane (`--interplane`), the variance without light V = sigma_r^2 (written in ADU^2, converted back to
     photons^2 by the pipeline) and empty camera metadata;
   - "gradient descent" cell: only its first part (definitions of the fit functions); the main loop (after the
     marker `################# main loop`) is kept as text, run later by `main()`;
   - every other cell unchanged (parameters, optics, noise model, losses, saving of the config);
   - after the parameter cell the npz folder is moved to `<run folder>/NPZ_<loss>`;
5. the folder dialogs of the pipeline are answered with the run folder (`FakeDialog`), figures not shown.
Weak point (review item 9): the cells are found by the text of their headers; a renamed header changes the
simulation without error.

## 6. Feeding the simulated PSF to the pipeline: `batch_builder`, `simulated_load_batch`
The real `load_batch` (SGD_jax.py / SGD_jax_CNN.py) does two things: (1) reads 20 raw frames, measures the noise model on them, cuts
a 13 x 13 patch around each pre-localisation and rotates it into the model frame; (2) packs the batch into the
dictionary `result` used by the fit:
| key | content |
|---|---|
| `noisy_psf` | the 100 patches (100, 3 planes, 2 pol, 13, 13), photons |
| `x`, `y` | position of each patch |
| `Nstart` | fixed start of N for SGD_jax.py (2571; the CNN version uses its own guess) |
| `background_array` | start of the 6 backgrounds = `level` of the noise model |
| `noise` | gain and shift of the 6 channels (noise model of the fit, var = gain x model + V, shift = V/gain) |
| `frame` | index of each PSF |
| `buffer`, `psf_buffer`, `last_frame_processed` | leftovers, counter |
`batch_builder` copies part (2) from the text of the pipeline itself, so the batch is packed exactly as for real
data. `simulated_load_batch` returns a replacement of `load_batch` (installed as `ns['load_batch']`) that does part
(1) with the simulation, at each call (batch b):
1. takes PSF b x 100 ... (b + 1) x 100 - 1 of the simulation (already in the model frame, no rotation);
2. simulates 20 background-only frames (6 channels, 64 x 64) with the same camera and background (+ warp with
   `--warp`);
3. Poisson camera: adds a uniform +-0.5 to these frames (integer values would leave empty bins in noise_model);
4. measures the noise model with the pipeline's `noise_model`. The background is uniform, so all pixels have the
   same level and the result is (variance - V)/level with V known. Measured (review 2026-11-12): Poisson 1.001 at
   b 25, 1.023 at b 5, 1.062 at b 2 (the dither adds 1/12); EMCCD 2.149 vs 2/QE = 2.174 at b 25, V 0.160. With
   background 0 nothing can be measured: the true noise is used (gain 1 or 2/QE, V = sigma_r^2; fix of 2026-11-12);
5. reorders the channels as the real `load_batch` (planes reversed; no effect, the 6 channels have the same noise);
6. x = y = 0 (centred patches, the true offset is the position), `frame` = PSF index;
7. runs the copied part (2).
Not as on real data: noise measured on clean uniform background frames (real data: the frames with molecules and an
uneven background); no pre-localisation (every PSF is fitted); no resuming (`n_skip` ignored).

## 7. Truth and images: `grid`, `model_psf`, `simulate`

**`grid`:** rho and eta at the centres of 20 bins of [0, 180[ (4.5, 13.5 ... 175.5 deg), every pair: 400 points
(or the lists `--rho-values`, `--eta-values`).

**`model_psf`:** noiseless image of each molecule with the pipeline's own functions taken from `ns`:
1. `compute_M_jax`: basis images M of the 3 planes x 2 polarisations at x, y, z (optics: pupil fields, wavelength,
   NA, SAF, plane positions `d_`, calibration J, aberrations none);
2. `PSF_jax`: combines M with the orientation (rho, eta) and the wobbling cone delta, normalises the whole image
   (3 planes x 2 pol) to N photons;
3. `channel_gain`: y channels x y gains (1 here); 4. crop 13 x 13 around the centre.
**Meaning of N:** the image is normalised to N whatever the orientation and z, so N = photons reaching the 6
channels, not the emitted ones (an axial dipole gives as many photons as an in-plane one). The total brightness
carries no orientation information in this model; same convention in the fit and in the CRB (old torch model).
Modelling choice, not a bug.

**Warp of the reconstruction (`--warp` only, not used in the runs of 2026-11-12; reviewed 2026-11-12).**
Why: in real data the reconstruction (reconstruction_juliette.py) cuts the 6 channels from the camera image and
resamples each one with `cv2.warpPerspective(channel, M_i, INTER_CUBIC)` to register it on the reference channel
(canal 1, M = identity). Each output pixel is a cubic interpolation of 4 x 4 raw pixels: slight blur of the PSF and
correlated noise between neighbouring pixels (lower variance per pixel), which the fit does not model. `--warp
calibration.npz` reproduces this on the simulated raw patches.

- `WARP_CANAL[plane][pol]`: camera canal of each channel of the model. The planes are reversed in the model (plane 0
  of the model = canals 2/5 of the reconstruction, as load_batch of SGD_jax.py does), pol 0 = top row of the camera
  (canals 0-2), pol 1 = bottom row (3-5). Checked against reconstruction_stack (order 0, 3, 1, 4, 2, 5) and load_batch.
- `load_warps`: reads M_0 ... M_5 of the calibration, keeps their inverses and the channel size.
- `warp_maps(rng, warps, canal, n, side)`: warpPerspective computes dst(X, Y) = src(M^-1 (X, Y)). For n regions of
  side x side pixels at random places of the channel, it gives the remap maps (which raw pixel, non-integer, each
  output pixel reads) on a local patch, and the sub-pixel shift at the centre (row, col in -0.5..0.5 px; the integer
  part only moves the region). The content at raw position p appears at p - shift after the warp. Random places:
  each PSF gets the sub-pixel phase of its own position, uniform, as molecules anywhere in the field.
- `remap`: OpenCV cubic remap with these maps (same kernel as warpPerspective).
- `pixels_per_um`: matrix A[plane, pol] (2 x 2) = displacement (row, col) of the PSF in pixels per um of (x, y),
  measured on the model (centroids of a noise-free PSF moved by 0.06 um in x, then y). About 8.3 px/um (120 nm
  pixels), with the sign / axis swap of each channel of the model.
- `warped_data`: for each channel, (1) random warp region -> maps + shift; (2) the emitter is moved by
  move = A^-1 shift (um) in this channel only, so that after the warp the PSF is back at the true x, y; (3) expected
  raw patch 17 x 17 (13 + 2 MARGIN), with the second emitter and the y gain of the data; (4) camera noise + background
  and crazy pixels on the raw patch, as on the camera; (5) cubic remap, crop of the central 13 x 13. MARGIN = 2 is the
  reach of the cubic kernel for |shift| <= 0.5, so the border reflection of OpenCV is never read.
- The background frames of the noise model (`simulated_load_batch`) go through the same warps, so the fit's noise
  model sees the reduced variance of the interpolated noise, as on real data.

Not reproduced (review item 13): the division of each channel by its lamp normalisation; the row flip of load_batch
(the warp is applied in the orientation of the model: mirrored sub-pixel pattern, same statistics). `warped_data`
uses `model_psf`, not `data_psf`: `--data-zernike` and the pupil fields of `--data-n2` are ignored with `--warp`
(`--data-shift` / `--data-zshift` are refused). The model is computed for the 6 channels and one is kept: `--warp`
runs are slower.

**`data_physics` (mismatch of the sample; options `--data-n2`, `--data-saf-dz`).** A context
(`with data_physics(ns, a) as phys:` in `simulate`): on entering it temporarily changes the model module
(simu_PSF_polarMFM_JAX.py), on leaving it puts everything back, so the fit sees the normal model.
- `--data-n2`: another sample index for the data (critical angle, Fresnel coefficients, phase vs depth); the pupil
  fields of the data are recomputed and passed to `model_psf` (`phys.fields`).
- `--data-saf-dz`: the axial phase k n2 sqrt(1 - (n1 sin(theta)/n2)^2) z is imaginary beyond the critical angle
  (evanescent light, a decay); dz is added to the decay only: the SAF light decays as for an emitter dz further from
  the glass, the propagating light is unchanged (too much / too little SAF, alone).
- `jax.clear_caches()` on entering and leaving: JAX keeps the compiled model functions, which would otherwise keep
  the old values.

**`data_psf` (expected images of the DATA).** `model_psf` with the sample of `data_physics`, then: `--data-zernike`
(aberrations of the data only), `--data-shift PLANE POL DX DY` (one channel, pol 2 = both, recomputed with the
emitter moved by DX, DY nm: registration error), `--data-zshift table` (shift of each channel depending on the
emitter's z, interpolated, end values outside the table). Without any `--data-*` option, `data_psf` = `model_psf`.
Known limit (review item 11): `--data-shift` and `--data-zshift` on the same channel do not add up.

**`simulate`:** truth of every PSF, then its image, batch by batch:
| parameter | how it is drawn |
|---|---|
| rho, eta | grid point, 80 consecutive PSF per point |
| delta | `--delta` for all, or uniform in `--delta-range` (own generator, seed + 3) |
| N | `--N` for all (photons of the molecule) |
| x, y | uniform +-0.2 um around the patch centre |
| z | uniform 0.2-1.2 um above the coverslip |
Draws in a fixed order (x, y, z, second emitter even when off, then camera noise batch by batch): the same seed
gives the same molecules in every run (any noise type, any fit). Images: `data_psf` -> second emitter (if any) ->
y channel of plane 1 x `--data-y-gain-J2` -> `camera` (background + noise) -> `crazy_pixels`. Saved in
`simulation.npz`. 20 x 20 x 80 = 32000 = 320 whole batches.
- Second emitter (`--two-emitter-fraction` > 0, preset actin_dense only): random orientation (uniform over the
  sphere), N x uniform(0.5, 1.5), own z, 3-5 pixels away in a random direction; not fitted, a perturbation of the patch.
- With `--delta-range` the 80 PSF of a grid point have different delta: the bias and precision of a point (and the
  maps) are averages over delta 60-120. The CRB to compare with must be averaged over the same delta range.
- The noise of the fit's noise model (background frames) uses another generator (rng_noise), so it does not
  change the noise of the data.

## 8. Errors and statistics: `wrap`, `collect`, `robust_std`, `statistics`
**`wrap`:** angle difference brought back into [-90, 90[ (175 - 5 = 170 -> -10).

**`collect`:** reads the npz of the run (batch order); `frame` = PSF index, so the truth of each fit is
`sim['true_...'][idx]` (one fit per simulated PSF, no matching). Errors:
| parameter | error |
|---|---|
| rho | found - true, wrapped to [-90, 90[ |
| eta | found - true after flipping eta (180 - eta) when the rho wrap moved rho by 180: (rho + 180, 180 - eta) is the same orientation |
| delta | found - true |
| x, y | found x correction_dft - true, nm (undoes the factor of the saving) |
| z | found - true, nm |
| N | (found - true) / true |
Example of the flip: truth (175, 60), fit (2, 120) = (182, 60) folded -> rho error +7, eta error 0. Valid because the
pipelines return rho, eta in [0, 180] (checked; ~8.5 % of the PSF cross, those near rho 0/180).

**`robust_std`:** 1.4826 x median absolute deviation (= std for a Gaussian, ignores outliers).

**`statistics`:** per grid point (80 PSF): bias = median of the error, precision = robust std; diverged fits
(nan) counted and left out. Statistical noise per point: bias +-0.14 sigma (delta +-2.8 deg, eta +-1.3 deg),
precision +-13 %: point-to-point patterns of that size in the maps are noise. rho near the axis (eta 4.5 / 175.5)
is almost undefined (large rho precision there expected).

## 9. Figures: `disc_map`, `case_figure`, `compare_losses`
Made at the end of the run by `summary` (after summary.csv), from the rows of `statistics` (one row per grid point).

**`disc_map`**: one value per grid point drawn on a polar disc, no interpolation (one coloured cell per simulated
(rho, eta)). Angle = rho, radius = eta: centre = dipole along the optical axis, edge = dipole in the plane. A dipole
(rho, eta > 90) is the same as (rho + 180, 180 - eta), so the grid points eta > 90 are drawn on the half 180-360 deg
of the disc, at radius 180 - eta; the half 0-180 holds eta < 90. Example: truth (30, 120) is drawn at angle 210,
radius 60.
- Reading an eta **bias** map: the bias is in the coordinates of the truth, so on the half 180-360 a positive eta bias
  means the fit moves the dipole towards the centre of the disc; on the half 0-180, towards the edge. rho bias has
  the same meaning on both halves.
- Needs a regular grid (rho bins covering 0-180 evenly, an even number of eta bins symmetric about 90); otherwise
  `summary` writes "no polar maps" in the log.

**`case_figure`** -> `polar_summary_<case>.png` (case = folder name without the date, e.g. nobg_poisson_cnn):
- top block bias (median of the error), bottom block precision (robust std), for rho, eta, delta (deg) and N (%);
- one row per loss (only huber in the runs of 2026-11-12);
- colour scales shared by the losses: bias +-(95th percentile of |bias|), precision 0 to the 95th percentile;
- panel title = median over the grid of |bias| or of the precision. Even an unbiased fit gives a median |bias| of
  about 0.1 sigma (statistical noise of 80 PSF per point, section 8).

**`compare_losses`** -> `compare_losses_vs_eta.png`: for each true eta, the median over rho of the bias (top) and of
the precision (bottom) of rho, eta, delta, z, x, N, one curve per loss. The median over rho of a signed bias can
cancel biases of opposite sign at different rho: look at the polar maps for that. N is relative here (0.01 = 1 %).

The CNN-start maps (rows 3-4: bias and std of the CNN first guess) are not made by this script: they come from
`analysis/polar_maps.py` next to the runs (colour bars 5-95 %).

## 10. One run: `main`
1. `parse_arguments`; `--cnn` switches the pipeline to SGD_jax_CNN.py (CNN first guess + per-PSF LM).
2. Run folder `<out>/<date>_<hh>h<mm>_<tag>`; all the options written in the log.
3. Calibration of the model, same for the data and the fit: ideal (J1 identity, J2 45 deg rotation, rotations 0,
   rho_offset 0, y gains 1), or that of a real run with `--calibration config.txt` (rho_offset kept 0).
4. Random generators: seed (truth + data noise), seed + 1 (background frames of the noise model), seed + 2 (warp
   regions), seed + 3 (delta of `--delta-range`).
5. For each loss of `--losses` (only huber in the runs of 2026-11-12):
   - overrides of the parameter cell: calibration, loss2 = this loss, middle_plane = -`--d` (default -1.0 -> 1.0 um),
     then `--set`. `--set loss2=` or `--set middle_plane=` crash (TypeError, review item 14);
   - `build_pipeline` (section 5); the data are simulated once, with the first pipeline (`simulate`, saved in
     simulation.npz), and reused for the other losses;
   - `batch_nb` = 320, `load_batch` replaced by `simulated_load_batch` (its background frames are the same for every
     loss: generator re-created);
   - the main loop of SGD_jax.py / SGD_jax_CNN.py runs unchanged in `ns` (one npz per batch in NPZ_<loss>);
   - `collect` -> results_<loss>.npz (found, true, error per PSF) -> `loss_statistics`; JAX caches cleared.
6. `summary`: summary.csv, figures, summary lines in the log.

**`median_line`**: one log line, "median over the grid of |bias| / median precision" per parameter (angles deg,
x y z nm, N relative: 0.03 = 3 %).

**`loss_statistics`** (per loss): `statistics` on all PSF, log of the number of diverged fits (nan) and of
`median_line`; with `--delta-range`, least-squares line found delta = slope x true + offset, on all PSF and on
eta 60-120: slope 1 = the fit follows delta, slope < 1 = shrinkage towards a common value, offset = global bias
(fits at the delta bound included). With second emitters, the isolated and neighbour PSF separately.
Example (2026-11-12, CNN + LM, no background): Poisson rho 0.26/2.53, eta 0.24/2.12, delta 0.56/4.57 deg, slope
1.003, offset -0.7; EMCCD rho 0.41/3.84, eta 0.55/3.20, delta 1.24/7.21 deg, slope 1.006, offset -1.9; N bias -3.3 %
with EMCCD (review item 15).

**`summary`** (end of the run): summary.csv (one row per grid point and loss: point, rho, eta, n, n_failed,
bias_<k>, sigma_<k> for rho, eta, delta, x, y, z, N), compare_losses_vs_eta.png, polar_summary_<case>.png, and one
log line per loss. The log is printed with the time (`SGD_batch.log`): the queue scripts redirect it to `<tag>.log`.

**`recompute`**: `python simu_precision/SGD_precision.py --recompute <run folder> [...]` redoes the statistics and figures
from simulation.npz and results_<loss>.npz without fitting again (after a change of the statistics or the figures).
It overwrites summary.csv and the png; its log lines go to the console only. The CNN-start maps come from
`analysis/polar_maps.py`, not from here.

## 11. Cramer-Rao bound
Files: fisher_4pMFM.py (Fisher matrix), simu_PSF_polarMFM.py (torch model of the PSF), simu_precision/crb_vs_unbiased_sim.py
(comparison with the simulations; cache and figures in D:\AMAURY\simu_precision_claude\2026-10-28_crb_fisher).

### 11.1 `fisher(M, RHO, ETA, DELTA, Np, BACKGROUND, gain=1, dark_var=0)` (fisher_4pMFM.py)
Fisher information of (rho, eta, delta) for each PSF, shape (3, 3, NPSF), angles in radians. x, y, z, N and the
background are known (not estimated): optimistic bound.
- Model, as `PSF()` of the torch model: image I = sum_ij Re(M_ij) m_ij, with the second-moment tensor of the
  wobbling dipole m = (1 - gamma)/3 Id + gamma u u^T, u = dipole direction (sin eta cos rho, sin eta sin rho, cos eta),
  delta = full cone angle, gamma = cos(delta/2)(1 + cos(delta/2))/2 (1 for a fixed dipole, 0 for delta = 360).
  The PSF is normalised to N photons over the whole computed image: s = N I/S.
- Derivatives: ds/dtheta = (N/S) dI/dtheta - (s/S) dS/dtheta, with dI/drho, dI/deta from d(u u^T), and dI/ddelta
  from (u u^T - Id/3) x dgamma/ddelta, dgamma/ddelta = (dgamma/dOmega) pi sin(delta/2) (fixed 2026-10-28).
- F_uv = sum over the 3 planes, 2 polarisations and the pixels of the window of ds/du ds/dv / var,
  var = gain (s + b) + dark_var. Exact for Poisson (gain 1, dark_var 0, the CRB of the topic).
- Checked 2026-11-12 against finite differences of `PSF()` (review/simulation_precision_fisher_2026-11-12/fisher_fd_check.py): identical (2e-4).
- Window: `window` x `window` pixels centred on each channel, default 13 = the patch of the fit (2026-11-12; it was
  14 x 14, one more row and column, CRB 1-4 % too low, review item 16; `window=14` gives the old pixels).
- Example (b = 0, N 2100, one PSF each, 13 x 13): CRB (rho, eta, delta) at eta 90: 1.37, 1.84, 3.52 deg; at
  eta 10: 16.2, 2.8, 4.0 (rho is ill-defined near the axis).

### 11.2 `fisher_ratiometric(..., window=11, fixed_eta=False, gain=1, dark_var=0)`
Bound of a ratiometric method: the information in the 6 channel intensities only (each summed over a window x
window square, `None` = the whole image), not in the shape of the image. Same model and derivatives as `fisher`;
`fixed_eta=True` gives the 2 x 2 matrix of (rho, delta) with eta known (with the whole image, eta and delta cannot be
separated). Used only in simu_precision/MFM_fisher.py, for comparison with the full-image bound. Checked 2026-11-12 against
finite differences (within 6e-4).

### 11.3 Torch model (CRB) vs JAX model (simulation)
The CRB uses the torch model (simu_PSF_polarMFM.py, `compute_M` + `PSF`), the simulation and the fit the JAX model
(simu_PSF_polarMFM_JAX.py, `compute_M_jax` + `PSF_jax`). Checked 2026-11-12 (review item 18): for the same molecules
they give the same image to 0.000 once the wavelength and the 45 deg convention are the same.
- Polarisation of the middle plane: the JAX model ignores `polar_projections` and uses the Jones matrices
  `J_dichroic` = [J1, J2, J1] applied to the fields, with `rot(a)` = [[cos a, -sin a], [sin a, cos a]]. The ideal
  simulation uses J2 = rot(45): a dipole at rho 45 deg lights the x channel of the middle plane (analyser at rho 45).
  The torch model uses `polar_projections`; the same convention is [0, -45, 0] (the fields are mirrored with respect
  to rho: [0, 45, 0] is the mirror system, same CRB averaged over rho, not at a given rho). crb_vs_unbiased_sim.py
  uses [0, -45, 0] since 2026-11-12.
- Wavelength: 617 nm in crb_vs_unbiased_sim.py, 620 nm in the simulation (CRB changed by < 0.5 %).

### 11.4 Analysis scripts of simu_precision/ (generic: any run folders)
- `crb_vs_simulation.py <run> [...] --out <folder>`: CRB (Poisson only, no background, 13 x 13 window, x y z N known)
  with the conditions of each run (N, delta, x, y, z = true values of a random subset of its simulation.npz; optics
  from its config.txt; ideal J only, torch polar_projections [0, -45, 0]) vs its simulated precision per true eta:
  plain std (filled points, the quantity the CRB bounds), robust std (open points), outliers (% with |error - median|
  > 5 robust std). One figure per method (SGD_jax.py, CNN + LM), a text summary, the CRB cached per condition.
- `polar_maps.py <run> [...]`: maps of any run (see section 9), with the CNN first guess rows for a CNN run.
- `compare_sgd_cnn.py <SGD run> <CNN run> [...] --out <file>`: table of paired runs (same truth).
Results of 2026-11-12 (no background): CNN + LM with Poisson noise at 1.07-1.12 x the CRB (rho, eta, delta), with
Poisson + read + excess at 1.6-1.7 x (EMCCD excess alone: sqrt(2/QE) = 1.47); SGD_jax.py 1.5-2.3 x and 2.3-3.2 x.

## 12. Changes of 2026-11-12 after the review (review list, "Fixes")
- Script moved: `simu_precision/SGD_precision.py` (was SGD_precision_claude.py), run as
  `python simu_precision/SGD_precision.py ...`.
- New options: `--middle-plane` (instead of --d, which stays), `--lambda-emission`, `--NA`, `--mag`, `--MAG`,
  `--l-pixel`, `--SAF` (default: the parameter cell of SGD_jax*.py); `--set` wins over all the other overrides.
- Log: the noise in plain words (`noise: Poisson ..., NO background`) and the model used (`model (data and fit): ...`).
- Checks: stops if a cell header / marker of SGD_jax*.py is missing or doubled; refuses --data-zernike / --data-n2
  with --warp; --data-zshift adds to --data-shift.
- summary.csv: `bias_err_<k>` and `sigma_relerr_<k>` (statistical noise of each grid point); map titles give it.
- Maps: eta bias with + = towards the edge of the disc (more in-plane) on both halves.

