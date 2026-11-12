# Code review: simulation precision and Fisher (branch simulation-review, started 2026-11-12)

Note (2026-11-12, after this review): SGD_precision_claude.py -> simu_precision/SGD_precision.py, MFM_fisher_claude.py ->
simu_precision/MFM_fisher.py, crb_vs_unbiased_sim.py -> simu_precision/crb_vs_simulation.py (generic). The names below are
those of the review.

Worktree GitHub\4polarMFM_these_simulation_review, branch simulation-review from fb633a7 + the uncommitted changes
of main as they were run on 2026-11-12 (SGD_jax.py, SGD_jax_CNN.py, SGD_precision_claude.py, simu_PSF_polarMFM_JAX.py,
stream_fit.py, ... identical to main at the start).

## Review list (file:line - issue - status)

1. SGD_precision_claude.py, simulated_load_batch - with background 0 the noise model of the fit is measured on empty
   background frames: gain -0.12..-0.16, deviance -8 (run nobg_poisson_cnn of 2026-11-12, stopped). FIXED in this
   branch: background 0 and sigma_b 0 -> the true noise of the simulation (gain 1 Poisson / 2/QE EMCCD, V = sigma_r^2).
2. SGD_precision_claude.py, PRESETS - every preset has background 25 (also 'poisson'): "Poisson" in the sense of the
   user (no background) needs --background 0. The 3 cases of 2026-11-10 all had background 25 (case1, case3 deleted).
   TO DO: print the resolved noise conditions in plain words in the log; rename / document the presets.
3. SGD_precision_claude.py, camera() - camera 'poisson' (noise_jax) has no QE (Poisson on s + b photons), camera 'emccd'
   thins by QE (variance 2 (s + b)/QE): "Poisson" -> "Poisson + excess" changes the variance by 2/QE = 2.17, not 2.
   To be stated in the comments (expected precision ratio sqrt(2/QE) = 1.47). DECISION (user, 2026-11-12): keep
   "Poisson" as a perfect detector (no QE). Comments added.
4. SGD_precision_claude.py, header - "delta = 90" in the docstring: also --delta-range. Comment to update.
5. Fisher (crb_vs_unbiased_sim.py): torch model of simu_PSF_polarMFM.py, lambda 617 nm, J identity +
   polar_projections 0/45/0, vs the JAX model of the simulation (lambda of SGD_jax.py / SGD_jax_CNN.py, J1 identity, J2 45 deg
   rotation). TO CHECK that both give the same PSF (same N normalisation, same planes, SAF, pixel size).
   CHECKED 2026-11-12, see item 18.
6. Precision definitions: maps use robust std (1.4826 MAD) per grid point; the CRB comparison uses sqrt(mean over rho of
   std^2) (non robust, as CRB_vs_SGD_eta.png): the CNN delta with EMCCD noise includes the 3-4 % at the delta bound.
   DECIDED (user, 2026-11-12): plain std against the CRB (the CRB bounds the variance, outliers are a real cost),
   robust std kept for the maps and shown as open points on the CRB plots, outliers (|error - median| > 5 robust std)
   tracked in the plots and the text (simu_precision/crb_vs_unbiased_sim.py, ad6639e).
   Measured 2026-11-12 (CNN runs without background, median over the grid points with eta 20-160 of std / robust
   std): rho 1.10, eta 1.08-1.09, delta 1.01-1.02, z 1.24-1.25 (same for Poisson and EMCCD). So the error
   distributions have heavier tails than a Gaussian (z most); the choice changes rho/eta by ~10 %.
7. SGD_jax.py / SGD_jax_CNN.py noise_model, used on simulated background frames (simulated_load_batch) - CHECKED 2026-11-12: with a
   uniform background all pixels have the same level, so the "slope" is in fact one point (variance - V)/level with
   V fixed at the known dark variance: correct only because V is known. Measured gain: Poisson camera (with the
   uniform dither of 1 added for the measurement) 1.001 at b 25, 1.023 at b 5, 1.062 at b 2 (true 1; the dither
   adds 1/12 to the variance: +0.3 / +1.7 / +4.2 %); EMCCD case2 (b 25) 2.149 +- 0.022 vs 2/QE = 2.174 (-1.2 %),
   V 0.160 (true 0.16). OK for b >= 5; b = 0 impossible (item 1). To comment in simulated_load_batch.
8. SGD_precision_claude.py, PRESETS / parse_arguments (user, 2026-11-12) - the optical parameters (lambda_emission,
   NA, mag, MAG, l_pixel, SAF, n1/n2, middle_plane / interplane) are not options: they come silently from the
   parameter cell of SGD_jax.py / SGD_jax_CNN.py (only --set can change them). TO DO: put them in the presets (and as
   options), passed to the pipeline as overrides, and printed in the log with the noise conditions (item 2).
9. SGD_precision_claude.py, build_pipeline / batch_builder - the simulation finds the cells of SGD_jax.py / SGD_jax_CNN.py by their
   header text ('PARAMETERS TO BE DEFINED', 'extracting positions', 'gradient descent', 'OPTIONAL',
   '################# main loop', "result['buffer'] = buffer"). All present once in both files (checked 2026-11-12),
   but renaming one in SGD_jax.py / SGD_jax_CNN.py changes the simulation silently (e.g. 'PARAMETERS TO BE DEFINED' renamed: the npz
   are no longer redirected to NPZ_<loss>). TO DO: check that each marker is found exactly once, else stop.
   Also: the source of SGD_jax.py / SGD_jax_CNN.py is read when each run starts: editing it during a queue changes the later runs.
10. SGD_precision_claude.py, statistics / maps - statistical noise per grid point (80 PSF): bias (median)
    +-0.14 sigma (delta +-2.8 deg, eta +-1.3 deg), robust std +-13 % (plain std +-8 %). Point-to-point colour patterns
    of that size in the bias maps are noise. TO DO: state it on the figures (or add the uncertainty to summary.csv).
11. SGD_precision_claude.py, data_psf - --data-shift and --data-zshift do not add up: --data-zshift recomputes the
    channel from the original position, so the --data-shift of a channel that also has a z shift is lost. Rarely used
    together; TO DO: apply the z shift on top of the fixed shift, or refuse the combination. data_physics reviewed:
    restore + jax.clear_caches correct; psi_z_saf changes only the evanescent part (ignores cos_th2, never passed).
12. SGD_precision_claude.py, disc_map / case_figure - on the polar maps the grid points eta > 90 are drawn at
    (rho+180, 180-eta) but the eta bias keeps the sign of the truth coordinates: the same physical shift (towards the
    optical axis) has opposite colours on the two halves of the disc. TO DECIDE: flip the sign of the eta bias on the
    half eta > 90 (bias = change of the radius on the disc) or state it in the figure title. Titles: median |bias| has
    a noise floor ~0.1 sigma (item 10). Commented 2026-11-12.
13. SGD_precision_claude.py, warped_data (--warp) - reviewed 2026-11-12: channel order (WARP_CANAL, planes reversed),
    direction of the warp (dst = src(M^-1 dst)), placement (move = A^-1 shift), MARGIN 2 correct. Issues: (a) uses
    model_psf, not data_psf: --data-zernike is silently ignored and --data-n2 only half applied (n2 of psi_z changed by
    data_physics, pupil fields not) -> TO DO: refuse these options with --warp (as --data-shift), or use data_psf;
    (b) the lamp normalisation of the reconstruction (gain per channel) is not applied (small, to note); (c) the
    model is computed for the 6 channels and one kept (6 x slower, efficiency only).
14. SGD_precision_claude.py, main - overrides = dict(calibration, loss2=loss, middle_plane=..., **extra): --set loss2=...
    or --set middle_plane=... crash with a TypeError (key given twice). TO DO: merge with {**calibration, ...,
    **extra} or refuse these names in --set (they have their own options --losses, --d). Also --d is minus the middle
    plane (--d -1.0 = middle_plane 1.0 um), sign easy to get wrong: TO DO: option --middle-plane instead.
15. Observation (loss_statistics log, 2026-11-12, CNN + LM, huber, no background): N bias -0.4 % (Poisson) but -3.3 %
    (EMCCD, all orientations), deviance median 0.88 / 0.63. Probably the huber loss (c 2) on the very skewed EMCCD
    distribution of pixels with < 1 photon (no background): the large values are clipped -> N underestimated. Not a
    bug of SGD_precision_claude.py; TO CHECK with the fit topic (compare with the SGD run, and case2_emccd with b 25).
    ANSWERED 2026-11-12 (test nobg_emccd_cnn_lossN, same data, --losses poisson huber): poisson N +0.15 %, background
    -0.004; huber N -3.26 %, background +0.10. The huber loss (one-sided clipping, huber_positive_only) causes it.
    -> fit topic (keep huber for real data with background, or a bias-corrected huber). SGD_jax.py without
    background: N -13 %, background +0.3 (SGD1 / SGD2 background estimate), also -> fit topic.
    Update 2026-11-12: N bias follows the background found in every run (median): nobg_poisson CNN -0.4 % (b found
    -0.01), nobg_emccd CNN -3.3 % (+0.10), nobg_poisson SGD -7.1 % (+0.30, no EMCCD skew involved), case2_emccd CNN
    +1.4 % (24.94 for 25), case2_emccd SGD -2.4 % (25.31). Main effect: N-background trade-off (true b = 0 at the edge
    of the allowed values), huber not proven. Idea (user): huber only on bright pixels; alternative: Fisher-consistent
    huber (subtract the expected huber term under the noise model). TEST proposed: same nobg_emccd data, CNN,
    --losses poisson huber.

## Fisher part (started 2026-11-12)
16. fisher_4pMFM.py, fisher() - REVIEWED 2026-11-12. Model identical to PSF() (gamma = cos(delta/2)(1+cos(delta/2))/2
    = lam3 - lam1 of PSF(), delta = full cone angle; derivatives of u u^T; d gamma/d delta = (dgamma/dOmega) pi
    sin(delta/2); derivative of the normalisation). Numerical check (review/simulation_precision_fisher_2026-11-12/fisher_fd_check.py): Fisher matrix = finite
    differences of PSF() within 2e-4, CRB identical to 0.01 deg (6 PSF, eta 10-160, b 0 and 25); no pixel with
    psf = 0 in the window (b = 0 would divide by 0). Issues: (a) window 14 x 14 (NN//2-7 ... NN//2+6), the fit uses
    13 x 13 centred: CRB 1-4 % too low (13 x 13: rho 1.36 -> 1.37-1.39, eta 1.79 -> 1.84, delta 3.49 -> 3.52 at
    eta 90) -> DONE 2026-11-12 (user): option window of fisher(), default 13 (centred, = the fit), window=14 = the old
    pixels; both checked against finite differences. Callers (MFM_fisher_claude.py, crb_vs_unbiased_sim.py) get the
    13 x 13 window once merged into main (crb_vs_unbiased_sim.py imports main: its CRB cache must then be redone); (b) x, y, z, N, b known (optimistic, plan
    point 3); (c) with gain != 1 the Gaussian Fisher term of the variance derivative is left out (irrelevant for the
    Poisson-only CRB).
17. fisher_4pMFM.py, fisher_ratiometric - REVIEWED 2026-11-12: used only by MFM_fisher_claude.py (ratiometric bound,
    6 channel intensities summed over a window; not the CRB of the simulations). Checked against finite differences of
    PSF() (window 11 and whole image, b 0 and 25, eta free / known): within 6e-4. Nothing to change.
18. Torch model (CRB) vs JAX model (simulation) - CHECKED 2026-11-12 (torch_vs_jax.py, crb_sign_check.py in this
    folder; read only). Same 8 molecules (random orientation, x, y, z), 13 x 13 patch of the 6 channels: torch with
    lambda 620 and the middle plane at -45 deg = JAX simulation to 0.000 (planes, SAF, defocus, pixel, padding, N
    normalisation identical). Two differences in the CRB set-up of crb_vs_unbiased_sim.py:
    (a) lambda 617 nm (simulation 620): ~3 % per channel, CRB changed by < 0.5 %;
    (b) sign of the 45 deg: torch polar_projections [0, 45, 0] projects the middle plane on +45 deg
        (Ex' = cos p Ex + sin p Ey); the simulation used J2 = rot(45) = [[c, -s], [s, c]] in J_dichroic, i.e. a
        projection on -45 deg (polar_projections is not used by compute_M_jax: that block is commented out). Mirror
        systems: information at (rho, eta) in one = at (-rho, eta) in the other. CRB averaged over rho identical
        (eta 20/45/90/135, 200 molecules: same to 0.01 deg), so the CRB plots of 2026-11-12 are valid; a CRB at a given
        rho was not.
    Sign convention (PSF figure analysis/psf_6channels, 2026-11-12): with J2 = rot(45) a dipole at rho 45 lights the x
    channel of the middle plane (the analyser at rho 45 in the rho frame of the model); "projection on +45 deg" above is
    in the coordinates of the fields, which are mirrored with respect to rho. rot(-45) was applied to the repo then
    REVERTED the same day (user, option a): the simulation keeps rot(45) (all the existing runs, figure labels right)
    and the torch model of the CRB uses polar_projections [0, -45, 0] (crb_vs_unbiased_sim.py), which gives the same
    PSF as the simulation to 0.000 (torch_vs_jax.py).
    Remaining: lambda 617 in crb_vs_unbiased_sim.py (TO DO: 620 when the CRB is redone with the 13 x 13 window).
19. simu_precision/polar_maps.py - REVIEWED 2026-11-12. The CNN start is not saved by the run
    (SGD_jax_CNN.py:1187 writes rho_start, delta_start, z_start = nan): the script runs cnn_first_guess again on
    simulation.npz with the same input as the run (noisy_psf = the simulated patches), the same checkpoint and the
    z table of the run (cnn_z_equivalence, middle_plane 1.0; the CNN is applied PSF by PSF, so the chunks of 2000
    instead of the pools of the run give the same values). Errors as collect(). FIXED: its scratch folder was a fixed
    path in Claude's session folder (not reproducible) -> tempfile. Fragile like item 9: cnn_z_equivalence is cut out
    of the source of SGD_jax_CNN.py by text. Suggestion for the fit topic: save the CNN start in the npz of the run.
20. simu_precision/compare_sgd_cnn.py - REVIEWED 2026-11-12: table only; same deviance formula in both pipelines
    (SGD_jax.py:311, SGD_jax_CNN.py:349), so "dev<other" is valid: without background SGD has the higher deviance on
    every PSF (its background +0.3 instead of 0), with background 25 on 99.5 %.
21. Point 7 of the Fisher plan (optional): EMCCD reference line CRB x sqrt(2/QE) = 1.47 x CRB on the CRB plots (the
    CNN + LM runs with Poisson + read + excess are at ~1.65 x CRB). Not done, user to decide.

## Fixes of 2026-11-12 (user: "do the fixes"), simu_precision/SGD_precision.py (= SGD_precision_claude.py)
- item 2 FIXED: noise_description(a) logged after the parameters, in the words of the user ("N 2100 photons per
  molecule; noise: Poisson + excess (EMCCD, QE 0.92, EM 200), read noise 0.4 photons, NO background").
- item 8 FIXED: options --lambda-emission, --NA, --mag, --MAG, --l-pixel, --SAF (None = parameter cell of
  SGD_jax*.py), passed as overrides; the model used is logged ("model (data and fit): lambda_emission 620, NA 1.4, ...").
- item 9 FIXED: build_pipeline stops if a cell header ('PARAMETERS TO BE DEFINED', 'extracting positions', 'gradient
  descent') or a marker ('################# main loop', result['buffer'] / result['last_frame_processed']) is not
  found exactly once.
- item 10 FIXED: statistics() adds bias_err_<k> (1.2533 sigma/sqrt(n)) and sigma_relerr_<k> (1.17/sqrt(n)) per grid
  point (summary.csv); the map titles give the noise per point (case_figure and polar_maps.py).
- item 11 FIXED: --data-zshift is added to the --data-shift of the same channel (was: fixed shift lost).
- item 12 FIXED: eta_bias_on_disc() flips the eta bias of the points eta > 90 on the maps (+ = towards the edge of the
  disc, more in-plane, everywhere); summary.csv keeps the bias in the coordinates of the truth.
- item 13 FIXED: --data-zernike / --data-n2 refused with --warp.
- item 14 FIXED: overrides merged with --set last (--set loss2= / middle_plane= no longer crash, --set wins); new
  option --middle-plane (sets --d = -middle_plane).
Tested 2026-11-12: run 2 x 2 x 50 PSF (--middle-plane 1.0 --NA 1.4 --set middle_plane=1.0), polar_maps.py on it,
--warp + --data-zernike refused.
22. simu_PSF_polarMFM.py, noise() (torch model; used only by PSF_demo.ipynb, not by the simulations nor the CRB) -
    REVIEWED 2026-11-12. Torch branch: Poisson on the photons, then Gamma(shape = photons x QE, scale EM)/(QE EM):
    variance (1 + 1/QE) mu = 2.09 mu, vs the physical chain of emccd_jax (Poisson(QE mu) photoelectrons, Gamma(e, EM)):
    2/QE mu = 2.17 mu (checked numerically at mu 0.3, 3, 30; equal for QE = 1). Adds a camera baseline (bias), the
    simulations have none. NumPy branch: no EM excess at all. fisher_4pMFM.py fisher() comment "gain = 1 + 1/QE: Poisson
    + EM excess" is this formula; 2/QE is the one of the simulations and of the camera (no effect on the CRB of
    2026-11-12: Poisson only, gain 1). PROPOSED: (a) noise() as emccd_jax, both branches; (b) comment of fisher() 2/QE.
23. simu_precision/MFM_fisher.py (= MFM_fisher_claude.py, script of the user's notebook) - REVIEWED 2026-11-12.
    Changes (user): lambda 620; set-up aligned with the simulations (x, y +-0.2 um, z 0.2-1.2 um, middle plane 1.0,
    planes 0.646 / 1 / 1.354, SAF on, polar_projections [0, -45, 0]); N per molecule = 2100 everywhere (N_MOLECULE);
    "WRT to eta" section on the runs of 2026-11-12 (nobg_poisson SGD and CNN, delta 60-120, Poisson, no background);
    CRB cache key with lambda, polar_projections and the fisher window; "CRB vs rho" section removed (did not run:
    eta_list undefined, rho in degrees passed as radians, identical copies of the curves). Dead code noted: measured
    J1 / J2 overwritten by the identity, phase mask outside the pupil set to 1 (no effect).

