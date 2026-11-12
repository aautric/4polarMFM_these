# Code review: simulation precision and Fisher (branch simulation-review, started 2026-11-12)

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
6. Precision definitions: maps use robust std (1.4826 MAD) per grid point; the CRB comparison uses sqrt(mean over rho of
   std^2) (non robust, as CRB_vs_SGD_eta.png): the CNN delta with EMCCD noise includes the 3-4 % at the delta bound.
   TO DECIDE which one to show against the CRB.
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
