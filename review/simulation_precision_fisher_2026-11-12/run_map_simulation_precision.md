# Run map: unbiased precision simulations (run_2026-11-12_nobg.sh), what is executed

Launched 2026-11-12 13:46 from the worktree simulation-review (code = main as run + fix of review item 1).
Results: D:\Amaury\simu_precision_claude\unbiased_2026-11-09\<date>_<tag>\ ; logs <tag>.log next to them.

## 1. Queue script: D:\Amaury\simu_precision_claude\unbiased_2026-11-09\run_2026-11-12_nobg.sh
For m in (cnn, sgd), for case in (nobg_poisson, nobg_emccd), one after the other (one GPU job):
    python SGD_precision_claude.py <preset> --N 2100 --delta-range 60 120 --losses huber --background 0 --sigma-b 0
                                   --out <folder> --tag <case>_<m> [--cnn]
    nobg_poisson: --preset poisson ; nobg_emccd: --preset actin
Kept from 2026-11-10 (run_overnight_gpu.sh): case2_emccd = --preset actin --sigma-b 0 (background 25).

## 2. Resolved options (presets filled in, SGD_precision_claude.py parse_arguments)
| option            | nobg_poisson | nobg_emccd | case2_emccd (kept) |
|-------------------|--------------|------------|--------------------|
| camera            | poisson (noise_jax) | emccd (emccd_jax) | emccd |
| N (molecule)      | 2100         | 2100       | 2100               |
| background b      | 0            | 0          | 25 (+ its Poisson noise) |
| sigma_b (extra)   | 0            | 0          | 0                  |
| read noise sigma_r| 0            | 0.4 photons (73.6 e- at the output) | 0.4 |
| QE, EM            | not used     | 0.92, 200  | 0.92, 200          |
| noise variance    | s            | 2 s/QE + 0.16 (+ ADC) | 2 (s + 25)/QE + 0.16 |
| noise model of the fit | true: gain 1, V 0 | true: gain 2/QE, V 0.16 | measured (noise_model on background frames) |
- grid: 20 rho x 20 eta at bin centres 4.5 ... 175.5 deg, 80 PSF each = 32000 PSF (320 batches of NPSF 100)
- true delta: uniform 60-120 per PSF (own generator, seed + 3); x, y uniform +-0.2 um; z uniform 0.2-1.2 um; seed 0
  (same truth for every case and method)
- no crazy pixels, no second emitter, no warp, no data mismatch options

## 3. Model (data and fit identical), from config.txt of the run
lambda_emission 620 nm, NA 1.4, mag 100 x 4/3, pixel 16 um (120 nm in the sample), SAF on, middle_plane 1.0 um,
interplane 0.354 um (planes 0.646 / 1 / 1.354), ideal polarisation: J1 identity, J2 45 deg rotation, rotations 0,
rho_offset 0, y gains 1, no aberration, registration_z None, 13 x 13 px patches.

## 4. Chain of execution
1. SGD_precision_claude.py main(): parse_arguments (presets) -> build_pipeline: SGD_batch.py helpers read SGD_jax.py
   (or SGD_jax_CNN.py with --cnn), apply the overrides (calibration ideal, loss2, middle_plane), execute its cells up
   to '################# main loop' (parameters, model fields, noise_model, losses, ...).
2. simulate(): truth (grid, delta, x, y, z) -> model_psf (compute_M_jax + PSF_jax of the pipeline namespace, x
   channel_gain) -> camera(): noise_jax / emccd_jax of simu_PSF_polarMFM_JAX.py -> simulation.npz.
3. load_batch replaced by simulated_load_batch: each batch = 100 simulated PSF; noise model of the fit measured by
   noise_model on 20 simulated background frames (or the true one when background 0, review item 1); the end of the
   real load_batch (batch_builder) builds the batch exactly as for real data.
4. Main loop of the pipeline, unchanged:
   - SGD_jax.py: SGD1 (position, N, background) + SGD2 (all), fixed starts rho 90 / eta 90 / delta 80.
   - SGD_jax_CNN.py: cnn_first_guess.py (CNN trained at middle_plane 1.4, rotation 7 / rotation2 2; z converted with
     cnn_z_equivalence) -> stream_fit.py per-PSF Levenberg-Marquardt (optim lm, 25 slots, delta held 5 steps,
     delta 2-179 deg, max_rejects 8, tol 1 deg / 10 nm), huber loss (huber_c 2), 6 backgrounds per PSF.
   npz per batch in NPZ_huber\ (as a real run).
5. collect(): found vs truth per PSF (rho wrapped to +-90, eta flipped when rho crosses 0/180, x y z nm, N relative)
   -> results_huber.npz; statistics(): per grid point bias = median error, precision = robust std (1.4826 MAD)
   -> summary.csv, polar maps, log (delta slope found vs true).

## 5. Analysis scripts (after the runs)
- D:\Amaury\simu_precision_claude\unbiased_2026-11-09\analysis\polar_maps_cnn_start.py: usual maps (5-95 %
  colour scales), CNN first guess maps (cnn_first_guess rerun on CPU on simulation.npz).
- ...\analysis\compare_sgd_cnn.py: table (delta slope, paired deviance).
- D:\AMAURY\simu_precision_claude\MFM_fisher\crb_vs_unbiased_sim.py: CRB (Poisson only, no background, torch model of
  simu_PSF_polarMFM.py + fisher_4pMFM.py) vs simulated precision (sqrt(mean over rho of std^2)).
