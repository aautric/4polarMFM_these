# analysis/ - figure and analysis scripts (moved into git on 2026-11-12)

Scripts written by Claude that produced figures sent to the suivi. They were in `D:\Amaury\figures_claude\`
(folder removed on 2026-11-12, its outputs can be regenerated with the commands below). Each script imports the
modules of the repo it is in (not a copy) and writes its outputs into `$OUT_DIR`, or the current folder if unset.
The input data stay on D: (paths at the top of each script).
Checked 2026-11-12: run from git, every output text / npz identical to the one of D:\Amaury\figures_claude.

| folder | script | what it does | input data (D:) |
|---|---|---|---|
| psf_6channels | `psf_6channels.py rho eta delta [z_um] [middle_plane]` | noiseless 6-channel PSF of the model (ideal calibration) next to the dipole and its wobbling cone | none (model only, CPU) |
| registration_projections | `projection_whole_stack_all.py [calibration name ...]` | whole-stack projection registration of the 6 channels for every calibration with registration_z.npz | bead stacks + registration_z.npz of each calibration |
| slb_bias_polar_fit | `slb_analysis_v2.py` | SLB analysis with the corrected bead circle fit (outliers removed): rho bias harmonics, eta boundary, ellipse check, per-run tables (valid version, 2026-11-10) | SLB fits, AMAURY\DATA\0_papier\SLB (read only) |
| slb_bias_polar_fit | `slb_circle_rho_th.py [run]` | steps of the bead circle fit and rho vs theta on the 2 SLB beads of one run (default 2026-11-03_11h46) | same |
| slb_bias_polar_fit | `plot_polar_fit_A.py` | fit of method A of the polarisation calibration of 2026-11-04 (rotating polariser, 37 angles) | simu_precision_claude\polar_calib_2026-11-04\data_AB.npz (calibration_polar_claude.py) |

Example: `OUT_DIR=/mnt/d/AMAURY/some_output python analysis/slb_bias_polar_fit/slb_analysis_v2.py`

`tools/update_claude_lists.py [new folder ...]`: regenerates section 2 of the Desktop list
`claude_files_outside_git.txt` (scripts outside git created by Claude).
