# simu_precision/ - analysis scripts of the precision simulations (SGD_precision_claude.py)

Each script imports the repo it is in. The runs and the outputs are on D: (paths at the top of each script):
runs in `D:\AMAURY\simu_precision_claude\unbiased_2026-11-09\` (renamed `2026-11-09_unbiased_sgd_vs_cnn` after the
merge of the simulation-review branch), CRB cache and figures in `D:\AMAURY\simu_precision_claude\2026-10-28_crb_fisher\`.
The queue that made the runs: `D:\AMAURY\polMFM_runs\simulations\2026-11-09_unbiased_sgd_vs_cnn\run_2026-11-12_nobg.sh`.

| script | what it does |
|---|---|
| `crb_vs_unbiased_sim.py` | CRB (fisher() of fisher_4pMFM.py, torch model, Poisson only, no background, N 2100, delta 60-120, polar_projections [0, -45, 0] = J2 rot(45) of the simulation) vs the simulated precision (sqrt of the mean over rho of std^2) of the nobg_poisson / nobg_emccd runs, SGD and CNN + LM: crb_vs_unbiased_sgd.png, crb_vs_unbiased_cnn.png, .txt |
| `polar_maps.py <run folder> [...]` | polar maps of ANY run of SGD_precision_claude.py: bias and precision (robust std) per (rho, eta) for rho, eta, delta, N, per loss; for a CNN run 2 more rows for the CNN first guess (rerun on CPU with the settings of the run, config.txt; cached cnn_start_<loss>.npz). Outputs in the run folder: polar_maps_<loss>.png / .txt |
| `compare_sgd_cnn.py` | SGD vs CNN + LM table: delta slope found vs true, paired deviance |

Review of these scripts and of the simulation: `review/simulation_precision_fisher_2026-11-12/`; manual:
`docs/manual_simulation_precision.md`.
Example: `python simu_precision/polar_maps.py D:/AMAURY/simu_precision_claude/unbiased_2026-11-09/2026-11-12_13h46_nobg_poisson_cnn` (WSL path /mnt/d/...).
