# simu_precision/ - analysis scripts of the precision simulations (SGD_precision_claude.py)

Each script imports the repo it is in. The runs and the outputs are on D: (paths at the top of each script):
runs in `D:\AMAURY\simu_precision_claude\unbiased_2026-11-09\` (renamed `2026-11-09_unbiased_sgd_vs_cnn` after the
merge of the simulation-review branch), CRB cache and figures in `D:\AMAURY\simu_precision_claude\2026-10-28_crb_fisher\`.
The queue that made the runs: `D:\AMAURY\polMFM_runs\simulations\2026-11-09_unbiased_sgd_vs_cnn\run_2026-11-12_nobg.sh`.

| script | what it does |
|---|---|
| `crb_vs_unbiased_sim.py` | CRB (fisher() of fisher_4pMFM.py, torch model, Poisson only, no background, N 2100, delta 60-120, polar_projections [0, -45, 0] = J2 rot(45) of the simulation) vs the simulated precision (sqrt of the mean over rho of std^2) of the nobg_poisson / nobg_emccd runs, SGD and CNN + LM: crb_vs_unbiased_sgd.png, crb_vs_unbiased_cnn.png, .txt |
| `polar_maps_cnn_start.py` | usual polar maps (bias, precision; colour bars 5-95 %) + 2 rows for the CNN first guess (cnn_first_guess rerun on CPU on simulation.npz) |
| `compare_sgd_cnn.py` | SGD vs CNN + LM table: delta slope found vs true, paired deviance |

Review of these scripts and of the simulation: `review/simulation_precision_fisher_2026-11-12/`; manual:
`docs/manual_simulation_precision.md`.
Open points (review list): lambda 617 in the CRB (simulation 620), CASES lists of the two map scripts still name the
deleted case1 / case3 runs, plain vs robust std against the CRB (item 6).
