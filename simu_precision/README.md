# simu_precision/ - precision simulations of the 4polarMFM fit and Cramer-Rao bound

Each script imports the repo it is in (root added to the path). Data and outputs stay on D:
(`D:\AMAURY\simu_precision_claude\`); the run files of the queues (.sh, logs) in `D:\AMAURY\polMFM_runs\simulations\`.

| script | what it does | command |
|---|---|---|
| `SGD_precision.py` | simulates PSF with known truth (grid of rho, eta; camera noise presets) and fits them with the real pipeline (SGD_jax.py, or SGD_jax_CNN.py with --cnn); one run folder per call (simulation.npz, results_<loss>.npz, summary.csv, maps). Was SGD_precision_claude.py at the root. | `python simu_precision/SGD_precision.py --preset poisson --N 2100 --background 0 --sigma-b 0 --out <folder> --tag <tag> [--cnn]` |
| `polar_maps.py` | polar maps of any run: bias and precision (robust std) per (rho, eta) for rho, eta, delta, N, per loss; for a CNN run, 2 more rows for the CNN first guess (rerun on CPU with the settings of the run, cached cnn_start_<loss>.npz). Outputs in the run folder. | `python simu_precision/polar_maps.py <run folder> [...]` |
| `compare_sgd_cnn.py` | table SGD_jax.py vs CNN + LM on pairs of runs with the same simulated PSF: bias / precision, delta slope, paired deviance | `python simu_precision/compare_sgd_cnn.py <SGD run> <CNN run> [...] --out <file.txt>` |
| `crb_vs_simulation.py` | CRB (Poisson only, no background, conditions of each run) vs the simulated precision (plain std, robust std, outliers) of any runs, one figure per method | `python simu_precision/crb_vs_simulation.py <run folder> [...] --out <folder>` |
| `MFM_fisher.py` | Fisher / CRB study of the 4polarMFM (script version of MFM_fisher.ipynb): CRB vs orientation, ratiometric bound, Monte-Carlo checks. Was MFM_fisher_claude.py at the root. | `python simu_precision/MFM_fisher.py` (cells) |

Example (WSL paths): `python simu_precision/polar_maps.py /mnt/d/AMAURY/simu_precision_claude/unbiased_2026-11-09/2026-11-12_13h46_nobg_poisson_cnn`

Runs of 2026-11-12 (no background, Poisson / Poisson + read + excess, SGD and CNN): `D:\AMAURY\simu_precision_claude\unbiased_2026-11-09\`
(renamed `2026-11-09_unbiased_sgd_vs_cnn` after the merge of the simulation-review branch); CRB figures in
`D:\AMAURY\simu_precision_claude\2026-10-28_crb_fisher\` (crb_vs_simulation_sgd.png, _cnn.png, .txt).

Review of these scripts: `review/simulation_precision_fisher_2026-11-12/`; manual: `docs/manual_simulation_precision.md`;
removed notebooks (SGD_simu_precision*.ipynb, MFM_fisher.ipynb): `docs/removed_files.md`.
