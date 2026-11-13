# Files removed from the repo (still in the git history)

Removed on 2026-11-12 (user request, branch simulation-review). Every file was committed and unchanged before the
removal, so it can be read or restored from the last commit that has it, `6e07c7f` (or the "last change" commit):

    git show 6e07c7f:<path>                 # print one file
    git checkout 6e07c7f -- <path>          # restore a file or a folder in the working tree

| path | files | last change | what it was |
|---|---|---|---|
| SGD_simu_precision.ipynb | 1 | 0262ce0 (2026-07-07) | first precision-simulation notebook |
| SGD_simu_precision2.ipynb | 1 | a8ab8da (2026-10-15) | precision-simulation notebook (camera noise, polar maps), replaced by simu_precision/SGD_precision.py |
| SGD_simu_precision_loss.ipynb | 1 | 0f57879 (2026-07-29) | precision vs loss notebook, replaced by SGD_precision.py --losses |
| SGD_script.py | 1 | a01a51e (2026-08-10) | old fit script |
| MFM_fisher.ipynb | 1 | 7fa6725 (2026-08-26) | Fisher / CRB notebook (old version), replaced by simu_precision/MFM_fisher.py |
| notebooks_experimental_SLB_1um_janvier_2026/ | 3 | fb633a7 (2026-11-10) | SLB notebooks of January 2026 |
| notebooks_plot_simu_stage/ | 7 | a01a51e (2026-08-10) | plots of the internship simulations |
| old_stuff_1st_year/ | 54 | 585dfd4 (2026-11-05) | first-year code |
| analysis/ | 6 | f0a5aa3 (2026-11-13) | Claude's analysis scripts (psf_6channels, registration_projections, slb_bias_polar_fit); removed 2026-11-13 (user), kept in D:\AMAURY\analysis_claude\ (restore with git checkout f0a5aa3 -- analysis) |

Not in git (never committed, .gitignore): 23 Jupyter autosave copies (.ipynb_checkpoints) and .pyc files inside these
folders, in the main checkout only.
