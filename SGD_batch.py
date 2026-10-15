# -*- coding: utf-8 -*-
"""
Runs SGD_jax.py without any interaction: no dialog, the figures are saved as png files, for one data set or a queue
of data sets run one after the other (e.g. over several days, started with nohup).

SGD_jax.py itself is executed, cell by cell (cells separated by '#%%' lines), so that a batch run uses exactly the
same code as an interactive run. The parameters given on the command line replace the corresponding top-level
assignments of SGD_jax.py before it is executed (so that what depends on them, e.g. J_dichroic on rotation, follows).

usage (with the python of the environment, e.g. /home/amaury/exit/envs/env_linux/bin/python):
    python SGD_batch.py run <RAW_DATA folder> [name=value ...] [--no-figures] [--post-process]
        one run on the data of the folder (the one that contains 'reconstruction'), with the parameters of
        SGD_jax.py changed as given, e.g. loss2=poisson rotation=5 rho_offset=-6 batch_nb=500
    python SGD_batch.py resume <NPZ folder> [--no-figures] [--post-process]
        continues a run that stopped, with the parameters of its config.txt (as the resume cell of SGD_jax.py)
    python SGD_batch.py queue <jobs file> [--no-figures] [--post-process] [--dry-run]
        runs the jobs of the file one after the other, each in its own process, a log file per job. All the jobs
        are checked before the first one starts; a job that fails is reported and the queue goes on.
        --dry-run only checks the jobs.

jobs file: one job per line, empty lines and lines starting with '#' are ignored
    <RAW_DATA folder> name=value name=value ...
    resume <NPZ folder>

values: python literals (5, -6.2, True, 'lms'); a bare word is a variable of SGD_jax.py if there is one with this
name (J1=J2), else a string (loss2=poisson); anything else is a python expression evaluated in SGD_jax.py
(e.g. xy_speed2=1/70). No space inside a value.

To run a queue in the background, independently of the terminal:
    nohup python SGD_batch.py queue jobs.txt --post-process > queue.out 2>&1 &
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg') # before matplotlib is imported by SGD_jax.py: no window
import sys
import re
import ast
import time
import subprocess
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
SGD_SCRIPT = HERE / 'SGD_jax.py'
POST_PROCESS_SCRIPT = HERE / 'post_process_SGD.py'
CELL_HEADER = re.compile(r'^# ?%%', re.M)


def log(message):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}', flush=True)


# ---------------------------------------------------------------- reading and changing the script

def read_script(path):
    return path.read_text(encoding='utf-8').replace('\r\n', '\n')


def value_expression(value, script_names=()):
    # python expression written in the script for a value given on the command line
    try:
        ast.literal_eval(value)
        expression = value
    except (ValueError, SyntaxError):
        if re.fullmatch(r'[A-Za-z_]\w*', value) and value not in script_names:
            expression = repr(value) # bare word that is not a variable of the script: string
        else:
            expression = value # expression evaluated in the script
    compile(expression, '<value>', 'eval') # raises SyntaxError if not a valid expression
    return expression


def apply_overrides(source, overrides):
    # replaces the top-level assignments 'name = ...' (one or several lines) of the source by 'name = value';
    # the line numbers are kept (blank lines) so that the errors point to the right lines of SGD_jax.py
    tree = ast.parse(source)
    lines = source.split('\n')
    replacements = []
    # names assigned or imported at the top level of the script: a bare word equal to one of them is that variable
    script_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                script_names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            script_names |= {(a.asname or a.name).split('.')[0] for a in node.names}
    for name, value in overrides.items():
        found = [node for node in tree.body
                 if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
                 or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name)]
        if not found:
            raise ValueError(f'{name}: no top-level assignment "{name} = ..." in {SGD_SCRIPT.name}')
        for node in found:
            if isinstance(node, ast.Assign) and len(node.targets) > 1:
                raise ValueError(f'{name}: assigned together with other names (line {node.lineno}), not supported')
            replacements.append((node.lineno, node.end_lineno, f'{name} = {value_expression(value, script_names)} # set by SGD_batch.py'))
    for start, end, text in sorted(replacements, reverse=True):
        lines[start-1:end] = [text] + ['']*(end-start)
    new_source = '\n'.join(lines)
    ast.parse(new_source) # the changed script must still be valid
    return new_source


def split_cells(source):
    # list of (first line number, header, text) of the cells, the text before the first header is a cell too
    starts = [0] + [m.start() for m in CELL_HEADER.finditer(source)]
    cells = []
    for i, start in enumerate(starts):
        end = starts[i+1] if i+1 < len(starts) else len(source)
        text = source[start:end]
        if text.strip():
            cells.append((source.count('\n', 0, start), text.split('\n', 1)[0], text))
    return cells


def parse_overrides(arguments):
    overrides = {}
    for argument in arguments:
        if '=' not in argument:
            raise ValueError(f'"{argument}" is not of the form name=value')
        name, value = argument.split('=', 1)
        if not re.fullmatch(r'[A-Za-z_]\w*', name) or value == '':
            raise ValueError(f'"{argument}" is not of the form name=value')
        overrides[name] = value
    return overrides


# ---------------------------------------------------------------- headless replacements

class FakeDialog:
    # answers the dialogs of the scripts
    def __init__(self, directory, resume_folder=None):
        self.directory, self.resume_folder = str(directory), resume_folder

    def askdirectory(self, title='', **kwargs):
        if 'resume' in title.lower():
            return str(self.resume_folder)
        return self.directory

    def askopenfilename(self, title='', **kwargs):
        raise RuntimeError(f'dialog "{title}": the reload cell is not used in batch mode, give the parameters instead')


def make_show(namespace, save_figures):
    # replaces plt.show: saves the open figures in <run folder>/figures (or not) and closes them
    counter = [0]
    def show(*args, **kwargs):
        plt = namespace['plt']
        if save_figures:
            folder = Path(namespace['save_folder']) / 'figures'
            folder.mkdir(exist_ok=True)
            for number in plt.get_fignums():
                plt.figure(number).savefig(folder / f'{counter[0]:04d}.png', dpi=80)
                counter[0] += 1
        plt.close('all')
    return show


def run_cells(script, cells, namespace, skip, dialog, show_factory, after_cell=None):
    for first_line, header, text in cells:
        if any(word in header for word in skip):
            log(f'skipped cell: {header.strip()}')
            continue
        code = compile('\n'*first_line + text, str(script), 'exec')
        exec(code, namespace)
        # after the imports: headless dialogs, figures, no progress bars
        if 'filedialog' in namespace and not isinstance(namespace['filedialog'], FakeDialog):
            namespace['filedialog'] = dialog
        if 'plt' in namespace and getattr(namespace['plt'].show, '__name__', '') != 'show_headless':
            show = show_factory(namespace)
            show.__name__ = 'show_headless'
            namespace['plt'].show = show
        if 'tqdm' in namespace and namespace['tqdm'] is not iter:
            namespace['tqdm'] = iter
        if after_cell is not None:
            after_cell(header, namespace)


# ---------------------------------------------------------------- one run

def channel_size(data_folder):
    # [height, width] of the channels of the reconstructed frames (files <frame>.tif with 6 pages)
    import tifffile
    files = [f for f in (Path(data_folder) / 'reconstruction').glob('*.tif') if f.stem.isdigit()]
    if not files:
        raise FileNotFoundError(f'no <frame>.tif file in {Path(data_folder) / "reconstruction"}')
    with tifffile.TiffFile(min(files, key=lambda f: int(f.stem))) as tif:
        return list(tif.pages[0].shape)


def run_sgd(data_folder, overrides, resume_folder=None, save_figures=True, post_process=False):
    data_folder = Path(data_folder)
    if not (data_folder / 'reconstruction').is_dir():
        raise FileNotFoundError(f'{data_folder / "reconstruction"} does not exist')
    source = read_script(SGD_SCRIPT)
    if resume_folder is None and 'dimensions' not in overrides:
        # size of the channels of the reconstruction, read on the data (it changes between data sets)
        overrides = dict(overrides, dimensions=str(channel_size(data_folder)))
        log(f'dimensions read on the data: {overrides["dimensions"]}')
    if overrides:
        source = apply_overrides(source, overrides)
    cells = split_cells(source)
    namespace = {'__name__': '__main__', '__file__': str(SGD_SCRIPT)}
    dialog = FakeDialog(data_folder, resume_folder)
    skip = ['OPTIONAL - reloading'] + ([] if resume_folder else ['OPTIONAL - resuming'])

    def after_cell(header, ns):
        if 'PARAMETERS TO BE DEFINED' in header:
            # two runs started the same minute on the same data would share the NPZ folder: seconds added
            folder = Path(ns['save_folder'])
            if any(folder.iterdir()):
                ns['save_folder'] = folder.with_name(folder.name + f'_{datetime.now():%S}s')
                ns['save_folder'].mkdir(parents=True, exist_ok=False)
                log(f'{folder.name} already used, saving in {ns["save_folder"]}')

    log(f'data: {data_folder}')
    log(f'parameters changed: {overrides if overrides else "none"}' + (f', resuming {resume_folder}' if resume_folder else ''))
    start = time.time()
    end_of_data = False
    try:
        run_cells(SGD_SCRIPT, cells, namespace, skip, dialog, lambda ns: make_show(ns, save_figures), after_cell)
    except FileNotFoundError as error:
        # the gradient descent reads the frames until a file is missing: normal end of the data
        if 'File not found' in str(error) and '/reconstruction/' in str(error).replace('\\', '/'):
            end_of_data = True
            log(f'end of the data: {error}')
        else:
            raise
    save_folder = Path(namespace['save_folder'])
    n_npz = len([f for f in save_folder.glob('*.npz') if f.stem.isdigit()])
    log(f'run finished in {(time.time()-start)/3600:.2f} h ({"end of the data" if end_of_data else "batch_nb reached"}), '
        f'{n_npz} npz files in {save_folder}')
    if post_process:
        run_post_process(save_folder)
    return save_folder


def run_post_process(npz_folder):
    # post_process_SGD.py on the NPZ folder: writes fit_<date>.csv next to it
    log(f'post-processing {npz_folder}')
    source = read_script(POST_PROCESS_SCRIPT)
    namespace = {'__name__': '__main__', '__file__': str(POST_PROCESS_SCRIPT)}
    run_cells(POST_PROCESS_SCRIPT, split_cells(source), namespace, [], FakeDialog(npz_folder),
              lambda ns: make_show(ns, False))
    log(f'csv written: {namespace.get("storing_folder")}')


# ---------------------------------------------------------------- queue

def read_jobs(jobs_file):
    jobs = []
    for number, line in enumerate(Path(jobs_file).read_text(encoding='utf-8').splitlines(), 1):
        words = line.split()
        if not words or words[0].startswith('#'):
            continue
        if words[0] == 'resume':
            if len(words) != 2:
                raise ValueError(f'line {number}: "resume <NPZ folder>" expected')
            jobs.append({'line': number, 'resume': Path(words[1]), 'data': Path(words[1]).parent, 'overrides': {}})
        else:
            jobs.append({'line': number, 'resume': None, 'data': Path(words[0]), 'overrides': parse_overrides(words[1:])})
    return jobs


def check_job(job, source):
    # everything that can be checked before starting: folders, parameters
    if not (job['data'] / 'reconstruction').is_dir():
        raise FileNotFoundError(f'{job["data"] / "reconstruction"} does not exist')
    if job['resume'] is not None and not (job['resume'] / 'config.txt').is_file():
        raise FileNotFoundError(f'{job["resume"] / "config.txt"} does not exist')
    if job['overrides']:
        apply_overrides(source, job['overrides'])


def run_queue(jobs_file, flags, dry_run=False):
    jobs = read_jobs(jobs_file)
    source = read_script(SGD_SCRIPT)
    problems = []
    for job in jobs:
        try:
            check_job(job, source)
        except Exception as error:
            problems.append(f'line {job["line"]}: {error}')
    log(f'{len(jobs)} jobs in {jobs_file}')
    for job in jobs:
        log(f'  line {job["line"]}: ' + (f'resume {job["resume"]}' if job['resume'] else
                                          f'{job["data"]} {" ".join(k+"="+v for k, v in job["overrides"].items())}'))
    if problems:
        for problem in problems:
            log('ERROR ' + problem)
        raise SystemExit('the jobs file has errors, nothing was started')
    if dry_run:
        log('dry run: all the jobs are valid')
        return
    log_folder = Path(jobs_file).resolve().parent / 'sgd_queue_logs'
    log_folder.mkdir(exist_ok=True)
    for i, job in enumerate(jobs, 1):
        if job['resume']:
            command = [sys.executable, str(Path(__file__).resolve()), 'resume', str(job['resume'])] + flags
        else:
            command = ([sys.executable, str(Path(__file__).resolve()), 'run', str(job['data'])]
                       + [f'{k}={v}' for k, v in job['overrides'].items()] + flags)
        log_file = log_folder / f'{datetime.now():%Y-%m-%d_%Hh%M}_job{i:02d}_line{job["line"]}.log'
        log(f'job {i}/{len(jobs)} started, log: {log_file}')
        start = time.time()
        with open(log_file, 'w') as output:
            status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        log(f'job {i}/{len(jobs)} ' + ('done' if status == 0 else f'FAILED (exit code {status}), see the log')
            + f' after {(time.time()-start)/3600:.2f} h')
    log('queue finished')


# ---------------------------------------------------------------- command line

def main(arguments):
    flags = [a for a in arguments if a.startswith('--')]
    words = [a for a in arguments if not a.startswith('--')]
    unknown = set(flags) - {'--no-figures', '--post-process', '--dry-run'}
    if len(words) < 2 or words[0] not in ('run', 'resume', 'queue') or unknown:
        raise SystemExit(__doc__)
    save_figures, post_process = '--no-figures' not in flags, '--post-process' in flags
    if words[0] == 'run':
        run_sgd(words[1], parse_overrides(words[2:]), save_figures=save_figures, post_process=post_process)
    elif words[0] == 'resume':
        if len(words) != 2:
            raise SystemExit('resume takes only the NPZ folder: the parameters are the ones of its config.txt')
        resume_folder = Path(words[1])
        run_sgd(resume_folder.parent, {}, resume_folder=resume_folder, save_figures=save_figures,
                post_process=post_process)
    else:
        run_queue(words[1], [f for f in flags if f != '--dry-run'], dry_run='--dry-run' in flags)


if __name__ == '__main__':
    main(sys.argv[1:])
