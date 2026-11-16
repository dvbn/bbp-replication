"""Rerun the historical six-method PPML/GPML simulation study."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import fnmatch
import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import platform
from pathlib import Path
import signal
import sys
import sysconfig
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def child(connection, lane, config, native_path):
    """A persistent process per library, with fit-only CPU and wall deadlines."""
    import resource
    import numpy as np
    from threadpoolctl import threadpool_limits
    signal.signal(signal.SIGPROF, signal.SIG_DFL)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        with threadpool_limits(limits=1):
            if lane == 'iols':
                import iols_fit
                backends = tuple(m[5:] for m in config['selected_methods'] if m.startswith('iols_'))
                iols_fit.initialize(config['iols_control'], backends, native_path)
            elif lane == 'statsmodels':
                import statsmodels_fit
                statsmodels_fit.initialize()
            else:
                sys.path.insert(0, str(ROOT / 'src/library'))
                import engine
                import zero_safe_kernel
                zero_safe_kernel.install()
            connection.send(('ready',))
            while True:
                task = connection.recv()
                if task is None:
                    return
                X, y, equation, method, cpu_cap = task
                if lane != 'library':
                    X.setflags(write=False)
                    y.setflags(write=False)
                gc.collect()
                cpu0, wall0 = time.process_time_ns(), time.monotonic_ns()
                connection.send(('started', cpu0, wall0))
                signal.setitimer(signal.ITIMER_PROF, cpu_cap if cpu_cap is not None else 0)
                tau = config['equations'][equation]
                if lane == 'iols':
                    result = iols_fit.fit(X, y, tau, method[5:])
                elif lane == 'statsmodels':
                    result = statsmodels_fit.fit(method, X, y, tau)
                else:
                    backend = 'newton' if method == 'sklearn_damped_newton' else 'lbfgs'
                    result = engine.solve(X, y, tau, backend,
                        maxiter=config['library_control']['maxiter'],
                        maxfun=config['library_control']['maxfun'])
                    result['native_status'] = result['status']
                    result['native_converged'] = result['native_success']
                    result['budget_exhausted'] = result['status'] in ('iteration_limit', 'evaluation_limit')
                signal.setitimer(signal.ITIMER_PROF, 0)
                cpu_ns = time.process_time_ns() - cpu0
                wall_ns = time.monotonic_ns() - wall0
                # Large fitted-index arrays are redundant with X and beta.
                result.pop('index', None)
                connection.send(('result', result, cpu_ns, wall_ns))
    except BaseException:
        signal.setitimer(signal.ITIMER_PROF, 0)
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
        raise


class FitWorker:
    def __init__(self, lane, config, native_path):
        self.lane, self.config, self.native_path = lane, config, native_path
        self.process = None

    @staticmethod
    def reaped_cpu():
        import resource
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime

    def start(self):
        parent, child_connection = mp.get_context('spawn').Pipe()
        self.connection = parent
        self.process = mp.get_context('spawn').Process(target=child,
            args=(child_connection, self.lane, self.config, self.native_path))
        self.process.start()
        child_connection.close()
        if not parent.poll(120):
            self.kill()
            raise RuntimeError('Worker initialization exceeded 120 seconds')
        message = parent.recv()
        if message[0] != 'ready':
            self.kill()
            raise RuntimeError(str(message))

    def kill(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
            self.process.join()
            self.connection.close()
            self.process = None

    def close(self):
        if self.process is not None and self.process.is_alive():
            self.connection.send(None)
            self.process.join(10)
        self.kill()

    def fit(self, X, y, equation, method, enforce_caps):
        if self.process is None:
            self.start()
        cpu_cap = self.config['cpu_cap_seconds'] if enforce_caps else None
        wall_cap = self.config['wall_cap_seconds'] if enforce_caps else None
        reaped_before = self.reaped_cpu()
        self.connection.send((X.copy(), y.copy(), equation, method, cpu_cap))
        if not self.connection.poll(120):
            self.kill()
            raise RuntimeError('Worker did not acknowledge fit')
        message = self.connection.recv()
        if message[0] != 'started':
            self.kill()
            raise RuntimeError(str(message))
        _, cpu0, wall0 = message
        while True:
            if self.connection.poll(.05):
                try:
                    message = self.connection.recv()
                except EOFError:
                    message = None
                if message:
                    if message[0] == 'error':
                        self.kill()
                        raise RuntimeError(message[1])
                    if message[0] != 'result':
                        self.kill()
                        raise RuntimeError(str(message))
                    _, result, cpu_ns, wall_ns = message
                    if enforce_caps and (cpu_ns > cpu_cap * 1e9 or wall_ns > wall_cap * 1e9):
                        # Keep an actual returned root/status and audit it. Eligibility for
                        # the common-budget comparison is decided separately by the parent.
                        result['deadline_reason'] = 'result_after_deadline'
                    return result, cpu_ns, wall_ns
            elapsed = (time.monotonic_ns() - wall0) / 1e9
            if not self.process.is_alive() or (wall_cap is not None and elapsed >= wall_cap):
                was_alive = self.process.is_alive()
                if was_alive:
                    self.process.kill()
                self.process.join()
                exitcode = self.process.exitcode
                cpu = max(0., self.reaped_cpu() - reaped_before - cpu0 / 1e9)
                reason = ('wall_time_limit' if was_alive else 'cpu_time_limit'
                          if exitcode == -signal.SIGPROF else 'unexpected_worker_exit')
                self.connection.close()
                self.process = None
                if reason == 'unexpected_worker_exit':
                    raise RuntimeError('Unexpected worker exit: ' + str(exitcode))
                return dict(beta=None, native_status='time_budget_exhausted',
                    native_converged=False, deadline_reason=reason), round(cpu * 1e9), round(elapsed * 1e9)


def lane_for(method):
    if method.startswith('iols_'):
        return 'iols'
    if method.startswith('statsmodels_'):
        return 'statsmodels'
    return 'library'


def method_order(cell, draw, config, methods):
    import numpy as np
    import designs
    arms = ['iols_numpy', 'iols_cpp', 'statsmodels_newton', 'statsmodels_irls']
    base = designs.seed_key(cell, draw, config['run_split'], config['master_seed'], 'order')
    order = []
    # The original four-arm lane used a separate order sub-key for each equation.
    for index, equation in enumerate(('gpml', 'ppml')):
        rng = np.random.default_rng(np.random.SeedSequence(base + [index]))
        order.extend((equation, arms[int(i)]) for i in rng.permutation(len(arms)))
    library = [(e, m) for e in ('ppml', 'gpml') for m in
               ('sklearn_damped_newton', 'scipy_lbfgs')]
    extra = designs.method_order(cell, draw, config['run_split'], config['master_seed'], library)
    return [(e, m) for e, m in order + extra if m in methods]


def source_digest():
    digest = hashlib.sha256()
    paths = [ROOT / 'run.py', ROOT / 'numeric_summary.py', ROOT / 'execution_schedule.json'] + sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'native').glob('*'))
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def runtime_identity(methods, native_path):
    """Bind resumption to the actual numerical environment and native bytes."""
    versions = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'statsmodels', 'pandas', 'patsy', 'threadpoolctl']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    resolved_native = None
    native_hash = None
    if 'iols_cpp' in methods:
        if native_path is None:
            candidates = sorted((ROOT/'build').glob('_kernel_cpp*.so')) + sorted((ROOT/'build').glob('_kernel_cpp*.pyd'))
            if len(candidates) != 1:
                raise ValueError('Build the C++ kernel first, or pass --native PATH')
            native_path = candidates[0]
        resolved_native = Path(native_path).resolve()
        if not resolved_native.is_file():
            raise ValueError('The selected native module does not exist')
        native_hash = hashlib.sha256(resolved_native.read_bytes()).hexdigest()
    identity = dict(python=platform.python_version(), implementation=sys.implementation.name,
        python_abi=sysconfig.get_config_var('SOABI'), platform=sysconfig.get_platform(),
        machine=platform.machine(), dependency_versions=versions, native_sha256=native_hash)
    return identity, resolved_native


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replications', type=int, help='Prefix of the original 1,000 draw IDs')
    parser.add_argument('--cells', help='Comma-separated cell names or glob patterns')
    parser.add_argument('--methods', help='Comma-separated methods; default all six')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--budget', choices=['historical', 'common', 'uncapped'], default='historical',
                        help='Historical per-fit schedule, all calls capped, or no process timers')
    parser.add_argument('--native', type=Path, help='Explicit compiled _kernel_cpp module')
    args = parser.parse_args(argv)
    if os.name != 'posix' or not hasattr(signal, 'ITIMER_PROF'):
        parser.error('The numerical runner requires Linux/macOS POSIX process timers')
    import numpy as np
    import designs
    import checks
    from threadpoolctl import threadpool_limits
    from summarize import summarize
    from numeric_summary import scalar_targets
    config = json.loads((ROOT / 'config.json').read_text())
    methods = args.methods.split(',') if args.methods else config['methods']
    if len(set(methods)) != len(methods) or not set(methods) <= set(config['methods']):
        parser.error('--methods must be unique method names from config.json')
    replications = args.replications if args.replications is not None else (1 if args.mode == 'smoke' else 1000)
    if not 1 <= replications <= config['draws_per_cell']:
        parser.error('--replications must be between 1 and 1000')
    cells = designs.cells(config)
    if args.cells:
        patterns = args.cells.split(',')
        cells = [c for c in cells if any(fnmatch.fnmatchcase(c['id'], pattern) for pattern in patterns)]
    elif args.mode == 'smoke':
        cells = cells[:1]
    if not cells:
        parser.error('No cells matched')
    try:
        environment, args.native = runtime_identity(methods, args.native)
    except ValueError as error:
        parser.error(str(error))
    scope = dict(mode=args.mode, replications=replications, cells=[c['id'] for c in cells],
                 methods=methods, budget=args.budget)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_file, run_file = output / 'fits.jsonl', output / 'run.json'
    scientific_id = hashlib.sha256((json.dumps(config, sort_keys=True) + source_digest() +
        json.dumps(environment, sort_keys=True) + json.dumps(scope, sort_keys=True)).encode()).hexdigest()
    schedule = json.loads((ROOT / 'execution_schedule.json').read_text())
    capped_keys = set()
    for item in schedule['entries']:
        for first, last in item['draw_ranges']:
            capped_keys.update((item['cell_id'], draw, item['estimator'], item['solver'])
                               for draw in range(first, last + 1))
    done = set()
    if data_file.exists():
        if not args.resume:
            parser.error('Output already contains results; use --resume or a fresh output directory')
        previous = json.loads(run_file.read_text())
        if previous['scientific_id'] != scientific_id:
            parser.error('The numerical source, environment, native module or run scope changed; choose a new output directory')
        with data_file.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row['cell_id'], row['draw'], row['estimator'], row['solver'])
                if key in done:
                    raise RuntimeError('Duplicate result in ' + str(data_file))
                done.add(key)
    metadata = dict(scientific_id=scientific_id, mode=args.mode, master_seed=config['master_seed'],
        split=config['run_split'], replications=replications, cells=[c['id'] for c in cells],
        methods=methods, equations=config['equations'], cpu_cap_seconds=config['cpu_cap_seconds'],
        wall_cap_seconds=config['wall_cap_seconds'], budget=args.budget, environment=environment)
    run_file.write_text(json.dumps(metadata, indent=2) + '\n')
    execution_config = dict(config, selected_methods=methods)
    workers = {lane: FitWorker(lane, execution_config, args.native) for lane in {lane_for(m) for m in methods}}
    try:
        with threadpool_limits(limits=1), data_file.open('a') as stream:
            for cell in cells:
                for draw in range(replications):
                    order = method_order(cell, draw, config, methods)
                    if all((cell['id'], draw, e, m) in done for e, m in order):
                        continue
                    data = designs.generate(cell, draw, config['run_split'], config['master_seed'])
                    context = checks.prepare(data['X'], data['y'], config)
                    for equation, method in order:
                        key = (cell['id'], draw, equation, method)
                        if key in done:
                            continue
                        tau = config['equations'][equation]
                        screen = context['screen_status'][tau]
                        attempted = not screen.startswith('structural')
                        prospective_cap = attempted and (args.budget == 'common' or
                            (args.budget == 'historical' and key in capped_keys))
                        if attempted:
                            result, cpu_ns, wall_ns = workers[lane_for(method)].fit(data['X'], data['y'], equation, method, prospective_cap)
                        else:
                            result = dict(beta=None, native_status=screen, native_converged=False)
                            cpu_ns = wall_ns = 0
                        beta = result.get('beta')
                        audit = checks.evaluate(context, beta, tau, config)
                        common_eligible = (attempted and cpu_ns <= config['cpu_cap_seconds']*1e9
                                           and wall_ns <= config['wall_cap_seconds']*1e9
                                           and result['native_status'] != 'time_budget_exhausted')
                        beta_d, theta, target_status = scalar_targets(beta)
                        row = dict(cell_id=cell['id'], dgp=cell['dgp'], group=cell['group'], n=cell['n'],
                            k=cell['k'], draw=draw, split=config['run_split'], estimator=equation, solver=method,
                            screen_status=screen, fit_attempted=attempted, accepted_1e8=bool(audit['accepted_1e8']),
                            accepted_1e6=bool(audit['accepted_1e6']), cpu_ns=cpu_ns, wall_ns=wall_ns,
                            prospective_cap=prospective_cap, common_cap_eligible=common_eligible,
                            common_cap_accepted_1e8=common_eligible and bool(audit['accepted_1e8']),
                            common_cap_accepted_1e6=common_eligible and bool(audit['accepted_1e6']),
                            beta_d=beta_d, theta=theta, target_status=target_status,
                            beta=beta, beta_truth=cell['beta_d_true'],
                            theta_truth=cell['theta_true'], audit=audit, **{k: v for k, v in result.items() if k != 'beta'})
                        stream.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + '\n')
                        stream.flush()
                        done.add(key)
                    print(cell['id'] + ' draw ' + str(draw) + ': saved', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    summarize(output, config)
    import agreement
    agreement.summarize(output, config)
    print('Results and display inputs: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
