"""Historical statsmodels Newton/IRLS adaptations with installed dependencies."""

import ast
import gc
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import warnings
import numpy as np
_STATE = {}

SM_CONTROL = dict(maxiter=100000, tol=1e-12, scale=1.0, ridge_factor=1e-10,
                  irls=dict(tol_criterion='params', atol=1e-12, rtol=0.0,
                            wls_method='lstsq'),
                  newton=dict(max_start_irls=0, optim_hessian='oim'))

IRLS_GC_BUDGET_BYTES = 256 * 1024 * 1024

IRLS_GC_MAX_INTERVAL = 4096

IRLS_GC_POLICY = 'irls_only_subclass_stock_update_history_then_gc_collect_every_budgeted_interval'

KNOWN_VALUE_ERRORS = ('estimation infeasible',        # regression/_tools._MinimalWLS
                      'first guess on the deviance')

SELECTED_COPY_SHA256 = {
    'genmod/generalized_linear_model.py':
        'd8c8e2150455194a79a71d477beb30415ea8fb87568301e1ea872def18f98b77',
    'base/optimizer.py':
        'f4e3e6fa3c3a81ae45fa21056f7e956387a97dcd680967614e42fd3d6f4fe0d2',
    'genmod/families/family.py':
        'c327b0c2f1cdd1537aca163492ecf038e5e50e2c0adc70624a631c249bfa7409',
}

_FIT_KEYWORDS = ('start_params', 'maxiter', 'full_output', 'method', 'disp')

def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def _fingerprint(config):
    return sha256_text(json.dumps(config, sort_keys=True, default=str))

def _is_super_fit(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'fit' and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == 'super')

def tol_forward_source(source):
    """Return the tol-forwarding _fit_gradient source, or report upstream already does.

    Only change: `super().fit(..., **kwargs)` -> `super(GLM, self).fit(..., tol=tol,
    **kwargs)`. Explicit super is needed because code compiled outside the class
    body has no __class__ cell. Raises on any unexpected structure.
    """
    original = ast.parse(source)
    tree = ast.parse(source)
    fn = tree.body[0] if len(tree.body) == 1 else None
    if not isinstance(fn, ast.FunctionDef) or fn.name != '_fit_gradient' or fn.decorator_list:
        raise RuntimeError('_fit_gradient: unexpected top-level structure')
    params = [a.arg for a in fn.args.args]
    if params[:1] != ['self'] or 'tol' not in params or getattr(fn.args.kwarg, 'arg', None) != 'kwargs':
        raise RuntimeError('_fit_gradient: unexpected signature %r' % params)
    nodes = list(ast.walk(fn))
    supers = [n for n in nodes if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == 'super']
    fits = [n for n in nodes if _is_super_fit(n)]
    if len(supers) != 1 or len(fits) != 1 or supers[0].args or supers[0].keywords:
        raise RuntimeError('_fit_gradient: expected exactly one zero-argument super().fit call')
    if any(isinstance(n, ast.Name) and (n.id == '__class__' or (n.id == 'tol' and not isinstance(n.ctx, ast.Load)))
           for n in nodes):
        raise RuntimeError('_fit_gradient: __class__ use or tol rebinding')
    call = fits[0]
    names = [k.arg for k in call.keywords]
    base = dict(original_sha256=sha256_text(source))
    if call.args:
        raise RuntimeError('_fit_gradient: positional arguments in super().fit')
    if 'tol' in names:
        value = call.keywords[names.index('tol')].value
        if not (isinstance(value, ast.Name) and value.id == 'tol'):
            raise RuntimeError('_fit_gradient: upstream forwards something other than tol')
        return dict(base, mode='upstream_forwards_tol', patched_source=None, patched_sha256=None)
    star = call.keywords[-1] if call.keywords else None
    if (sorted(n for n in names if n) != sorted(_FIT_KEYWORDS) or names.count(None) != 1
            or names[-1] is not None or not (isinstance(star.value, ast.Name) and star.value.id == 'kwargs')):
        raise RuntimeError('_fit_gradient: super().fit keywords changed: %r' % names)
    supers[0].args = [ast.Name('GLM', ast.Load()), ast.Name('self', ast.Load())]
    call.keywords.insert(len(call.keywords) - 1, ast.keyword(arg='tol', value=ast.Name('tol', ast.Load())))
    patched = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    # Round trip: undoing exactly these two edits must reproduce the original AST.
    back = ast.parse(patched)
    undo = [n for n in ast.walk(back) if _is_super_fit(n)]
    if len(undo) != 1:
        raise RuntimeError('tol patch round trip failed')
    undo[0].func.value.args = []
    undo[0].keywords = [k for k in undo[0].keywords if k.arg != 'tol']
    if ast.dump(back) != ast.dump(original):
        raise RuntimeError('tol patch changed more than the super().fit call')
    return dict(base, mode='patched_forward_tol', patched_source=patched,
                patched_sha256=sha256_text(patched))

def irls_gc_estimated_bytes(n, k):
    """Estimated cyclic bytes retained per IRLS update: an n x K design plus 8 n-vectors."""
    for name, value in (('n', n), ('K', k)):
        if type(value) is not int or value < 1:
            raise ValueError('IRLS design %s must be a positive int, got %r' % (name, value))
    return 8 * n * (k + 8)

def irls_gc_interval(n, k, budget_bytes=IRLS_GC_BUDGET_BYTES):
    """Deterministic collection spacing from design size and the memory budget only."""
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError('IRLS gc budget must be a positive int, got %r' % (budget_bytes,))
    return max(1, min(IRLS_GC_MAX_INTERVAL, budget_bytes // irls_gc_estimated_bytes(n, k)))

def make_irls_glm(base, collect=None, interval_override=None, budget_bytes=IRLS_GC_BUDGET_BYTES):
    """IRLS-only subclass of `base` (GLM): stock _update_history, then periodic gc.collect().

    The interval is selected per model from the initialized exog (n, K) and the budget.
    `interval_override` (positive int) is for fake/unit tests and paired diagnostics ONLY;
    the statsmodels_irls arm never passes it (provenance records the source and
    _fit_statsmodels refuses anything but 'budget'). Only GLM._fit_irls calls
    _update_history (check_irls_source pins that one call site), so a Newton fit on
    another class never reaches this hook. `collect` defaults to gc.collect; tests pass
    fakes. Counters on the instance give the compact provenance exported by
    irls_gc_provenance(); they are reset for every model.
    """
    if interval_override is not None and (type(interval_override) is not int or interval_override < 1):
        raise ValueError('IRLS gc interval override must be a positive int, got %r' % (interval_override,))
    irls_gc_interval(1, 1, budget_bytes)                       # validates the budget
    collect = gc.collect if collect is None else collect

    class IRLSGLM(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            n, k = self.exog.shape
            self._bbp_gc_estimated_bytes = irls_gc_estimated_bytes(int(n), int(k))
            self._bbp_gc_budget_bytes = budget_bytes
            self._bbp_gc_source = 'budget' if interval_override is None else 'override'
            self._bbp_gc_interval = (irls_gc_interval(int(n), int(k), budget_bytes)
                                     if interval_override is None else interval_override)
            self._bbp_history_updates = 0
            self._bbp_gc_collections = 0
            self._bbp_gc_unreachable = 0

        def _update_history(self, tmp_result, mu, history):
            history = super()._update_history(tmp_result, mu, history)
            self._bbp_history_updates += 1
            if self._bbp_history_updates % self._bbp_gc_interval == 0:
                self._bbp_gc_collections += 1
                self._bbp_gc_unreachable += int(collect())
            return history

    return IRLSGLM

def irls_gc_provenance(model, iterations=None):
    """Compact scalar provenance of one IRLS fit; fails closed on a broken schedule
    or when the reported iteration count differs from the observed history updates."""
    updates, collections = int(model._bbp_history_updates), int(model._bbp_gc_collections)
    interval, estimated = int(model._bbp_gc_interval), int(model._bbp_gc_estimated_bytes)
    budget, source = int(model._bbp_gc_budget_bytes), str(model._bbp_gc_source)
    if collections != updates // interval:
        raise RuntimeError('IRLS gc schedule violated: %d updates, %d collections, interval %d'
                           % (updates, collections, interval))
    if source == 'budget' and interval != max(1, min(IRLS_GC_MAX_INTERVAL, budget // estimated)):
        raise RuntimeError('IRLS gc interval %d does not follow the budget policy' % interval)
    if iterations is not None and updates != int(iterations):
        raise RuntimeError('IRLS history updates (%d) differ from reported iterations (%d)' % (updates, iterations))
    return dict(irls_gc_policy=IRLS_GC_POLICY, irls_gc_interval_source=source, irls_gc_interval=interval,
                irls_gc_estimated_bytes_per_iteration=estimated, irls_gc_budget_bytes=budget,
                irls_history_updates=updates, irls_gc_collections=collections,
                irls_gc_unreachable_total=int(model._bbp_gc_unreachable))

def _start(X, y, scale):
    beta = np.zeros(X.shape[1])
    if _STATE['start'] == 'mean_intercept':
        ic = _STATE['intercept']
        if ic is None or not np.all(X[:, ic] == 1.0) or (scale is not None and scale[ic] != 1.0):
            raise ValueError('mean_intercept start needs an all-ones intercept column')
        ybar = y.mean()
        if ybar <= 0:
            raise ValueError('all-zero response reached fit; it must be screened out')
        beta[ic] = np.log(ybar)
    return beta

def _fit_statsmodels(method, X, y, tau, start):
    sm, control, cap = _STATE['sm'], SM_CONTROL, _STATE['str_cap']
    family = sm['Poisson'](link=sm['Log']()) if tau == 1 else sm['ZeroGPML']()
    details = dict(start_rule=_STATE['start'], start_comparability='software_default_not_same_trajectory',
                   control_sha256=_fingerprint(SM_CONTROL)[:16])
    converged, iterations, beta, status, irls_model = False, None, None, None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            if method == 'statsmodels_irls':
                irls_model = sm['IRLSGLM'](y, X, family=family)
                result = irls_model.fit(
                    start_params=start, maxiter=control['maxiter'], method='IRLS', tol=control['tol'],
                    scale=control['scale'], **control['irls'])
                converged = bool(result.converged)
                iterations = int(result.fit_history['iteration'])
            else:
                model = sm['NewtonGLM'](y, X, family=family)
                model._bbp_mle_calls = []
                result = model.fit(start_params=start, maxiter=control['maxiter'], method='newton',
                                   tol=control['tol'], scale=control['scale'], **control['newton'])
                details.update(_check_mle_settings(model._bbp_mle_calls))
                retvals = result.mle_retvals
                converged = bool(retvals['converged'])
                iterations = int(retvals['iterations'])
                details.update(warnflag=int(retvals['warnflag']),
                               library_score_maxabs=float(np.max(np.abs(retvals['score']))))
            beta = np.array(result.params, dtype=float)
        except np.linalg.LinAlgError as error:
            status = 'exception_linalg'
            details['exception'] = repr(error)[:cap]
        except ValueError as error:
            if not any(text in str(error) for text in KNOWN_VALUE_ERRORS):
                raise
            status = 'exception_infeasible_value'
            details['exception'] = repr(error)[:cap]
    if irls_model is not None:
        # Recorded on every path, including library exceptions (updates then count
        # the iterations completed before the failure; iterations is None there).
        details.update(irls_gc_provenance(irls_model, iterations))
        if details['irls_gc_interval_source'] != 'budget':
            raise RuntimeError('statsmodels_irls must use the budget-derived gc interval')
    misuse = [w for w in caught if issubclass(w.category, FutureWarning)
              and 'passed to the optimizer' in str(w.message)]
    if misuse:
        raise RuntimeError('statsmodels ignored optimizer keywords: %s' % misuse[0].message)
    details['warning_count'] = len(caught)
    details['warnings'] = ' | '.join(sorted({'%s: %s' % (w.category.__name__, str(w.message)[:120])
                                             for w in caught}))[:cap]
    if status is None:
        # Both statsmodels loops leave unconverged only after maxiter iterations.
        status = ('converged' if converged else 'maxiter' if iterations is not None
                  and iterations >= control['maxiter'] else 'not_converged_below_maxiter')
        if beta.shape != (X.shape[1],):
            raise RuntimeError('statsmodels returned beta of shape %r' % (beta.shape,))
        if not np.isfinite(beta).all():
            details['nonfinite_beta_count'] = int((~np.isfinite(beta)).sum())
            status += '_nonfinite_beta'
            beta = None
    # native_converged stays the library's own flag even when beta is nonfinite.
    return dict(details, beta=beta, native_converged=converged, native_status=status,
                iterations=iterations, budget_exhausted=status.startswith('maxiter'))

def _check_mle_settings(calls):
    """Observed optimizer settings must be exactly the frozen control (one call)."""
    if len(calls) != 1:
        raise RuntimeError('expected one LikelihoodModel.fit call, saw %d' % len(calls))
    used = calls[0]
    expected = dict(optimizer='newton', tol=SM_CONTROL['tol'], maxiter=SM_CONTROL['maxiter'],
                    ridge_factor=SM_CONTROL['ridge_factor'])
    wrong = {k: used.get(k) for k, v in expected.items() if used.get(k) != v}
    if wrong:
        raise RuntimeError('statsmodels Newton ran with %r, expected %r' % (wrong, expected))
    return dict(mle_tol=float(used['tol']), mle_ridge_factor=float(used['ridge_factor']),
                tol_forward_mode=_STATE['sm']['mode'])

def initialize():
    import statsmodels
    from statsmodels.base import model as base_model
    from statsmodels.genmod import generalized_linear_model as glm_mod
    from statsmodels.genmod.families import family as fam_mod, links
    if statsmodels.__version__ != "0.15.0":
        raise RuntimeError("This study requires statsmodels==0.15.0")
    root = Path(statsmodels.__file__).resolve().parent
    for relative, expected in SELECTED_COPY_SHA256.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("The installed statsmodels source differs from the study version: " + relative)
    GLM = glm_mod.GLM
    patch = tol_forward_source(textwrap.dedent(inspect.getsource(GLM._fit_gradient)))
    class _CaptureMLE(base_model.LikelihoodModel):
            """Records the optimizer settings LikelihoodModel.fit actually used; no other effect."""
            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs)
                self._bbp_mle_calls.append(dict(result.mle_settings))
                return result
    class NewtonGLM(GLM, _CaptureMLE):
            pass
    if patch["mode"] == "patched_forward_tol":
        namespace = dict(GLM._fit_gradient.__globals__)
        exec(compile(patch["patched_source"], "<BBP Newton tolerance forwarding>", "exec"), namespace)
        NewtonGLM._fit_gradient = namespace["_fit_gradient"]
    IRLSGLM = make_irls_glm(GLM)
    class ZeroGPML(fam_mod.Gamma):
            """GPML quasi-family: Gamma variance, log link, loss sum(y/mu + log mu).

            Not the Gamma density. Zero outcomes are retained; y/mu is evaluated only
            where y > 0, so no clipping and no 0 * inf. GLM.score/hessian are left
            stock: for this link/variance they give y/mu - 1 and -y/mu, which are the
            derivatives of loglike_obs below w.r.t. the index.
            """
            def __init__(self):
                super().__init__(link=links.Log())

            @staticmethod
            def _ratio(endog, mu):
                endog, mu = np.broadcast_arrays(np.asarray(endog, float), np.asarray(mu, float))
                ratio = np.zeros(mu.shape)
                positive = endog > 0
                ratio[positive] = endog[positive] / mu[positive]
                return ratio

            def loglike_obs(self, endog, mu, var_weights=1.0, scale=1.0):
                return -(var_weights / scale) * (self._ratio(endog, mu) + np.log(mu))

            def _resid_dev(self, endog, mu):
                # Signed quasi-loss x2; enters only IRLS history (tol_criterion='params').
                return 2.0 * (self._ratio(endog, mu) + np.log(mu))

            def resid_dev(self, *args, **kwargs):
                raise NotImplementedError('quasi-loss, not deviance residuals')

            def get_distribution(self, *args, **kwargs):
                raise NotImplementedError('no distribution for the GPML quasi-family')
    _STATE.update(start="mean_intercept", intercept=0, str_cap=200,
        sm=dict(GLM=GLM, IRLSGLM=IRLSGLM, NewtonGLM=NewtonGLM, ZeroGPML=ZeroGPML,
                Poisson=fam_mod.Poisson, Log=links.Log, mode=patch["mode"]))


def fit(method, X, y, tau):
    return _fit_statsmodels(method, X, y, tau, _start(X, y, None))

