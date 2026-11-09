"""No-fixed-effect DGPs for the PPML/GPML solver comparison.

The 16 ordinary designs and four stress designs preserve the original
covariate and outcome laws. Every design has E(Y|X)=exp(X beta), an
independent Bernoulli(1/2) treatment with coefficient log(1.08), and
population untreated mean 2. Draws are never redrawn, clipped or trimmed.
"""
from __future__ import annotations

import hashlib
import json
import math

TREATMENT_COEFFICIENT = math.log(1.08)
TREATMENT_COLUMN = 1
UNTREATED_MEAN = 2.
SPLITS = {"development": 1, "validation": 2, "test": 3}
STREAMS = {"covariates": 0, "outcome": 1, "order": 2}
SEED_RECIPE = ("numpy.random.SeedSequence([master_seed, split_code, dgp_number, n, k, draw, "
               "stream_code]); split_code development=1 validation=2 test=3; "
               "stream_code covariates=0 outcome=1 order=2")
# Master seeds of earlier campaigns (CP11 dgp11.MASTER, CP8 CAMPAIGN_MASTER_SEED,
# DMC MASTER_SEED). Their entropy lists also have different lengths (4 or 5).
HISTORICAL_MASTER_SEEDS = (2026092111, 2026092808, 20260926)

_ROOT = ""
CP11 = "ordinary Gaussian design"
CP8 = "mixed-covariate design"
CAL = "outcome-law specification"
DMC = "stress-design specification"


def _dgp(number, dgp_id, group, covariates, law, source, **parameters):
    return dict(number=number, id=dgp_id, group=group, covariates=covariates,
                law=law, source=source, parameters=parameters)


_G = f"{CP11} menu G (Gaussian nuisances, index SD .6)"
DGPS = (
    # CP11 continuity panel ("ordinary"): dgp11.LAWS on menu G, then A/B/C x NB2 1/.2.
    _dgp(1, "G_poisson", "ordinary", "cp11_gaussian", "poisson", f"{_G}; law {CAL}"),
    _dgp(2, "G_gamma", "ordinary", "cp11_gaussian", "gamma", f"{_G}; law {CAL} (shape 1)"),
    _dgp(3, "G_nb2_mild", "ordinary", "cp11_gaussian", "nb2", f"{_G}; dgp11.outcomes", size=5.),
    _dgp(4, "G_nb2", "ordinary", "cp11_gaussian", "nb2", f"{_G}; dgp11.outcomes", size=1.),
    _dgp(5, "G_nb2_strong", "ordinary", "cp11_gaussian", "nb2", f"{_G}; dgp11.outcomes", size=.2),
    _dgp(6, "G_tweedie15", "ordinary", "cp11_gaussian", "tweedie15",
         f"{_G}; law {CAL} (compound Poisson-Gamma, variance 2 mu^1.5; outcome DGP only)"),
    _dgp(7, "G_ig3", "ordinary", "cp11_gaussian", "ig3", f"{_G}; law {CAL} (Wald, variance mu^3)"),
    _dgp(8, "G_heterogeneous_nb2", "ordinary", "cp11_gaussian", "heterogeneous_nb2",
         f"{_G}; dgp11.outcomes, H = 1[first nuisance > 0]", size_h1=1/3, size_h0=1.),
    _dgp(9, "G_lognormal", "ordinary", "cp11_gaussian", "lognormal_cp11",
         f"{_G}; dgp11.outcomes", residual_sd=1.),
    _dgp(10, "G_zero_lognormal", "ordinary", "cp11_gaussian", "zero_lognormal_cp11",
         f"{_G}; dgp11.outcomes", residual_sd=1., positive_probability=.5),
    _dgp(11, "A_nb2", "ordinary", "cp8_menu", "nb2", f"{CP11}; menu A from {CP8}", menu="A", size=1.),
    _dgp(12, "A_nb2_strong", "ordinary", "cp8_menu", "nb2", f"{CP11}; menu A from {CP8}", menu="A", size=.2),
    _dgp(13, "B_nb2", "ordinary", "cp8_menu", "nb2", f"{CP11}; menu B from {CP8}", menu="B", size=1.),
    _dgp(14, "B_nb2_strong", "ordinary", "cp8_menu", "nb2", f"{CP11}; menu B from {CP8}", menu="B", size=.2),
    _dgp(15, "C_nb2", "ordinary", "cp8_menu", "nb2", f"{CP11}; menu C from {CP8}", menu="C", size=1.),
    _dgp(16, "C_nb2_strong", "ordinary", "cp8_menu", "nb2", f"{CP11}; menu C from {CP8}", menu="C", size=.2),
    # Intentional stress mechanisms copied from DMC no-FE cells. DMC used n=1000,
    # K=25; here they use the CP11 sizes, with DMC's dimension-free index SD.
    _dgp(17, "S1_dispersion25_nb2_strong", "stress", "dmc_gaussian", "nb2",
         f"{DMC} d04 (high index dispersion, strong NB2)", index_sd=2.5, size=.2),
    _dgp(18, "S2_dispersion25_lognormal25", "stress", "dmc_gaussian", "lognormal_dmc",
         f"{DMC} d06 (residual lognormal SD 2.5; d06 also has index SD 2.5)",
         index_sd=2.5, residual_sd=2.5),
    _dgp(19, "S3_contamination_p001_x500", "stress", "dmc_gaussian", "contamination",
         f"{DMC} d07 (mean-preserving contamination; d11 variant p=.01,x10 not used)",
         index_sd=1.5, residual_sd=.5, contamination_probability=.001,
         contamination_multiplier=500.),
    _dgp(20, "S4_equicorrelated999_lognormal", "stress", "dmc_equicorrelated", "lognormal_dmc",
         f"{DMC} d08 (near-collinear Gaussian nuisances, correlation .999)",
         index_sd=1.5, correlation=.999, residual_sd=1.),
)
DGPS_BY_ID = {d["id"]: d for d in DGPS}
assert len(DGPS_BY_ID) == len(DGPS) == len({d["number"] for d in DGPS})


def cells(config):
    """Cell metadata in DGP-major, size-minor order (as dgp11.cells); draws nothing."""
    ids, sizes = config["dgp_ids"], config["sizes"]
    if len(set(ids)) != len(ids) or not set(ids) <= set(DGPS_BY_ID):
        raise ValueError("config dgp_ids must be unique known DGP ids")
    result = []
    for dgp_id in ids:
        dgp = DGPS_BY_ID[dgp_id]
        for n, k in sizes:
            if type(n) is not int or type(k) is not int or n < 1 or k < 4:
                raise ValueError("sizes must be integer pairs with n >= 1 and K >= 4")
            beta = coefficients(dgp_id, k)
            result.append(dict(
                id=f"{dgp_id}_n{n}_k{k}", dgp=dgp_id, dgp_number=dgp["number"],
                group=dgp["group"], n=n, k=k, covariates=dgp["covariates"],
                law=dgp["law"], parameters=dict(dgp["parameters"]), source=dgp["source"],
                draws=config["draws_per_cell"], treatment_column=TREATMENT_COLUMN,
                beta_d_true=TREATMENT_COEFFICIENT, theta_true=math.expm1(TREATMENT_COEFFICIENT),
                intercept=beta["beta"][0], nuisance_log_mgf=beta["nuisance_log_mgf"],
                population_untreated_mean=math.exp(beta["beta"][0]+beta["nuisance_log_mgf"])))
    return result


def seed_key(cell, draw, split, master_seed, stream):
    """SeedSequence entropy for one stream; covariate/outcome/order never share a key."""
    if isinstance(draw, bool) or not isinstance(draw, int) or not 0 <= draw < cell["draws"]:
        raise ValueError("draw must be an integer in [0, draws)")
    if int(master_seed) in HISTORICAL_MASTER_SEEDS:
        raise ValueError("master seed belongs to an earlier campaign")
    return [int(master_seed), SPLITS[split], cell["dgp_number"], cell["n"], cell["k"],
            draw, STREAMS[stream]]


# ---------------------------------------------------------------- coefficients

def _logaddexp(a, b):
    # Same branch formula as numpy's npy_logaddexp; pure math, no numpy import.
    if a == b:
        return a+math.log(2.)
    high, low = (a, b) if a > b else (b, a)
    return high+math.log1p(math.exp(low-high))


def _cp8_features(menu, k):
    """Port of CP8 feature_spec: (name, kind, probability) in prespecified order."""
    features = [("R", "rare_bernoulli", .02)] if menu == "C" else []
    remaining = k-2-len(features)
    binary = (remaining+1)//2
    for j in range(remaining):
        if menu == "B":
            features.append((f"W{j}", "std_gamma", None))
        elif j < binary:
            features.append((f"W{j}", "std_bernoulli", (.05, .2, .5)[j % 3]))
        else:
            features.append((f"W{j}", "normal", None))
    return tuple(features)


def _cp8_log_mgf(kind, p, t):
    """Port of CP8 log_mgf: exact population log E exp(tW)."""
    if kind == "normal":
        return t*t/2
    if kind == "std_gamma":
        if t >= math.sqrt(2.):
            raise ValueError("Standardized Gamma MGF requires t < sqrt(2)")
        return -math.sqrt(2.)*t-2*math.log1p(-t/math.sqrt(2.))
    scale = math.sqrt(p*(1-p)) if kind == "std_bernoulli" else 1.
    shift = -t*p/scale if kind == "std_bernoulli" else 0.
    return float(shift+_logaddexp(math.log1p(-p), math.log(p)+t/scale))


def coefficients(dgp_id, k):
    """Population beta and analytical log E exp(nuisance index); draws nothing."""
    dgp = DGPS_BY_ID[dgp_id]
    kind, p = dgp["covariates"], dgp["parameters"]
    if kind == "cp11_gaussian":
        # CP11 literal: alpha = log 2 - .18 with nuisance index variance .36.
        slopes = [.6*(-1.)**j/math.sqrt(k-2) for j in range(k-2)]
        alpha = math.log(2)-.18
        log_mgf = .5*sum(b*b for b in slopes)
    elif kind == "cp8_menu":
        features = _cp8_features(p["menu"], k)
        extra = 1 if p["menu"] == "C" else 0
        count = len(features)-extra
        slopes = [.6*(-1 if j % 2 else 1)/math.sqrt(count) for j in range(count)]
        if extra:
            slopes.insert(0, -math.log(20.))
        log_mgf = sum(_cp8_log_mgf(kind_, q, b) for (_, kind_, q), b in zip(features, slopes))
        alpha = math.log(2.)-log_mgf
    elif kind in ("dmc_gaussian", "dmc_equicorrelated"):
        # DMC _covariates: unit alternating slopes rescaled to index SD sigma.
        sigma, m = p["index_sd"], k-2
        slopes = [(-1.)**j for j in range(m)]
        if kind == "dmc_equicorrelated":
            rho = p["correlation"]
            variance = (1-rho)*sum(b*b for b in slopes)+rho*sum(slopes)**2
        else:
            variance = sum(b*b for b in slopes)
        slopes = [b*(sigma/math.sqrt(variance)) for b in slopes]
        log_mgf = .5*sigma**2
        alpha = math.log(2.)-log_mgf
    else:
        raise ValueError(f"Unknown covariate design {kind}")
    return dict(beta=[alpha, TREATMENT_COEFFICIENT, *slopes], nuisance_log_mgf=log_mgf)


# ---------------------------------------------------------------- drawing

def _rng(np, key):
    return np.random.default_rng(np.random.SeedSequence(key))


def _draw_covariates(np, cell, rng):
    """Returns x, H (heterogeneity indicator or None), rare indicator or None."""
    n, k, kind, p = cell["n"], cell["k"], cell["covariates"], cell["parameters"]
    if kind == "cp11_gaussian":
        x = np.column_stack((np.ones(n), rng.binomial(1, .5, n), rng.normal(size=(n, k-2))))
        return x, (x[:, 2] > 0).astype(int), None
    if kind == "cp8_menu":
        # Port of CP8 draw_covariates, same draw order; constant columns retained.
        x = np.empty((n, k), dtype=float)
        x[:, 0] = 1.
        x[:, 1] = rng.binomial(1, .5, size=n)
        h, rare = None, None
        for column, (_, feature, q) in enumerate(_cp8_features(p["menu"], k), start=2):
            if feature in ("std_bernoulli", "rare_bernoulli"):
                raw = rng.binomial(1, q, size=n)
                if feature == "rare_bernoulli":
                    rare = raw.copy()
                    x[:, column] = raw
                else:
                    x[:, column] = (raw-q)/math.sqrt(q*(1-q))
                    if h is None and q == .5:
                        h = raw.copy()
            elif feature == "std_gamma":
                raw = rng.gamma(2., 1., size=n)
                x[:, column] = (raw-2.)/math.sqrt(2.)
                if h is None:
                    h = (raw > 2.).astype(int)
            else:
                x[:, column] = rng.normal(size=n)
        return x, h, rare
    # DMC _covariates order: treatment, idiosyncratic z, then the common factor.
    treatment = rng.binomial(1, .5, n)
    z = rng.normal(size=(n, k-2))
    if kind == "dmc_equicorrelated":
        rho = p["correlation"]
        z = math.sqrt(rho)*rng.normal(size=(n, 1))+math.sqrt(1-rho)*z
    return np.column_stack((np.ones(n), treatment, z)), None, None


def _draw_outcome(np, cell, index, mu, h, rng):
    law, p, n = cell["law"], cell["parameters"], len(mu)
    if law == "poisson":
        return rng.poisson(mu).astype(float)
    if law == "gamma":
        return rng.gamma(1., mu)
    if law == "tweedie15":
        count = rng.poisson(np.sqrt(mu))
        y = np.zeros_like(mu)
        active = count > 0
        y[active] = rng.gamma(count[active], np.sqrt(mu[active]))
        return y
    if law == "ig3":
        return rng.wald(mu, 1.)
    if law in ("nb2", "heterogeneous_nb2"):
        size = (np.where(np.asarray(h) == 1, p["size_h1"], p["size_h0"])
                if law == "heterogeneous_nb2" else p["size"])
        return rng.negative_binomial(size, size/(size+mu)).astype(float)
    if law in ("lognormal_cp11", "zero_lognormal_cp11"):
        y = mu*np.exp(rng.normal(size=n)-.5)
        if law == "zero_lognormal_cp11":
            y *= rng.binomial(1, .5, n)/.5
        return y
    if law in ("lognormal_dmc", "contamination"):
        sd = p["residual_sd"]
        log_y = index+rng.normal(0, sd, n)-.5*sd*sd
        if law == "contamination":
            q, high = p["contamination_probability"], p["contamination_multiplier"]
            low = (1-q*high)/(1-q)
            if not low > 0:
                raise ValueError("contamination low component must have a positive mean")
            log_y = log_y+np.log(np.where(rng.binomial(1, q, n), high, low))
        return np.exp(log_y)
    raise ValueError(f"Unknown outcome law {law}")


def dataset_checksums(x, y):
    """SHA-256 of little-endian C-order float64 X and y; used for replay checks."""
    import numpy as np
    x = np.ascontiguousarray(x, dtype="<f8")
    y = np.ascontiguousarray(y, dtype="<f8")
    xs, ys = hashlib.sha256(x.tobytes()).hexdigest(), hashlib.sha256(y.tobytes()).hexdigest()
    header = json.dumps(dict(x_shape=list(x.shape), y_shape=list(y.shape), dtype="<f8"),
                        sort_keys=True)
    combined = hashlib.sha256(f"{header}|{xs}|{ys}".encode()).hexdigest()
    return dict(x_sha256=xs, y_sha256=ys, dataset_sha256=combined)


def generate(cell, draw, split, master_seed):
    """One replayable dataset: X, y, beta_true, scalar diagnostics, seeds, checksums.

    The covariate and outcome streams are independent SeedSequence keys. No
    draw is retried; exceptional support is reported, not fixed.
    """
    import numpy as np
    keys = {s: seed_key(cell, draw, split, master_seed, s) for s in ("covariates", "outcome")}
    x, h, rare = _draw_covariates(np, cell, _rng(np, keys["covariates"]))
    beta = np.asarray(coefficients(cell["dgp"], cell["k"])["beta"], dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        index = x@beta
        mu = np.exp(index)
    y = _draw_outcome(np, cell, index, mu, h, _rng(np, keys["outcome"]))
    treated = x[:, TREATMENT_COLUMN] == 1
    constant = np.flatnonzero(np.ptp(x[:, 1:], axis=0) == 0)+1
    diagnostics = dict(
        n=len(y), k=x.shape[1], draw_attempts=1, zero_count=int((y == 0).sum()),
        zero_fraction=float(np.mean(y == 0)), positive_support=bool(np.all(y > 0)),
        outcome_mean=float(y.mean()), outcome_max=float(y.max()), outcome_sum=float(y.sum()),
        numeric_finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
        index_min=float(index.min()), index_max=float(index.max()),
        treated_count=int(treated.sum()), treated_zero_count=int((treated & (y == 0)).sum()),
        untreated_zero_count=int((~treated & (y == 0)).sum()),
        constant_nonintercept_columns=len(constant),
        constant_columns_first10=",".join(str(int(c)) for c in constant[:10]))
    if rare is not None:
        mask = rare == 1
        count, positives = int(mask.sum()), int((y[mask] > 0).sum())
        diagnostics.update(rare_group_count=count, rare_positive_count=positives,
                           rare_status="absent_group" if count == 0 else
                           "no_positive_outcome" if positives == 0 else "has_positive_outcome")
    if cell["law"] == "heterogeneous_nb2":
        diagnostics["h_count"] = int(np.sum(h == 1))
    return dict(X=x, y=y, beta_true=beta, diagnostics=diagnostics, seed_keys=keys,
                checksums=dataset_checksums(x, y), cell_id=cell["id"], draw=draw, split=split)


def method_order(cell, draw, split, master_seed, methods):
    """Per-dataset method order from the separate order stream; never outcome-based."""
    import numpy as np
    permutation = _rng(np, seed_key(cell, draw, split, master_seed, "order")).permutation(len(methods))
    return [tuple(methods[int(i)]) for i in permutation]
