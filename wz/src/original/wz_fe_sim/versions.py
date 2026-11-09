"""Version labels recorded on every result row (standard library only)."""

PACKAGE_VERSION = "0.2.0"
# /2: root acceptance adds the fixed mean-scale metric gates (ACCEPTANCE_RULE_VERSION)
SOLVER_VERSION = "newton-lu-fixedW-armijo/2"
# Conjunction applied on every acceptance route (initial stop, final fresh
# certificate, certificate-after-failed-line-search): coordinate residual,
# coordinate Newton correction, strict rcond, mean-scale fixed-metric residual
# and mean-scale fixed-metric Newton correction.
ACCEPTANCE_RULE_VERSION = "coord-resid+coord-step+rcond+fixed-metric-mean/1"
# Start policy for joint roots: the prescribed direct start, then (only for a
# failed direct J target) the author-approved bounded tau continuation.
START_POLICY_DIRECT = "prescribed-direct/1"
START_POLICY_CONTINUATION = "prescribed-direct+joint-tau-continuation/1"


def start_policy_version(cfg):
    """Start-policy label implied by a configuration (continuation on/off)."""
    jc = cfg.get("joint_continuation") if hasattr(cfg, "get") else None
    return START_POLICY_CONTINUATION if (jc is not None and jc.get("enabled")) else START_POLICY_DIRECT
