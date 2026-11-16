"""Package-owned diagnostics helpers."""

from .fe_routing import (
    AutoFERoutingDecision,
    FESupportDiagnostics,
    diagnose_fe_support,
    resolve_auto_fe_routing,
)

__all__ = [
    "AutoFERoutingDecision",
    "FESupportDiagnostics",
    "diagnose_fe_support",
    "resolve_auto_fe_routing",
]
