"""Stable preprocessing contracts for the staged migration packets."""

from .contracts import (
    CollinearityDiagnosticsContract,
    FESupportSummary,
    PreprocessContract,
    PreprocessSampleAccounting,
    PreprocessStepRecord,
    SingletonDiagnosticsContract,
    dropped_observations_from_mask,
    fe_level_counts,
    fullrecover_research_contract,
    partitioned_research_contract,
    scope_only_contract,
)
from .partitioned import (
    PartitionedPreprocessMeta,
    RegressorRankScreen,
    preprocess_partitioned_inputs,
    rank_reveal_qr,
    screen_regressor_matrix,
    solve_ols_qr_dropcollinear,
)

__all__ = [
    "CollinearityDiagnosticsContract",
    "FESupportSummary",
    "PreprocessContract",
    "PreprocessSampleAccounting",
    "PreprocessStepRecord",
    "SingletonDiagnosticsContract",
    "PartitionedPreprocessMeta",
    "RegressorRankScreen",
    "dropped_observations_from_mask",
    "fe_level_counts",
    "fullrecover_research_contract",
    "partitioned_research_contract",
    "preprocess_partitioned_inputs",
    "rank_reveal_qr",
    "screen_regressor_matrix",
    "scope_only_contract",
    "solve_ols_qr_dropcollinear",
]
