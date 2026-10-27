"""PhysioCircuit: auditable organ-graph multi-agent trajectory forecasting.

The package deliberately separates four concerns:

* public patient inputs used by model prompts;
* private gold labels used only by the scorer;
* system-specific mechanism cards and graph configuration;
* model/API execution records.

No module reads ``dataset_pg/l1`` implicitly.  Precomputed L1 artifacts are
available only through :mod:`physiograph.legacy`, where their provenance is
recorded as ``precomputed`` rather than ``api_run``.
"""

from .schema import (
    ArbitrationVerdict,
    Direction,
    EvidenceRef,
    GraphConfig,
    MechanismCard,
    MechanismMessage,
    OrganEstimate,
    PatientCase,
    Verdict,
)

__all__ = [
    "ArbitrationVerdict",
    "Direction",
    "EvidenceRef",
    "GraphConfig",
    "MechanismCard",
    "MechanismMessage",
    "OrganEstimate",
    "PatientCase",
    "Verdict",
]
