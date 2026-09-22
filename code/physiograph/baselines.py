"""Executable registry for PhysioCircuit and baselines A--L.

Baseline orchestration is intentionally adapter based.  The core engine can expose
hooks through a mapping without this module importing or duplicating its schemas.
When a baseline needs an unavailable external artifact (for example a Vital
Trace-style protocol or trained implicit graph), the result is explicitly marked
``unavailable``; no placeholder prediction or score is fabricated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
from typing import Any, Callable, Mapping


RunHook = Callable[..., Any]


@dataclass(frozen=True)
class BaselineSpec:
    method_id: str
    name: str
    description: str
    variants: tuple[str, ...] = ()
    required_hooks: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()
    semantic_interventions: bool = False


@dataclass
class BaselineOutcome:
    method_id: str
    variant: str | None
    status: str
    predictions: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINES: dict[str, BaselineSpec] = {
    "PG": BaselineSpec(
        "PG",
        "PhysioCircuit",
        "Full typed, gated mechanism graph with bounded residual updates.",
        required_hooks=("run_pg",),
        required_resources=("graph", "cards"),
        semantic_interventions=True,
    ),
    "A": BaselineSpec(
        "A",
        "Global single agent (no RAG)",
        "One model reads the complete public prefix and predicts all organs.",
        required_hooks=("run_global",),
    ),
    "B": BaselineSpec(
        "B",
        "Global single agent + mechanism RAG",
        "Global predictor with deterministic frozen-card retrieval but no graph.",
        required_hooks=("run_global",),
        required_resources=("cards",),
    ),
    "C": BaselineSpec(
        "C",
        "Independent organs",
        "Organ-local predictions with deterministic public aggregation.",
        required_hooks=("run_organs", "aggregate"),
    ),
    "D": BaselineSpec(
        "D",
        "Typed communication without gates",
        "Correct graph and typed messages, with functional gates in fail-open mode.",
        required_hooks=("run_graph",),
        required_resources=("graph", "cards"),
        semantic_interventions=True,
    ),
    "E": BaselineSpec(
        "E",
        "Organ prediction + dev-optimized voting (H0)",
        "Independent organ probabilities combined by weights fitted on dev only.",
        required_hooks=("run_organs", "aggregate"),
        required_resources=("voting_weights",),
    ),
    "F": BaselineSpec(
        "F",
        "Compute-matched multi-agent without graph",
        "Same call/round budget as PG, replacing cross-organ exchange with self-review.",
        required_hooks=("run_compute_matched",),
    ),
    "G": BaselineSpec(
        "G",
        "Free-text multi-agent",
        "Correct topology with untyped free-text messages and no semantic gates.",
        required_hooks=("run_graph",),
        required_resources=("graph",),
    ),
    "H": BaselineSpec(
        "H",
        "Random graph",
        "Seeded degree/edge-count matched graph rewiring.",
        required_hooks=("run_graph", "mutate_graph"),
        required_resources=("graph", "cards"),
        semantic_interventions=True,
    ),
    "I": BaselineSpec(
        "I",
        "Topology controls",
        "Fully connected, reversed, or key-edge-deleted graph.",
        variants=("fully_connected", "reversed", "edge_deleted"),
        required_hooks=("run_graph", "mutate_graph"),
        required_resources=("graph", "cards"),
        semantic_interventions=True,
    ),
    "J": BaselineSpec(
        "J",
        "Gate/update controls",
        "No-gates or unbounded residual-update control.",
        variants=("no_gates", "unbounded"),
        required_hooks=("run_graph",),
        required_resources=("graph", "cards"),
        semantic_interventions=True,
    ),
    "K": BaselineSpec(
        "K",
        "Vital Trace-style protocol system",
        "Persistent patient memory coordinated by frozen patient-level protocols.",
        required_hooks=("run_protocol",),
        required_resources=("protocol",),
    ),
    "L": BaselineSpec(
        "L",
        "Implicit graph learning",
        "Dev-trained sparse edge weights, frozen before blind evaluation.",
        required_hooks=("run_implicit_graph",),
        required_resources=("learned_graph",),
    ),
}


ALIASES: dict[str, str] = {
    "pg": "PG",
    "physiograph": "PG",
    "global": "A",
    "global_no_rag": "A",
    "global_rag": "B",
    "independent": "C",
    "independent_organs": "C",
    "typed_no_gates": "D",
    "voting": "E",
    "optimal_voting": "E",
    "compute_matched": "F",
    "free_text": "G",
    "random_graph": "H",
    "topology": "I",
    "controls": "J",
    "vitaltrace": "K",
    "implicit": "L",
}


MAIN_METHODS: tuple[str, ...] = ("PG", "A", "B", "E", "F", "K", "L")


def normalize_method_id(method_id: str) -> str:
    token = method_id.strip()
    normalized = ALIASES.get(token.lower(), token.upper())
    if normalized not in BASELINES:
        raise KeyError(f"unknown method: {method_id!r}")
    return normalized


def get_spec(method_id: str) -> BaselineSpec:
    return BASELINES[normalize_method_id(method_id)]


def list_methods() -> list[dict[str, Any]]:
    return [asdict(BASELINES[key]) for key in BASELINES]


def _lookup(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def _invoke(hook: RunHook, **kwargs: Any) -> Any:
    """Call a hook while tolerating adapters with a narrower named signature."""

    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return hook(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return hook(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return hook(**accepted)


def check_availability(
    method_id: str,
    adapter: Any,
    resources: Mapping[str, Any] | None = None,
    *,
    variant: str | None = None,
) -> BaselineOutcome:
    method = normalize_method_id(method_id)
    spec = BASELINES[method]
    resources = resources or {}
    if spec.variants and variant not in spec.variants:
        return BaselineOutcome(
            method,
            variant,
            "invalid",
            reason=f"variant must be one of: {', '.join(spec.variants)}",
        )
    if not spec.variants and variant is not None:
        return BaselineOutcome(method, variant, "invalid", reason="method has no variants")
    missing_hooks = [name for name in spec.required_hooks if not callable(_lookup(adapter, name))]
    missing_resources = [name for name in spec.required_resources if resources.get(name) is None]
    missing = [f"hook:{name}" for name in missing_hooks] + [
        f"resource:{name}" for name in missing_resources
    ]
    if missing:
        return BaselineOutcome(
            method,
            variant,
            "unavailable",
            reason="required implementation or artifact is unavailable",
            missing=missing,
        )
    return BaselineOutcome(method, variant, "ready")


def _as_outcome(method: str, variant: str | None, value: Any) -> BaselineOutcome:
    if isinstance(value, BaselineOutcome):
        return value
    if isinstance(value, Mapping) and value.get("status") in {
        "ok",
        "unavailable",
        "invalid",
        "error",
        "skipped",
    }:
        return BaselineOutcome(
            method_id=method,
            variant=variant,
            status=str(value["status"]),
            predictions=value.get("predictions"),
            artifacts=dict(value.get("artifacts", {})),
            reason=value.get("reason"),
            missing=list(value.get("missing", [])),
        )
    return BaselineOutcome(method, variant, "ok", predictions=value)


def run_method(
    method_id: str,
    case: Any,
    adapter: Any,
    *,
    resources: Mapping[str, Any] | None = None,
    variant: str | None = None,
    run_seed: int = 0,
    intervention: Any = None,
    options: Mapping[str, Any] | None = None,
) -> BaselineOutcome:
    """Execute PG or one A--L baseline through engine adapter hooks."""

    method = normalize_method_id(method_id)
    resources = resources or {}
    options_dict = dict(options or {})
    availability = check_availability(method, adapter, resources, variant=variant)
    if availability.status != "ready":
        return availability

    common = {
        "case": case,
        "resources": resources,
        "run_seed": run_seed,
        "intervention": intervention,
        "options": options_dict,
    }
    try:
        if method == "PG":
            value = _invoke(_lookup(adapter, "run_pg"), **common)
        elif method in {"A", "B"}:
            value = _invoke(
                _lookup(adapter, "run_global"),
                **common,
                use_rag=method == "B",
                intervention_mode="natural_language" if intervention is not None else None,
            )
        elif method in {"C", "E"}:
            local = _invoke(_lookup(adapter, "run_organs"), **common, local_only=True)
            value = _invoke(
                _lookup(adapter, "aggregate"),
                case=case,
                estimates=local,
                mode="dev_optimized_voting" if method == "E" else "public",
                weights=resources.get("voting_weights") if method == "E" else None,
                options=options_dict,
            )
        elif method == "D":
            value = _invoke(
                _lookup(adapter, "run_graph"),
                **common,
                graph=resources["graph"],
                cards=resources["cards"],
                message_mode="typed",
                gate_mode="fail_open",
            )
        elif method == "F":
            value = _invoke(_lookup(adapter, "run_compute_matched"), **common)
        elif method == "G":
            value = _invoke(
                _lookup(adapter, "run_graph"),
                **common,
                graph=resources["graph"],
                message_mode="free_text",
                gate_mode="none",
            )
        elif method == "H":
            randomized = _invoke(
                _lookup(adapter, "mutate_graph"),
                graph=resources["graph"],
                mutation="random_degree_matched",
                seed=run_seed,
                options=options_dict,
            )
            value = _invoke(
                _lookup(adapter, "run_graph"),
                **common,
                graph=randomized,
                cards=resources["cards"],
                message_mode="typed",
                gate_mode="full",
            )
        elif method == "I":
            mutation = {
                "fully_connected": "fully_connected",
                "reversed": "reversed",
                "edge_deleted": "edge_deleted",
            }[str(variant)]
            graph = _invoke(
                _lookup(adapter, "mutate_graph"),
                graph=resources["graph"],
                mutation=mutation,
                seed=run_seed,
                intervention=intervention,
                options=options_dict,
            )
            value = _invoke(
                _lookup(adapter, "run_graph"),
                **common,
                graph=graph,
                cards=resources["cards"],
                message_mode="typed",
                gate_mode="full",
            )
        elif method == "J":
            graph_options = dict(options_dict)
            if variant == "unbounded":
                graph_options["alpha_max"] = 1.0
            value = _invoke(
                _lookup(adapter, "run_graph"),
                **{**common, "options": graph_options},
                graph=resources["graph"],
                cards=resources["cards"],
                message_mode="typed",
                gate_mode="none" if variant == "no_gates" else "full",
            )
        elif method == "K":
            value = _invoke(
                _lookup(adapter, "run_protocol"),
                **common,
                protocol=resources["protocol"],
            )
        elif method == "L":
            value = _invoke(
                _lookup(adapter, "run_implicit_graph"),
                **common,
                learned_graph=resources["learned_graph"],
            )
        else:  # pragma: no cover - registry normalization makes this unreachable.
            raise AssertionError(method)
        return _as_outcome(method, variant, value)
    except Exception as exc:  # preserve batch execution and log the case-level error
        return BaselineOutcome(
            method,
            variant,
            "error",
            reason=f"{type(exc).__name__}: {exc}",
        )


def run_baseline(*args: Any, **kwargs: Any) -> BaselineOutcome:
    """Alias retained for explicit baseline callers."""

    return run_method(*args, **kwargs)


__all__ = [
    "ALIASES",
    "BASELINES",
    "MAIN_METHODS",
    "BaselineOutcome",
    "BaselineSpec",
    "check_availability",
    "get_spec",
    "list_methods",
    "normalize_method_id",
    "run_baseline",
    "run_method",
]
