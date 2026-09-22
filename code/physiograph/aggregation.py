"""Deterministic probability updates and patient-level phase aggregation."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


DIRECTIONS = ("improve", "stable", "worsen")
PHASES = (
    "baseline",
    "load_accumulation",
    "compensation",
    "conflict_emerges",
    "transition",
    "decompensation",
    "recovery",
    "residual_state",
)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def normalize_probabilities(values: Mapping[str, Any] | Sequence[Any]) -> dict[str, float]:
    """Return a finite, non-negative three-class distribution."""

    if isinstance(values, Mapping):
        raw = [values.get(direction, 0.0) for direction in DIRECTIONS]
    else:
        raw = list(values)
        if len(raw) != 3:
            raise ValueError("direction probabilities must have exactly three entries")
    clean: list[float] = []
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        clean.append(number if math.isfinite(number) and number > 0.0 else 0.0)
    total = sum(clean)
    if total <= 0.0:
        clean = [0.0, 1.0, 0.0]
        total = 1.0
    return {direction: clean[index] / total for index, direction in enumerate(DIRECTIONS)}


def estimate_probabilities(estimate: Any) -> dict[str, float]:
    """Read probabilities from an ``OrganEstimate`` or compatible mapping."""

    if isinstance(estimate, Mapping):
        if "probabilities" in estimate:
            return normalize_probabilities(estimate["probabilities"])
        return normalize_probabilities(
            {
                "improve": estimate.get("p_improve", estimate.get("improve", 0.0)),
                "stable": estimate.get("p_stable", estimate.get("stable", 0.0)),
                "worsen": estimate.get("p_worsen", estimate.get("worsen", 0.0)),
            }
        )
    return normalize_probabilities(
        {
            "improve": getattr(estimate, "p_improve", 0.0),
            "stable": getattr(estimate, "p_stable", 0.0),
            "worsen": getattr(estimate, "p_worsen", 0.0),
        }
    )


def bounded_residual_update(
    prior: Mapping[str, Any] | Sequence[Any],
    conditioned: Mapping[str, Any] | Sequence[Any],
    alpha: float,
    *,
    alpha_max: float = 0.6,
) -> dict[str, float]:
    """Apply ``P'=(1-alpha)P+alpha Q`` with a hard public upper bound."""

    if not 0.0 <= float(alpha_max) <= 1.0:
        raise ValueError("alpha_max must lie in [0, 1]")
    bounded = min(max(float(alpha), 0.0), float(alpha_max))
    p = normalize_probabilities(prior)
    q = normalize_probabilities(conditioned)
    return normalize_probabilities(
        {direction: (1.0 - bounded) * p[direction] + bounded * q[direction] for direction in DIRECTIONS}
    )


def normalized_entropy(probabilities: Mapping[str, Any]) -> float:
    p = normalize_probabilities(probabilities)
    entropy = -sum(value * math.log(value) for value in p.values() if value > 0.0)
    return entropy / math.log(len(DIRECTIONS))


def _organ_name(estimate: Any, fallback: str) -> str:
    if isinstance(estimate, Mapping):
        return str(estimate.get("organ", fallback))
    return str(getattr(estimate, "organ", fallback))


def _weights_from_graph(graph: Any) -> dict[str, float]:
    if graph is None:
        return {}
    values = getattr(graph, "aggregation_weights", None)
    if values is None and isinstance(graph, Mapping):
        values = graph.get("aggregation_weights", graph.get("organ_weights", {}))
    if isinstance(values, Mapping):
        items = values.items()
    else:
        items = values or ()
    result: dict[str, float] = {}
    for organ, weight in items:
        try:
            number = float(weight)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            result[str(organ)] = number
    return result


def _phase_thresholds(graph: Any) -> dict[str, float]:
    defaults = {
        "decompensation": 0.58,
        "recovery": 0.52,
        "conflict_emerges": 0.08,
        "load_accumulation": 0.43,
        "compensation": 0.36,
        "residual_state": 0.48,
        "baseline": 0.58,
    }
    if graph is None:
        return defaults
    values = getattr(graph, "phase_thresholds", None)
    if values is None and isinstance(graph, Mapping):
        values = graph.get("phase_thresholds", {})
    items = values.items() if isinstance(values, Mapping) else (values or ())
    for name, value in items:
        key = str(_enum_value(name))
        if key in defaults:
            try:
                defaults[key] = float(value)
            except (TypeError, ValueError):
                continue
    return defaults


def _js_disagreement(distributions: Sequence[Mapping[str, float]], mean: Mapping[str, float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0 or len(distributions) <= 1:
        return 0.0
    divergence = 0.0
    for distribution, weight in zip(distributions, weights):
        kl = 0.0
        for direction in DIRECTIONS:
            p, q = distribution[direction], mean[direction]
            if p > 0.0 and q > 0.0:
                kl += p * math.log(p / q)
        divergence += (weight / total_weight) * kl
    return min(max(divergence / math.log(len(DIRECTIONS)), 0.0), 1.0)


def phase_from_state(
    probabilities: Mapping[str, float],
    *,
    disagreement: float,
    dominant_directions: Sequence[str],
    previous_phase: Optional[str] = None,
    graph: Any = None,
) -> tuple[str, str]:
    """Public deterministic mapping from aggregate state to eight phases."""

    p = normalize_probabilities(probabilities)
    previous = str(_enum_value(previous_phase)) if previous_phase is not None else None
    thresholds = _phase_thresholds(graph)
    dominant = set(dominant_directions)

    if p["worsen"] >= thresholds["decompensation"]:
        return "decompensation", "aggregate worsen probability crossed the decompensation threshold"
    if p["improve"] >= thresholds["recovery"]:
        return "recovery", "aggregate improve probability crossed the recovery threshold"
    if {"improve", "worsen"}.issubset(dominant) and disagreement >= thresholds["conflict_emerges"]:
        return "conflict_emerges", "organ directions oppose one another above the disagreement threshold"
    if previous in {"recovery", "residual_state"} and p["stable"] >= thresholds["residual_state"]:
        return "residual_state", "state stabilized after a recovery phase"
    if p["worsen"] >= thresholds["load_accumulation"] and previous in {None, "baseline", "load_accumulation"}:
        return "load_accumulation", "worsening load rose above the accumulation threshold"
    if p["worsen"] >= thresholds["compensation"] and p["stable"] >= 0.30:
        return "compensation", "worsening signal is present while stable reserve remains"
    if previous in {"load_accumulation", "compensation", "conflict_emerges"} or max(p["improve"], p["worsen"]) >= 0.34:
        return "transition", "directional evidence indicates a phase transition"
    if p["stable"] >= thresholds["baseline"]:
        return "baseline", "stable probability remains above the baseline threshold"
    return "transition", "state is between deterministic phase thresholds"


@dataclass(frozen=True)
class AggregatedState:
    step: int
    probabilities: Mapping[str, float]
    phase: str
    uncertainty: float
    disagreement: float
    dominant_direction: str
    organ_order: tuple[str, ...]
    weights: Mapping[str, float]
    phase_reason: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def aggregate_estimates(
    estimates: Mapping[str, Any] | Sequence[Any],
    *,
    graph: Any = None,
    previous_phase: Optional[str] = None,
    step: int = 0,
) -> AggregatedState:
    """Aggregate organ probabilities in stable organ-name order."""

    if isinstance(estimates, Mapping):
        pairs = [(str(organ), estimate) for organ, estimate in estimates.items()]
    else:
        pairs = [(_organ_name(estimate, str(index)), estimate) for index, estimate in enumerate(estimates)]
    pairs.sort(key=lambda pair: pair[0])
    if not pairs:
        raise ValueError("at least one organ estimate is required")

    configured = _weights_from_graph(graph)
    organ_order = tuple(organ for organ, _ in pairs)
    weights = [configured.get(organ, 1.0) for organ in organ_order]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        weights = [1.0] * len(pairs)
        total_weight = float(len(pairs))
    distributions = [estimate_probabilities(estimate) for _, estimate in pairs]
    mean = {
        direction: sum(weight * distribution[direction] for weight, distribution in zip(weights, distributions)) / total_weight
        for direction in DIRECTIONS
    }
    mean = normalize_probabilities(mean)
    disagreement = _js_disagreement(distributions, mean, weights)
    dominant_per_organ = [max(DIRECTIONS, key=lambda key: (distribution[key], -DIRECTIONS.index(key))) for distribution in distributions]
    phase, reason = phase_from_state(
        mean,
        disagreement=disagreement,
        dominant_directions=dominant_per_organ,
        previous_phase=previous_phase,
        graph=graph,
    )
    dominant = max(DIRECTIONS, key=lambda key: (mean[key], -DIRECTIONS.index(key)))
    return AggregatedState(
        step=int(step),
        probabilities=mean,
        phase=phase,
        uncertainty=normalized_entropy(mean),
        disagreement=disagreement,
        dominant_direction=dominant,
        organ_order=organ_order,
        weights={organ: weights[index] / total_weight for index, organ in enumerate(organ_order)},
        phase_reason=reason,
    )


__all__ = [
    "AggregatedState",
    "DIRECTIONS",
    "PHASES",
    "aggregate_estimates",
    "bounded_residual_update",
    "estimate_probabilities",
    "normalize_probabilities",
    "normalized_entropy",
    "phase_from_state",
]
