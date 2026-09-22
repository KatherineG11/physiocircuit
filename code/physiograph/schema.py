"""Strict, immutable schemas used by the PhysioCircuit reference implementation.

The dataclasses in this module deliberately accept already-normalised Python
values.  JSON/YAML-facing code should use ``from_dict``; those constructors
perform the small, explicit coercions needed at the configuration boundary and
reject unknown fields.  Internal code therefore never has to guess whether a
direction, lag, or probability has already been validated.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string")
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters: {value!r}")
    return value


def _text(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{name} must be a trimmed string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _tuple(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple inside the validated model")
    return value


def _string_tuple(
    value: object, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    items = _tuple(value, name)
    if not allow_empty and not items:
        raise ValueError(f"{name} must not be empty")
    for index, item in enumerate(items):
        _identifier(item, f"{name}[{index}]")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicates")
    return items  # type: ignore[return-value]


def _text_tuple(
    value: object, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    items = _tuple(value, name)
    if not allow_empty and not items:
        raise ValueError(f"{name} must not be empty")
    for index, item in enumerate(items):
        _text(item, f"{name}[{index}]")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicates")
    return items  # type: ignore[return-value]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], kind: str) -> None:
    unknown = set(data).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {kind} field(s): {', '.join(sorted(unknown))}")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    return value


def _enum(enum_type: type[Enum], value: object, name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(f"{name} must be one of {choices}") from exc


def _pairs_from_mapping(value: object, name: str) -> tuple[tuple[str, float], ...]:
    mapping = _mapping(value, name)
    pairs: list[tuple[str, float]] = []
    for key, raw in mapping.items():
        _identifier(key, f"{name} key")
        pairs.append((key, _finite_number(raw, f"{name}[{key!r}]")))
    return tuple(sorted(pairs))


def _pairs_to_dict(value: tuple[tuple[str, float], ...]) -> dict[str, float]:
    return {key: number for key, number in value}


def _parse_lag_steps(value: object) -> tuple[int, ...]:
    """Parse an inclusive JSON lag window or an explicit list of steps.

    ``{"min": 0, "max": 2}`` and a two-element list ``[0, 2]`` denote an
    inclusive window.  Lists with any other length are interpreted as explicit
    allowed steps.  This matches the paper's ``[min,max]`` card representation
    while retaining support for cards such as ``[0, 1, 2]``.
    """

    if isinstance(value, Mapping):
        _reject_unknown(value, {"min", "max"}, "lag_steps")
        if "min" not in value or "max" not in value:
            raise ValueError("lag_steps range requires min and max")
        lower = _nonnegative_int(value["min"], "lag_steps.min")
        upper = _nonnegative_int(value["max"], "lag_steps.max")
        if lower > upper:
            raise ValueError("lag_steps.min must be <= lag_steps.max")
        return tuple(range(lower, upper + 1))
    items = _list(value, "lag_steps")
    if not items:
        raise ValueError("lag_steps must not be empty")
    parsed = tuple(_nonnegative_int(item, f"lag_steps[{i}]") for i, item in enumerate(items))
    if len(parsed) == 2:
        lower, upper = parsed
        if lower > upper:
            raise ValueError("lag_steps lower bound must be <= upper bound")
        return tuple(range(lower, upper + 1))
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("explicit lag_steps must be sorted and unique")
    return parsed


def _lag_steps_to_json(value: tuple[int, ...]) -> list[int]:
    if value == tuple(range(value[0], value[-1] + 1)):
        return [value[0], value[-1]]
    return list(value)


class Direction(str, Enum):
    IMPROVE = "improve"
    STABLE = "stable"
    WORSEN = "worsen"


class Verdict(str, Enum):
    ACCEPT = "accept"
    PARTIALLY_ACCEPT = "partially_accept"
    REJECT = "reject"


class TreatmentAction(str, Enum):
    ACTIVATE = "activate"
    SUPPRESS = "suppress"


class MacroPhase(str, Enum):
    BASELINE = "baseline"
    LOAD_ACCUMULATION = "load_accumulation"
    COMPENSATION = "compensation"
    CONFLICT_EMERGES = "conflict_emerges"
    TRANSITION = "transition"
    DECOMPENSATION = "decompensation"
    RECOVERY = "recovery"
    RESIDUAL_STATE = "residual_state"


class InterventionKind(str, Enum):
    DELETE = "delete"
    REPLACE = "replace"
    REVERSE = "reverse"
    EXTEND_LAG = "extend_lag"
    INJECT_WRONG_MESSAGE = "inject_wrong_message"
    MASK_LOCAL_EVIDENCE = "mask_local_evidence"


@dataclass(frozen=True)
class TreatmentBinding:
    treatment: str
    action: TreatmentAction

    def __post_init__(self) -> None:
        _text(self.treatment, "treatment")
        if not isinstance(self.action, TreatmentAction):
            raise TypeError("action must be TreatmentAction")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TreatmentBinding":
        data = _mapping(data, "treatment binding")
        _reject_unknown(data, {"treatment", "action"}, "treatment binding")
        return cls(
            treatment=data.get("treatment"),
            action=_enum(TreatmentAction, data.get("action"), "action"),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, str]:
        return {"treatment": self.treatment, "action": self.action.value}


@dataclass(frozen=True)
class MechanismCard:
    mechanism_id: str
    source: str
    target: str
    source_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    allowed_directions: tuple[Direction, ...]
    lag_steps: tuple[int, ...]
    activation_evidence: tuple[str, ...]
    counterevidence: tuple[str, ...] = ()
    treatment_bindings: tuple[TreatmentBinding, ...] = ()
    mechanism_description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.mechanism_id, "mechanism_id")
        _identifier(self.source, "source")
        _identifier(self.target, "target")
        _string_tuple(self.source_variables, "source_variables")
        _string_tuple(self.target_variables, "target_variables")
        _tuple(self.allowed_directions, "allowed_directions")
        if not self.allowed_directions:
            raise ValueError("allowed_directions must not be empty")
        if any(not isinstance(item, Direction) for item in self.allowed_directions):
            raise TypeError("allowed_directions must contain Direction values")
        if len(set(self.allowed_directions)) != len(self.allowed_directions):
            raise ValueError("allowed_directions must be unique")
        _tuple(self.lag_steps, "lag_steps")
        if not self.lag_steps:
            raise ValueError("lag_steps must not be empty")
        for index, lag in enumerate(self.lag_steps):
            _nonnegative_int(lag, f"lag_steps[{index}]")
        if tuple(sorted(set(self.lag_steps))) != self.lag_steps:
            raise ValueError("lag_steps must be sorted and unique")
        _text_tuple(self.activation_evidence, "activation_evidence")
        _text_tuple(self.counterevidence, "counterevidence", allow_empty=True)
        _tuple(self.treatment_bindings, "treatment_bindings")
        if any(not isinstance(item, TreatmentBinding) for item in self.treatment_bindings):
            raise TypeError("treatment_bindings must contain TreatmentBinding values")
        _text(self.mechanism_description, "mechanism_description", allow_empty=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MechanismCard":
        data = _mapping(data, "mechanism card")
        allowed = {
            "mechanism_id", "source", "target", "source_variables",
            "target_variables", "allowed_directions", "lag_steps",
            "activation_evidence", "counterevidence", "treatment_bindings",
            "mechanism_description", "description",
        }
        _reject_unknown(data, allowed, "mechanism card")
        description = data.get("mechanism_description", data.get("description", ""))
        return cls(
            mechanism_id=data.get("mechanism_id"),
            source=data.get("source"),
            target=data.get("target"),
            source_variables=tuple(_list(data.get("source_variables"), "source_variables")),
            target_variables=tuple(_list(data.get("target_variables"), "target_variables")),
            allowed_directions=tuple(
                _enum(Direction, value, "allowed_directions item")
                for value in _list(data.get("allowed_directions"), "allowed_directions")
            ),  # type: ignore[arg-type]
            lag_steps=_parse_lag_steps(data.get("lag_steps")),
            activation_evidence=tuple(
                _list(data.get("activation_evidence"), "activation_evidence")
            ),
            counterevidence=tuple(
                _list(data.get("counterevidence", []), "counterevidence")
            ),
            treatment_bindings=tuple(
                TreatmentBinding.from_dict(item)
                for item in _list(data.get("treatment_bindings", []), "treatment_bindings")
            ),
            mechanism_description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "source": self.source,
            "target": self.target,
            "source_variables": list(self.source_variables),
            "target_variables": list(self.target_variables),
            "allowed_directions": [item.value for item in self.allowed_directions],
            "lag_steps": _lag_steps_to_json(self.lag_steps),
            "activation_evidence": list(self.activation_evidence),
            "counterevidence": list(self.counterevidence),
            "treatment_bindings": [item.to_dict() for item in self.treatment_bindings],
            "mechanism_description": self.mechanism_description,
        }


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    mechanism_ids: tuple[str, ...]
    active: bool = True

    def __post_init__(self) -> None:
        _identifier(self.source, "edge.source")
        _identifier(self.target, "edge.target")
        _string_tuple(self.mechanism_ids, "edge.mechanism_ids")
        _bool(self.active, "edge.active")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edge":
        data = _mapping(data, "edge")
        _reject_unknown(data, {"source", "target", "mechanism_ids", "active"}, "edge")
        return cls(
            source=data.get("source"),
            target=data.get("target"),
            mechanism_ids=tuple(_list(data.get("mechanism_ids"), "mechanism_ids")),
            active=data.get("active", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "mechanism_ids": list(self.mechanism_ids),
            "active": self.active,
        }


EdgeSpec = Edge


@dataclass(frozen=True)
class GraphConfig:
    domain: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    aggregation_weights: tuple[tuple[str, float], ...] = ()
    core_organs: tuple[str, ...] = ()
    phase_thresholds: tuple[tuple[str, float], ...] = ()
    schema_version: str = "physiograph.graph.v1"

    def __post_init__(self) -> None:
        _identifier(self.domain, "domain")
        _identifier(self.schema_version, "schema_version")
        _string_tuple(self.nodes, "nodes")
        if len(self.nodes) < 2:
            raise ValueError("a graph requires at least two nodes")
        _tuple(self.edges, "edges")
        if any(not isinstance(edge, Edge) for edge in self.edges):
            raise TypeError("edges must contain Edge values")
        edge_keys = [(edge.source, edge.target) for edge in self.edges]
        if len(set(edge_keys)) != len(edge_keys):
            raise ValueError("only one Edge object is allowed per directed endpoint pair")
        node_set = set(self.nodes)
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError(
                    f"edge {edge.source}->{edge.target} refers to an unknown node"
                )
        self._validate_pairs(self.aggregation_weights, "aggregation_weights")
        weights = dict(self.aggregation_weights)
        if set(weights).difference(node_set):
            raise ValueError("aggregation_weights contains an unknown node")
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("aggregation weights must be non-negative")
        _string_tuple(self.core_organs, "core_organs", allow_empty=True)
        if set(self.core_organs).difference(node_set):
            raise ValueError("core_organs contains an unknown node")
        self._validate_pairs(self.phase_thresholds, "phase_thresholds")

    @staticmethod
    def _validate_pairs(value: object, name: str) -> None:
        items = _tuple(value, name)
        keys: list[str] = []
        for index, item in enumerate(items):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(f"{name}[{index}] must be a (key, value) tuple")
            key, number = item
            _identifier(key, f"{name}[{index}].key")
            _finite_number(number, f"{name}[{index}].value")
            keys.append(key)
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError(f"{name} must be sorted by unique keys")

    @property
    def organs(self) -> tuple[str, ...]:
        return self.nodes

    @property
    def system_id(self) -> str:
        return self.domain

    def edge(self, source: str, target: str) -> Optional[Edge]:
        return next(
            (item for item in self.edges if item.source == source and item.target == target),
            None,
        )

    def allows(self, source: str, target: str, mechanism_id: str) -> bool:
        edge = self.edge(source, target)
        return bool(edge and edge.active and mechanism_id in edge.mechanism_ids)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphConfig":
        data = _mapping(data, "graph config")
        allowed = {
            "domain", "nodes", "edges", "aggregation_weights",
            "core_organs", "phase_thresholds", "schema_version",
        }
        _reject_unknown(data, allowed, "graph config")
        return cls(
            domain=data.get("domain"),
            nodes=tuple(_list(data.get("nodes"), "nodes")),
            edges=tuple(Edge.from_dict(item) for item in _list(data.get("edges"), "edges")),
            aggregation_weights=_pairs_from_mapping(
                data.get("aggregation_weights", {}), "aggregation_weights"
            ),
            core_organs=tuple(_list(data.get("core_organs", []), "core_organs")),
            phase_thresholds=_pairs_from_mapping(
                data.get("phase_thresholds", {}), "phase_thresholds"
            ),
            schema_version=data.get("schema_version", "physiograph.graph.v1"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": self.domain,
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges],
            "aggregation_weights": _pairs_to_dict(self.aggregation_weights),
            "core_organs": list(self.core_organs),
            "phase_thresholds": _pairs_to_dict(self.phase_thresholds),
        }


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    organ: str
    variable: str
    step: int
    text: str
    concept_code: str = ""
    predicted: bool = False

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        _identifier(self.organ, "evidence.organ")
        _identifier(self.variable, "evidence.variable")
        _nonnegative_int(self.step, "evidence.step")
        _text(self.text, "evidence.text")
        if self.concept_code:
            _identifier(self.concept_code, "evidence.concept_code")
        _bool(self.predicted, "evidence.predicted")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRef":
        data = _mapping(data, "evidence")
        allowed = {
            "evidence_id", "id", "organ", "variable", "step", "text",
            "concept_code", "predicted",
        }
        _reject_unknown(data, allowed, "evidence")
        return cls(
            evidence_id=data.get("evidence_id", data.get("id")),
            organ=data.get("organ"),
            variable=data.get("variable"),
            step=data.get("step"),
            text=data.get("text"),
            concept_code=data.get("concept_code", ""),
            predicted=data.get("predicted", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "organ": self.organ,
            "variable": self.variable,
            "step": self.step,
            "text": self.text,
            "concept_code": self.concept_code,
            "predicted": self.predicted,
        }


@dataclass(frozen=True)
class MechanismMessage:
    message_id: str
    sender: str
    receiver: str
    mechanism_id: str
    source_variable: str
    target_variable: str
    direction: Direction
    strength: float
    lag: int
    evidence_ids: tuple[str, ...]
    counterevidence_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    created_at_step: int = 0

    def __post_init__(self) -> None:
        _identifier(self.message_id, "message_id")
        _identifier(self.sender, "sender")
        _identifier(self.receiver, "receiver")
        _identifier(self.mechanism_id, "mechanism_id")
        _identifier(self.source_variable, "source_variable")
        _identifier(self.target_variable, "target_variable")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be Direction")
        object.__setattr__(self, "strength", _unit_interval(self.strength, "strength"))
        _nonnegative_int(self.lag, "lag")
        _string_tuple(self.evidence_ids, "evidence_ids")
        _string_tuple(self.counterevidence_ids, "counterevidence_ids", allow_empty=True)
        if set(self.evidence_ids).intersection(self.counterevidence_ids):
            raise ValueError("the same ID cannot be evidence and counterevidence")
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        _nonnegative_int(self.created_at_step, "created_at_step")

    @property
    def fingerprint(self) -> str:
        payload = {
            "sender": self.sender,
            "receiver": self.receiver,
            "mechanism_id": self.mechanism_id,
            "source_variable": self.source_variable,
            "target_variable": self.target_variable,
            "direction": self.direction.value,
            "lag": self.lag,
            "evidence_ids": self.evidence_ids,
            "counterevidence_ids": self.counterevidence_ids,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MechanismMessage":
        data = _mapping(data, "mechanism message")
        allowed = {field.name for field in fields(cls)}
        _reject_unknown(data, allowed, "mechanism message")
        return cls(
            message_id=data.get("message_id"),
            sender=data.get("sender"),
            receiver=data.get("receiver"),
            mechanism_id=data.get("mechanism_id"),
            source_variable=data.get("source_variable"),
            target_variable=data.get("target_variable"),
            direction=_enum(Direction, data.get("direction"), "direction"),  # type: ignore[arg-type]
            strength=data.get("strength"),
            lag=data.get("lag"),
            evidence_ids=tuple(_list(data.get("evidence_ids"), "evidence_ids")),
            counterevidence_ids=tuple(
                _list(data.get("counterevidence_ids", []), "counterevidence_ids")
            ),
            confidence=data.get("confidence", 1.0),
            created_at_step=data.get("created_at_step", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "mechanism_id": self.mechanism_id,
            "source_variable": self.source_variable,
            "target_variable": self.target_variable,
            "direction": self.direction.value,
            "strength": self.strength,
            "lag": self.lag,
            "evidence_ids": list(self.evidence_ids),
            "counterevidence_ids": list(self.counterevidence_ids),
            "confidence": self.confidence,
            "created_at_step": self.created_at_step,
        }


@dataclass(frozen=True)
class OrganEstimate:
    organ: str
    step: int
    p_improve: float
    p_stable: float
    p_worsen: float
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.organ, "organ")
        _nonnegative_int(self.step, "step")
        values = (
            _unit_interval(self.p_improve, "p_improve"),
            _unit_interval(self.p_stable, "p_stable"),
            _unit_interval(self.p_worsen, "p_worsen"),
        )
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("organ probabilities must sum to 1")
        object.__setattr__(self, "p_improve", values[0])
        object.__setattr__(self, "p_stable", values[1])
        object.__setattr__(self, "p_worsen", values[2])
        _string_tuple(self.evidence_ids, "evidence_ids", allow_empty=True)

    @property
    def probabilities(self) -> tuple[tuple[Direction, float], ...]:
        return (
            (Direction.IMPROVE, self.p_improve),
            (Direction.STABLE, self.p_stable),
            (Direction.WORSEN, self.p_worsen),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrganEstimate":
        data = _mapping(data, "organ estimate")
        allowed = {field.name for field in fields(cls)} | {"probabilities"}
        _reject_unknown(data, allowed, "organ estimate")
        probabilities = data.get("probabilities")
        if probabilities is not None:
            probabilities = _mapping(probabilities, "probabilities")
        else:
            probabilities = {}
        return cls(
            organ=data.get("organ"),
            step=data.get("step"),
            p_improve=data.get("p_improve", probabilities.get("improve")),
            p_stable=data.get("p_stable", probabilities.get("stable")),
            p_worsen=data.get("p_worsen", probabilities.get("worsen")),
            evidence_ids=tuple(_list(data.get("evidence_ids", []), "evidence_ids")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "step": self.step,
            "probabilities": {
                "improve": self.p_improve,
                "stable": self.p_stable,
                "worsen": self.p_worsen,
            },
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ArbitrationVerdict:
    message_id: str
    receiver: str
    verdict: Verdict
    reason: str
    weight: float
    conflict_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.message_id, "message_id")
        _identifier(self.receiver, "receiver")
        if not isinstance(self.verdict, Verdict):
            raise TypeError("verdict must be Verdict")
        _text(self.reason, "reason")
        object.__setattr__(self, "weight", _unit_interval(self.weight, "weight"))
        _string_tuple(
            self.conflict_evidence_ids, "conflict_evidence_ids", allow_empty=True
        )
        if self.verdict is Verdict.REJECT and self.weight != 0.0:
            raise ValueError("a rejected message must have zero weight")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArbitrationVerdict":
        data = _mapping(data, "arbitration verdict")
        allowed = {field.name for field in fields(cls)}
        _reject_unknown(data, allowed, "arbitration verdict")
        return cls(
            message_id=data.get("message_id"),
            receiver=data.get("receiver"),
            verdict=_enum(Verdict, data.get("verdict"), "verdict"),  # type: ignore[arg-type]
            reason=data.get("reason"),
            weight=data.get("weight"),
            conflict_evidence_ids=tuple(
                _list(data.get("conflict_evidence_ids", []), "conflict_evidence_ids")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "receiver": self.receiver,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "weight": self.weight,
            "conflict_evidence_ids": list(self.conflict_evidence_ids),
        }


@dataclass(frozen=True)
class ObservationNode:
    step: int
    time_label: str
    evidence: tuple[EvidenceRef, ...]
    elapsed_steps: int = 1
    predicted: bool = False

    def __post_init__(self) -> None:
        _nonnegative_int(self.step, "node.step")
        _text(self.time_label, "node.time_label")
        _tuple(self.evidence, "node.evidence")
        if any(not isinstance(item, EvidenceRef) for item in self.evidence):
            raise TypeError("node.evidence must contain EvidenceRef values")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique within a node")
        _nonnegative_int(self.elapsed_steps, "node.elapsed_steps")
        _bool(self.predicted, "node.predicted")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationNode":
        data = _mapping(data, "observation node")
        allowed = {field.name for field in fields(cls)} | {"time"}
        _reject_unknown(data, allowed, "observation node")
        return cls(
            step=data.get("step"),
            time_label=data.get("time_label", data.get("time")),
            evidence=tuple(
                EvidenceRef.from_dict(item)
                for item in _list(data.get("evidence", []), "evidence")
            ),
            elapsed_steps=data.get("elapsed_steps", 1),
            predicted=data.get("predicted", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "time_label": self.time_label,
            "evidence": [item.to_dict() for item in self.evidence],
            "elapsed_steps": self.elapsed_steps,
            "predicted": self.predicted,
        }


@dataclass(frozen=True)
class PatientCase:
    case_id: str
    domain: str
    nodes: tuple[ObservationNode, ...]
    forecast_horizon: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        _identifier(self.domain, "domain")
        _tuple(self.nodes, "case.nodes")
        if not self.nodes or any(not isinstance(item, ObservationNode) for item in self.nodes):
            raise ValueError("case.nodes must contain at least one ObservationNode")
        steps = [item.step for item in self.nodes]
        if steps != sorted(set(steps)):
            raise ValueError("case node steps must be strictly increasing")
        _nonnegative_int(self.forecast_horizon, "forecast_horizon")
        if self.forecast_horizon == 0:
            raise ValueError("forecast_horizon must be positive")
        _tuple(self.metadata, "metadata")
        keys: list[str] = []
        for index, pair in enumerate(self.metadata):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"metadata[{index}] must be a (key, value) tuple")
            _identifier(pair[0], f"metadata[{index}].key")
            _text(pair[1], f"metadata[{index}].value", allow_empty=True)
            keys.append(pair[0])
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("metadata keys must be sorted and unique")

    @property
    def observed_nodes(self) -> tuple[ObservationNode, ...]:
        return tuple(item for item in self.nodes if not item.predicted)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatientCase":
        data = _mapping(data, "patient case")
        allowed = {
            "case_id", "domain", "nodes", "observation_nodes", "forecast_horizon",
            "metadata",
        }
        _reject_unknown(data, allowed, "patient case")
        raw_nodes = data.get("nodes", data.get("observation_nodes"))
        metadata = _mapping(data.get("metadata", {}), "metadata")
        metadata_pairs = tuple(sorted((str(key), str(value)) for key, value in metadata.items()))
        return cls(
            case_id=data.get("case_id"),
            domain=data.get("domain"),
            nodes=tuple(
                ObservationNode.from_dict(item) for item in _list(raw_nodes, "nodes")
            ),
            forecast_horizon=data.get("forecast_horizon"),
            metadata=metadata_pairs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "nodes": [item.to_dict() for item in self.nodes],
            "forecast_horizon": self.forecast_horizon,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GateDecision:
    gate_number: int
    name: str
    passed: bool
    reason: str
    layer: str

    def __post_init__(self) -> None:
        if isinstance(self.gate_number, bool) or not isinstance(self.gate_number, int):
            raise TypeError("gate_number must be int")
        if not 1 <= self.gate_number <= 14:
            raise ValueError("gate_number must be in [1, 14]")
        _identifier(self.name, "gate name")
        _bool(self.passed, "passed")
        _text(self.reason, "gate reason")
        if self.layer not in {"structural", "semantic"}:
            raise ValueError("gate layer must be 'structural' or 'semantic'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_number": self.gate_number,
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "layer": self.layer,
        }


@dataclass(frozen=True)
class GateReport:
    message_id: str
    decisions: tuple[GateDecision, ...]

    def __post_init__(self) -> None:
        _identifier(self.message_id, "message_id")
        _tuple(self.decisions, "gate decisions")
        if len(self.decisions) != 14:
            raise ValueError("a GateReport must contain exactly fourteen decisions")
        if any(not isinstance(item, GateDecision) for item in self.decisions):
            raise TypeError("decisions must contain GateDecision values")
        if tuple(item.gate_number for item in self.decisions) != tuple(range(1, 15)):
            raise ValueError("gate decisions must be ordered from 1 through 14")
        names = [item.name for item in self.decisions]
        if len(set(names)) != 14:
            raise ValueError("gate decision names must be unique")

    @property
    def eligible(self) -> bool:
        return all(item.passed for item in self.decisions)

    @property
    def passed(self) -> bool:
        """Compatibility alias used by orchestration code."""

        return self.eligible

    @property
    def failed_gate_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.decisions if not item.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "eligible": self.eligible,
            "decisions": [item.to_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class InterventionSnapshot:
    graph: GraphConfig
    cards: tuple[MechanismCard, ...]
    messages: tuple[MechanismMessage, ...]
    evidence: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphConfig):
            raise TypeError("graph must be GraphConfig")
        for name, values, kind in (
            ("cards", self.cards, MechanismCard),
            ("messages", self.messages, MechanismMessage),
            ("evidence", self.evidence, EvidenceRef),
        ):
            _tuple(values, name)
            if any(not isinstance(item, kind) for item in values):
                raise TypeError(f"{name} contains an invalid value")
        identifiers = (
            ("mechanism IDs", [item.mechanism_id for item in self.cards]),
            ("message IDs", [item.message_id for item in self.messages]),
            ("evidence IDs", [item.evidence_id for item in self.evidence]),
        )
        for label, values in identifiers:
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            _canonical(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuditChange:
    field: str
    before: str
    after: str

    def __post_init__(self) -> None:
        _identifier(self.field, "audit field")
        _text(self.before, "audit before", allow_empty=True)
        _text(self.after, "audit after", allow_empty=True)


@dataclass(frozen=True)
class InterventionAudit:
    kind: InterventionKind
    target_id: str
    before_digest: str
    after_digest: str
    changes: tuple[AuditChange, ...]
    requires_regeneration: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InterventionKind):
            raise TypeError("kind must be InterventionKind")
        _identifier(self.target_id, "target_id")
        for name, value in (
            ("before_digest", self.before_digest), ("after_digest", self.after_digest)
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        _tuple(self.changes, "changes")
        if not self.changes or any(not isinstance(item, AuditChange) for item in self.changes):
            raise ValueError("changes must contain at least one AuditChange")
        _bool(self.requires_regeneration, "requires_regeneration")


@dataclass(frozen=True)
class InterventionResult:
    snapshot: InterventionSnapshot
    audit: InterventionAudit

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, InterventionSnapshot):
            raise TypeError("snapshot must be InterventionSnapshot")
        if not isinstance(self.audit, InterventionAudit):
            raise TypeError("audit must be InterventionAudit")
        if self.snapshot.digest != self.audit.after_digest:
            raise ValueError("audit after_digest does not match the returned snapshot")


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_canonical(item) for item in value)
    return value


__all__ = [
    "ArbitrationVerdict", "AuditChange", "Direction", "Edge", "EdgeSpec",
    "EvidenceRef", "GateDecision", "GateReport", "GraphConfig",
    "InterventionAudit", "InterventionKind", "InterventionResult",
    "InterventionSnapshot", "MacroPhase", "MechanismCard", "MechanismMessage",
    "ObservationNode", "OrganEstimate", "PatientCase", "TreatmentAction",
    "TreatmentBinding", "Verdict",
]
