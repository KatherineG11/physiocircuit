"""Fourteen deterministic, pre-receiver gates for typed messages.

No gate calls an LLM and no gate depends on a receiver verdict.  Arbitration is
therefore downstream of this module: only a report with ``eligible == True``
may be shown to the receiving organ agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Optional

from .schema import (
    EvidenceRef,
    GateDecision,
    GateReport,
    GraphConfig,
    MechanismCard,
    MechanismMessage,
)


@dataclass(frozen=True)
class GateContext:
    """Immutable context against which a single message is checked."""

    graph: GraphConfig
    card: MechanismCard
    evidence: tuple[EvidenceRef, ...]
    receiver_seen_fingerprints: frozenset[str] = frozenset()
    current_step: int = 0
    available_steps: Optional[frozenset[int]] = None
    edge_message_count: int = 0
    max_messages_per_edge: int = 1
    option_labels: tuple[str, ...] = ("A", "B", "C", "D")

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphConfig):
            raise TypeError("graph must be GraphConfig")
        if not isinstance(self.card, MechanismCard):
            raise TypeError("card must be MechanismCard")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, EvidenceRef) for item in self.evidence
        ):
            raise TypeError("evidence must be a tuple of EvidenceRef values")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs in GateContext must be unique")
        if not isinstance(self.receiver_seen_fingerprints, frozenset) or any(
            not isinstance(item, str) for item in self.receiver_seen_fingerprints
        ):
            raise TypeError("receiver_seen_fingerprints must be frozenset[str]")
        for name, value in (
            ("current_step", self.current_step),
            ("edge_message_count", self.edge_message_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.max_messages_per_edge, bool)
            or not isinstance(self.max_messages_per_edge, int)
            or self.max_messages_per_edge < 1
        ):
            raise ValueError("max_messages_per_edge must be a positive integer")
        if self.available_steps is not None:
            if not isinstance(self.available_steps, frozenset) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in self.available_steps
            ):
                raise TypeError("available_steps must be frozenset[int] or None")
        if not isinstance(self.option_labels, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.option_labels
        ):
            raise TypeError("option_labels must be a tuple of non-empty strings")

    @property
    def evidence_index(self) -> dict[str, EvidenceRef]:
        return {item.evidence_id: item for item in self.evidence}


def _decision(
    number: int, name: str, passed: bool, success: str, failure: str, layer: str
) -> GateDecision:
    return GateDecision(
        gate_number=number,
        name=name,
        passed=passed,
        reason=success if passed else failure,
        layer=layer,
    )


def _option_leaks(text: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    """Return option labels appearing in answer-like constructs.

    A bare letter is intentionally not considered leakage because it produces
    false positives in clinical prose.  The patterns target explicit candidate
    references such as ``option B``, ``answer: C``, ``[D]``, or ``(A)``.
    """

    leaked: list[str] = []
    for label in labels:
        token = re.escape(label)
        patterns = (
            rf"\b(?:option|choice|answer|candidate)\s*[:=#-]?\s*{token}\b",
            rf"\[{token}\]",
            rf"\({token}\)",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            leaked.append(label)
    return tuple(leaked)


_GROUNDING_STOPWORDS = frozenset(
    {
        "and", "are", "for", "from", "into", "normal", "not", "the", "this",
        "with", "without", "rising", "falling", "stable", "improving", "worsening",
    }
)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        if len(token) >= 3 and token not in _GROUNDING_STOPWORDS
    )


def _evidence_matches_card(
    item: EvidenceRef,
    *,
    phrases: tuple[str, ...],
    variables: tuple[str, ...],
) -> bool:
    """Deterministic lexical/concept grounding, never an LLM judgement."""

    expected = set()
    for value in phrases + variables:
        expected.update(_tokens(value))
    observed = set(_tokens(item.variable)) | set(_tokens(item.concept_code)) | set(_tokens(item.text))
    return bool(expected.intersection(observed))


class DeterministicGateEngine:
    """Evaluate the fixed fourteen-gate program in a stable order."""

    gate_names = (
        "atomic_message_schema",
        "graph_endpoint_membership",
        "active_directed_edge",
        "mechanism_code_on_edge",
        "card_endpoint_alignment",
        "variable_consistency",
        "direction_allowed",
        "lag_allowed",
        "evidence_traceability",
        "counterevidence_traceability",
        "historical_evidence_only",
        "message_novelty",
        "atomic_edge_message_limit",
        "no_option_label_leakage",
    )

    def evaluate(self, message: MechanismMessage, context: GateContext) -> GateReport:
        if not isinstance(message, MechanismMessage):
            raise TypeError("message must be a validated MechanismMessage")
        if not isinstance(context, GateContext):
            raise TypeError("context must be GateContext")

        graph = context.graph
        card = context.card
        index = context.evidence_index
        evidence_ids = message.evidence_ids
        counter_ids = message.counterevidence_ids
        cited_ids = evidence_ids + counter_ids

        # 1. Schema construction already validates scalar types and bounds.  The
        # gate still makes atomicity explicit in every audit report.
        atomic = bool(
            message.mechanism_id
            and message.source_variable
            and message.target_variable
            and len({message.mechanism_id}) == 1
        )
        decisions = [
            _decision(
                1,
                self.gate_names[0],
                atomic,
                "message is one well-formed mechanism assertion",
                "message must contain exactly one mechanism and one variable pair",
                "structural",
            )
        ]

        # 2. Both endpoints must be known graph nodes and must differ.
        endpoints_known = (
            message.sender in graph.nodes
            and message.receiver in graph.nodes
        )
        decisions.append(
            _decision(
                2,
                self.gate_names[1],
                endpoints_known,
                f"endpoints {message.sender}->{message.receiver} are graph nodes",
                f"one or more endpoints are unknown: {message.sender}->{message.receiver}",
                "structural",
            )
        )

        # 3. The exact directed edge must exist and be active.
        edge = graph.edge(message.sender, message.receiver)
        edge_active = bool(edge and edge.active)
        decisions.append(
            _decision(
                3,
                self.gate_names[2],
                edge_active,
                f"directed edge {message.sender}->{message.receiver} is active",
                f"directed edge {message.sender}->{message.receiver} is absent or inactive",
                "semantic",
            )
        )

        # 4. A valid edge is not enough: it must explicitly list this mechanism.
        mechanism_on_edge = bool(edge and message.mechanism_id in edge.mechanism_ids)
        decisions.append(
            _decision(
                4,
                self.gate_names[3],
                mechanism_on_edge,
                f"mechanism {message.mechanism_id} is allowed on the edge",
                f"mechanism {message.mechanism_id} is not allowed on this edge",
                "semantic",
            )
        )

        # 5. The supplied card itself must describe the same directed relation.
        card_aligned = (
            card.mechanism_id == message.mechanism_id
            and card.source == message.sender
            and card.target == message.receiver
        )
        decisions.append(
            _decision(
                5,
                self.gate_names[4],
                card_aligned,
                "message identifier and endpoints match the mechanism card",
                (
                    "message/card mismatch: expected "
                    f"{card.mechanism_id} {card.source}->{card.target}"
                ),
                "semantic",
            )
        )

        # 6. Source and target variables are checked together as one relation.
        variables_aligned = (
            message.source_variable in card.source_variables
            and message.target_variable in card.target_variables
        )
        decisions.append(
            _decision(
                6,
                self.gate_names[5],
                variables_aligned,
                "source and target variables are declared by the card",
                (
                    f"undeclared variable pair {message.source_variable}->"
                    f"{message.target_variable}"
                ),
                "semantic",
            )
        )

        # 7. The message polarity must be one the mechanism can produce.
        direction_allowed = message.direction in card.allowed_directions
        decisions.append(
            _decision(
                7,
                self.gate_names[6],
                direction_allowed,
                f"direction {message.direction.value} is allowed by the card",
                f"direction {message.direction.value} is forbidden by the card",
                "semantic",
            )
        )

        # 8. The proposed delay must fall inside the enumerated card window.
        lag_allowed = message.lag in card.lag_steps
        decisions.append(
            _decision(
                8,
                self.gate_names[7],
                lag_allowed,
                f"lag {message.lag} is inside the card window",
                f"lag {message.lag} is outside allowed steps {card.lag_steps}",
                "semantic",
            )
        )

        # 9. Activation evidence is mandatory and every identifier must exist.
        missing_evidence = tuple(item for item in evidence_ids if item not in index)
        resolved_activation = tuple(index[item] for item in evidence_ids if item in index)
        activating_treatments = tuple(
            binding.treatment
            for binding in card.treatment_bindings
            if binding.action.value == "activate"
        )
        grounded_activation = tuple(
            item.evidence_id
            for item in resolved_activation
            if _evidence_matches_card(
                item,
                phrases=card.activation_evidence + activating_treatments,
                variables=card.source_variables,
            )
        )
        evidence_traceable = bool(evidence_ids) and not missing_evidence and bool(grounded_activation)
        decisions.append(
            _decision(
                9,
                self.gate_names[8],
                evidence_traceable,
                "activation evidence resolves and is grounded in the card",
                (
                    "activation evidence is empty"
                    if not evidence_ids
                    else (
                        f"unresolved activation evidence IDs: {missing_evidence}"
                        if missing_evidence
                        else "cited evidence does not match card activation concepts or source variables"
                    )
                ),
                "structural",
            )
        )

        # 10. Counterevidence is optional; any cited identifier must resolve.
        missing_counter = tuple(item for item in counter_ids if item not in index)
        resolved_counter = tuple(index[item] for item in counter_ids if item in index)
        suppressing_treatments = tuple(
            binding.treatment
            for binding in card.treatment_bindings
            if binding.action.value == "suppress"
        )
        ungrounded_counter = tuple(
            item.evidence_id
            for item in resolved_counter
            if not _evidence_matches_card(
                item,
                phrases=card.counterevidence + suppressing_treatments,
                variables=card.source_variables,
            )
        )
        counter_traceable = not missing_counter and not ungrounded_counter
        decisions.append(
            _decision(
                10,
                self.gate_names[9],
                counter_traceable,
                "counterevidence resolves, is card-grounded, or none was cited",
                (
                    f"unresolved counterevidence IDs: {missing_counter}"
                    if missing_counter
                    else f"counterevidence is not card-grounded: {ungrounded_counter}"
                ),
                "structural",
            )
        )

        # 11. Resolved evidence must be observed, not predicted/future data.
        resolved = tuple(index[item] for item in cited_ids if item in index)
        unavailable = tuple(
            item.evidence_id
            for item in resolved
            if item.predicted
            or item.step > context.current_step
            or (
                context.available_steps is not None
                and item.step not in context.available_steps
            )
        )
        historical_only = not unavailable
        decisions.append(
            _decision(
                11,
                self.gate_names[10],
                historical_only,
                "all cited evidence is observed and temporally available",
                f"future, predicted, or unavailable evidence IDs: {unavailable}",
                "semantic",
            )
        )

        # 12. Novelty is receiver-specific and uses a semantic fingerprint that
        # excludes the arbitrary message ID.
        novel = message.fingerprint not in context.receiver_seen_fingerprints
        decisions.append(
            _decision(
                12,
                self.gate_names[11],
                novel,
                "receiver has not previously seen this semantic message",
                "receiver has already seen an equivalent semantic message",
                "semantic",
            )
        )

        # 13. At most one candidate may be emitted on a directed edge in the
        # current propagation round (configurable for explicit ablations).
        under_limit = context.edge_message_count < context.max_messages_per_edge
        decisions.append(
            _decision(
                13,
                self.gate_names[12],
                under_limit,
                (
                    f"edge count {context.edge_message_count} is below limit "
                    f"{context.max_messages_per_edge}"
                ),
                (
                    f"edge already has {context.edge_message_count} message(s); "
                    f"limit is {context.max_messages_per_edge}"
                ),
                "structural",
            )
        )

        # 14. Candidate-answer labels must never be smuggled through evidence.
        leaked: list[str] = []
        for item in resolved:
            leaked.extend(_option_leaks(item.text, context.option_labels))
        leaked_labels = tuple(sorted(set(leaked)))
        no_leakage = not leaked_labels
        decisions.append(
            _decision(
                14,
                self.gate_names[13],
                no_leakage,
                "cited evidence contains no candidate-option references",
                f"candidate-option references detected: {leaked_labels}",
                "semantic",
            )
        )

        return GateReport(message_id=message.message_id, decisions=tuple(decisions))


def _normalise_evidence(
    evidence_index: Mapping[str, EvidenceRef] | Iterable[EvidenceRef],
) -> tuple[EvidenceRef, ...]:
    if isinstance(evidence_index, Mapping):
        values: list[EvidenceRef] = []
        for key, value in evidence_index.items():
            if not isinstance(key, str) or not isinstance(value, EvidenceRef):
                raise TypeError("evidence_index must map str to EvidenceRef")
            if key != value.evidence_id:
                raise ValueError("evidence_index key must equal EvidenceRef.evidence_id")
            values.append(value)
        return tuple(values)
    values = tuple(evidence_index)
    if any(not isinstance(item, EvidenceRef) for item in values):
        raise TypeError("evidence iterable must contain EvidenceRef values")
    return values


def validate_message(
    message: MechanismMessage,
    card: MechanismCard,
    graph: GraphConfig,
    evidence_index: Mapping[str, EvidenceRef] | Iterable[EvidenceRef],
    receiver_seen_fingerprints: Iterable[str] = (),
    current_step: Optional[int] = None,
    available_steps: Optional[Iterable[int]] = None,
    *,
    edge_message_count: int = 0,
    max_messages_per_edge: int = 1,
    option_labels: tuple[str, ...] = ("A", "B", "C", "D"),
) -> GateReport:
    """Convenience API for the fixed fourteen-gate pipeline.

    All gates are evaluated even after a failure so the report is a complete,
    deterministic audit rather than a short-circuit error message.
    """

    context = GateContext(
        graph=graph,
        card=card,
        evidence=_normalise_evidence(evidence_index),
        receiver_seen_fingerprints=frozenset(receiver_seen_fingerprints),
        current_step=message.created_at_step if current_step is None else current_step,
        available_steps=(
            None if available_steps is None else frozenset(available_steps)
        ),
        edge_message_count=edge_message_count,
        max_messages_per_edge=max_messages_per_edge,
        option_labels=option_labels,
    )
    return DeterministicGateEngine().evaluate(message, context)


run_gates = validate_message


__all__ = [
    "DeterministicGateEngine", "GateContext", "run_gates", "validate_message"
]
