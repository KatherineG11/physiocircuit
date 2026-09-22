"""Pure, audited interventions over immutable PhysioCircuit snapshots."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Optional

from .gates import validate_message
from .schema import (
    AuditChange,
    Edge,
    EvidenceRef,
    GraphConfig,
    InterventionAudit,
    InterventionKind,
    InterventionResult,
    InterventionSnapshot,
    MechanismCard,
    MechanismMessage,
)


def _json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finish(
    before: InterventionSnapshot,
    after: InterventionSnapshot,
    kind: InterventionKind,
    target_id: str,
    changes: tuple[AuditChange, ...],
    *,
    requires_regeneration: bool = False,
) -> InterventionResult:
    if before.digest == after.digest:
        raise ValueError("an intervention must make an observable change")
    audit = InterventionAudit(
        kind=kind,
        target_id=target_id,
        before_digest=before.digest,
        after_digest=after.digest,
        changes=changes,
        requires_regeneration=requires_regeneration,
    )
    return InterventionResult(snapshot=after, audit=audit)


def _card_index(snapshot: InterventionSnapshot, mechanism_id: str) -> int:
    indices = [
        index
        for index, card in enumerate(snapshot.cards)
        if card.mechanism_id == mechanism_id
    ]
    if not indices:
        raise KeyError(f"unknown mechanism_id: {mechanism_id}")
    if len(indices) != 1:
        raise ValueError(f"mechanism_id is not unique: {mechanism_id}")
    return indices[0]


def _replace_card_and_edge(
    snapshot: InterventionSnapshot,
    old_id: str,
    replacement: MechanismCard,
) -> InterventionSnapshot:
    card_index = _card_index(snapshot, old_id)
    if replacement.source not in snapshot.graph.nodes or replacement.target not in snapshot.graph.nodes:
        raise ValueError("replacement mechanism endpoints must be graph nodes")
    for index, card in enumerate(snapshot.cards):
        if index != card_index and card.mechanism_id == replacement.mechanism_id:
            raise ValueError(
                f"replacement mechanism_id already exists: {replacement.mechanism_id}"
            )

    # Remove the old code from every edge, preserving unrelated mechanisms.
    edges: list[Edge] = []
    for edge in snapshot.graph.edges:
        remaining = tuple(item for item in edge.mechanism_ids if item != old_id)
        if remaining:
            edges.append(replace(edge, mechanism_ids=remaining))

    # Attach the replacement code to its declared directed edge.
    replacement_key = (replacement.source, replacement.target)
    matching = [
        index
        for index, edge in enumerate(edges)
        if (edge.source, edge.target) == replacement_key
    ]
    if matching:
        edge_index = matching[0]
        existing = edges[edge_index]
        if replacement.mechanism_id in existing.mechanism_ids:
            raise ValueError("replacement mechanism is already assigned to its edge")
        edges[edge_index] = replace(
            existing,
            mechanism_ids=existing.mechanism_ids + (replacement.mechanism_id,),
        )
    else:
        edges.append(
            Edge(
                source=replacement.source,
                target=replacement.target,
                mechanism_ids=(replacement.mechanism_id,),
                active=True,
            )
        )

    cards = list(snapshot.cards)
    cards[card_index] = replacement
    graph = replace(snapshot.graph, edges=tuple(edges))
    return replace(snapshot, graph=graph, cards=tuple(cards))


def delete_mechanism_message(
    snapshot: InterventionSnapshot, message_id: str
) -> InterventionResult:
    """Delete exactly one named message, leaving cards and evidence unchanged."""

    matches = [item for item in snapshot.messages if item.message_id == message_id]
    if not matches:
        raise KeyError(f"unknown message_id: {message_id}")
    messages = tuple(item for item in snapshot.messages if item.message_id != message_id)
    after = replace(snapshot, messages=messages)
    return _finish(
        snapshot,
        after,
        InterventionKind.DELETE,
        message_id,
        (
            AuditChange(
                field="messages",
                before=_json(matches[0].to_dict()),
                after="deleted",
            ),
        ),
    )


def replace_mechanism(
    snapshot: InterventionSnapshot,
    mechanism_id: str,
    replacement_card: MechanismCard,
) -> InterventionResult:
    """Replace one card and rewire its edge to the replacement endpoints.

    Existing messages are retained in the immutable snapshot for auditability,
    but the audit marks the result as requiring message regeneration before a
    normal inference run.  This prevents silently rewriting an LLM assertion.
    """

    old = snapshot.cards[_card_index(snapshot, mechanism_id)]
    if old == replacement_card:
        raise ValueError("replacement_card is identical to the existing card")
    after = _replace_card_and_edge(snapshot, mechanism_id, replacement_card)
    return _finish(
        snapshot,
        after,
        InterventionKind.REPLACE,
        mechanism_id,
        (
            AuditChange(
                field="mechanism_card",
                before=_json(old.to_dict()),
                after=_json(replacement_card.to_dict()),
            ),
            AuditChange(
                field="graph.edges",
                before=_json([edge.to_dict() for edge in snapshot.graph.edges]),
                after=_json([edge.to_dict() for edge in after.graph.edges]),
            ),
        ),
        requires_regeneration=True,
    )


def reverse_direction(
    snapshot: InterventionSnapshot, mechanism_id: str
) -> InterventionResult:
    """Reverse a mechanism's directed endpoints, not its improve/worsen polarity."""

    old = snapshot.cards[_card_index(snapshot, mechanism_id)]
    reversed_card = replace(
        old,
        source=old.target,
        target=old.source,
        source_variables=old.target_variables,
        target_variables=old.source_variables,
    )
    after = _replace_card_and_edge(snapshot, mechanism_id, reversed_card)
    return _finish(
        snapshot,
        after,
        InterventionKind.REVERSE,
        mechanism_id,
        (
            AuditChange(
                field="mechanism_direction",
                before=f"{old.source}->{old.target}",
                after=f"{reversed_card.source}->{reversed_card.target}",
            ),
            AuditChange(
                field="graph.edges",
                before=_json([edge.to_dict() for edge in snapshot.graph.edges]),
                after=_json([edge.to_dict() for edge in after.graph.edges]),
            ),
        ),
        requires_regeneration=True,
    )


def extend_lag(
    snapshot: InterventionSnapshot,
    mechanism_id: str,
    additional_steps: int,
) -> InterventionResult:
    """Delay a mechanism by shifting every allowed lag by a positive amount."""

    if (
        isinstance(additional_steps, bool)
        or not isinstance(additional_steps, int)
        or additional_steps <= 0
    ):
        raise ValueError("additional_steps must be a positive integer")
    old = snapshot.cards[_card_index(snapshot, mechanism_id)]
    delayed = replace(
        old,
        lag_steps=tuple(item + additional_steps for item in old.lag_steps),
    )
    after = replace(
        snapshot,
        cards=tuple(
            delayed if card.mechanism_id == mechanism_id else card
            for card in snapshot.cards
        ),
    )
    return _finish(
        snapshot,
        after,
        InterventionKind.EXTEND_LAG,
        mechanism_id,
        (
            AuditChange(
                field="lag_steps",
                before=_json(list(old.lag_steps)),
                after=_json(list(delayed.lag_steps)),
            ),
        ),
        requires_regeneration=True,
    )


def inject_wrong_message(
    snapshot: InterventionSnapshot, message: MechanismMessage
) -> InterventionResult:
    """Append a well-formed message that fails at least one deterministic gate."""

    if not isinstance(message, MechanismMessage):
        raise TypeError("message must be MechanismMessage")
    if any(item.message_id == message.message_id for item in snapshot.messages):
        raise ValueError(f"duplicate message_id: {message.message_id}")

    card = next(
        (item for item in snapshot.cards if item.mechanism_id == message.mechanism_id),
        None,
    )
    if card is not None:
        report = validate_message(
            message=message,
            card=card,
            graph=snapshot.graph,
            evidence_index=snapshot.evidence,
            current_step=message.created_at_step,
        )
        if report.eligible:
            raise ValueError(
                "inject_wrong_message requires a message that fails at least one gate"
            )
        failed = ",".join(report.failed_gate_names)
    else:
        failed = "mechanism_code_on_edge,card_endpoint_alignment"

    after = replace(snapshot, messages=snapshot.messages + (message,))
    return _finish(
        snapshot,
        after,
        InterventionKind.INJECT_WRONG_MESSAGE,
        message.message_id,
        (
            AuditChange(
                field="messages",
                before="not_present",
                after=_json(message.to_dict()),
            ),
            AuditChange(
                field="expected_failed_gates",
                before="none",
                after=failed,
            ),
        ),
    )


def mask_local_evidence(
    snapshot: InterventionSnapshot,
    organ: str,
    evidence_ids: Optional[tuple[str, ...]] = None,
) -> InterventionResult:
    """Remove selected observed evidence from one source organ.

    Message objects are intentionally not rewritten.  If masking is applied
    after message generation, their now-dangling citations are intercepted by
    the evidence-traceability gate; when applied before generation, the caller
    simply regenerates messages from the returned evidence set.
    """

    if not isinstance(organ, str) or not organ:
        raise ValueError("organ must be a non-empty string")
    if organ not in snapshot.graph.nodes:
        raise KeyError(f"unknown organ: {organ}")

    organ_items = tuple(item for item in snapshot.evidence if item.organ == organ)
    if evidence_ids is None:
        selected = tuple(item.evidence_id for item in organ_items)
    else:
        if not isinstance(evidence_ids, tuple) or any(
            not isinstance(item, str) or not item for item in evidence_ids
        ):
            raise TypeError("evidence_ids must be tuple[str, ...] or None")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence_ids must be unique")
        by_id = {item.evidence_id: item for item in snapshot.evidence}
        missing = tuple(item for item in evidence_ids if item not in by_id)
        if missing:
            raise KeyError(f"unknown evidence IDs: {missing}")
        wrong_organ = tuple(
            item for item in evidence_ids if by_id[item].organ != organ
        )
        if wrong_organ:
            raise ValueError(f"evidence does not belong to {organ}: {wrong_organ}")
        selected = evidence_ids
    if not selected:
        raise ValueError(f"no local evidence selected for organ {organ}")

    selected_set = set(selected)
    after = replace(
        snapshot,
        evidence=tuple(
            item for item in snapshot.evidence if item.evidence_id not in selected_set
        ),
    )
    return _finish(
        snapshot,
        after,
        InterventionKind.MASK_LOCAL_EVIDENCE,
        organ,
        (
            AuditChange(
                field="evidence_ids",
                before=_json(list(selected)),
                after="masked",
            ),
        ),
        requires_regeneration=True,
    )


def apply_intervention(
    kind: InterventionKind | str,
    snapshot: InterventionSnapshot,
    **parameters: Any,
) -> InterventionResult:
    """Dispatch one of the six named interventions with explicit parameters."""

    if isinstance(kind, str):
        try:
            kind = InterventionKind(kind)
        except ValueError as exc:
            raise ValueError(f"unknown intervention kind: {kind}") from exc
    if not isinstance(kind, InterventionKind):
        raise TypeError("kind must be InterventionKind or its exact string value")

    dispatch = {
        InterventionKind.DELETE: delete_mechanism_message,
        InterventionKind.REPLACE: replace_mechanism,
        InterventionKind.REVERSE: reverse_direction,
        InterventionKind.EXTEND_LAG: extend_lag,
        InterventionKind.INJECT_WRONG_MESSAGE: inject_wrong_message,
        InterventionKind.MASK_LOCAL_EVIDENCE: mask_local_evidence,
    }
    try:
        return dispatch[kind](snapshot, **parameters)
    except TypeError as exc:
        raise TypeError(
            f"invalid parameters for intervention {kind.value}: {exc}"
        ) from exc


# Exact paper-operation aliases, useful in CLI/config dispatch.
delete = delete_mechanism_message
replace_card = replace_mechanism
reverse = reverse_direction


__all__ = [
    "apply_intervention", "delete", "delete_mechanism_message", "extend_lag",
    "inject_wrong_message", "mask_local_evidence", "replace_card",
    "replace_mechanism", "reverse", "reverse_direction",
]
