"""Prompt construction with an explicit anti-leakage boundary.

Only public case fields and evidence available at the current step are ever
serialized.  Gold trajectories, design cards, expected intervention outcomes,
future observations, and benchmark strata are excluded by construction.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


_PUBLIC_CASE_KEYS = {
    "case_id",
    "domain",
    "field",
    "system",
    "patient",
    "presentation",
    "metadata",
}
_OBSERVATION_KEYS = {
    "node_id",
    "observation_id",
    "step",
    "t",
    "time",
    "time_label",
    "elapsed_steps",
    "elapsed_time",
    "predicted",
    "evidence",
    "text",
    "note",
    "symptoms",
    "labs",
    "vital_signs",
    "treatments",
    "interventions",
    "organ",
    "organs",
    "variable",
}
_BLOCKED_KEY_PARTS = (
    "gold",
    "ground_truth",
    "answer",
    "target_label",
    "design_card",
    "expected_outcome",
    "expected_response",
    "future_node",
    "probe",
    "fingerprint",
    "counterfactual",
    "stratum",
    "split",
    "backbone",
    "source_file",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def _is_blocked(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _BLOCKED_KEY_PARTS)


def _sanitize_nested(value: Any) -> Any:
    plain = _plain(value)
    if isinstance(plain, Mapping):
        return {
            str(key): _sanitize_nested(item)
            for key, item in plain.items()
            if not _is_blocked(str(key))
        }
    if isinstance(plain, list):
        return [_sanitize_nested(item) for item in plain]
    return plain


def public_case_view(case: Any, *, current_step: int) -> dict[str, Any]:
    """Return the only case representation that prompt builders may consume."""

    raw = _plain(case)
    if not isinstance(raw, Mapping):
        raise TypeError("case must be a mapping or dataclass")
    public: dict[str, Any] = {}
    for key in sorted(_PUBLIC_CASE_KEYS):
        if key in raw and not _is_blocked(key):
            if key == "metadata":
                metadata = raw[key]
                if isinstance(metadata, Mapping):
                    pairs = metadata.items()
                elif isinstance(metadata, list):
                    pairs = (
                        item for item in metadata
                        if isinstance(item, list) and len(item) == 2
                    )
                else:
                    pairs = ()
                public[key] = {
                    str(meta_key): _sanitize_nested(meta_value)
                    for meta_key, meta_value in pairs
                    if not _is_blocked(str(meta_key))
                }
            else:
                public[key] = _sanitize_nested(raw[key])

    nodes = raw.get("nodes", raw.get("observed_window", raw.get("observations", [])))
    visible_nodes: list[dict[str, Any]] = []
    for position, node in enumerate(nodes or []):
        node_map = _plain(node)
        if not isinstance(node_map, Mapping):
            continue
        node_step = node_map.get("step", node_map.get("t", position))
        try:
            if int(node_step) > int(current_step):
                continue
        except (TypeError, ValueError):
            continue
        visible = {
            str(key): _sanitize_nested(value)
            for key, value in node_map.items()
            if str(key) in _OBSERVATION_KEYS and not _is_blocked(str(key))
        }
        visible_nodes.append(visible)
    public["observations"] = visible_nodes
    return public


def evidence_public_view(evidence: Iterable[Any], *, current_step: int) -> list[dict[str, Any]]:
    """Serialize traceable evidence while dropping future evidence."""

    result: list[dict[str, Any]] = []
    for item in evidence:
        raw = _plain(item)
        if not isinstance(raw, Mapping):
            continue
        try:
            step = int(raw.get("step", 0))
        except (TypeError, ValueError):
            continue
        if step > current_step:
            continue
        allowed = {
            key: raw[key]
            for key in (
                "evidence_id",
                "organ",
                "variable",
                "step",
                "text",
                "concept_code",
                "predicted",
            )
            if key in raw and not _is_blocked(key)
        }
        result.append(_sanitize_nested(allowed))
    result.sort(key=lambda row: (int(row.get("step", 0)), str(row.get("evidence_id", ""))))
    return result


def mechanism_card_public_view(card: Any) -> dict[str, Any]:
    raw = _plain(card)
    if not isinstance(raw, Mapping):
        raise TypeError("mechanism card must be a mapping or dataclass")
    fields = (
        "mechanism_id",
        "source",
        "target",
        "source_variables",
        "target_variables",
        "allowed_directions",
        "lag_steps",
        "activation_evidence",
        "counterevidence",
        "treatment_bindings",
        "description",
        "mechanism_description",
    )
    return {key: _sanitize_nested(raw[key]) for key in fields if key in raw}


def _json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2)


def build_local_forecast_prompt(
    *,
    case: Any,
    organ: str,
    current_step: int,
    local_evidence: Sequence[Any],
    outgoing_cards: Sequence[Any],
    history: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    """Build the independent organ forecast prompt.

    Candidate global phases and benchmark labels are intentionally absent.
    """

    public_case = public_case_view(case, current_step=current_step)
    evidence = evidence_public_view(local_evidence, current_step=current_step)
    cards = [mechanism_card_public_view(card) for card in outgoing_cards]
    cards.sort(key=lambda row: str(row.get("mechanism_id", "")))
    safe_history = [_sanitize_nested(dict(item)) for item in history]
    system = (
        f"You are the local physiological agent for organ '{organ}'. "
        "Estimate only this organ's next-step direction from evidence available now. "
        "Do not infer benchmark labels, future observations, or a patient-level phase. "
        "Emit at most one atomic message per directed mechanism. Use only listed mechanism_id values "
        "and trace every claim to observed evidence_id values. Items marked predicted=true are rollout "
        "context and must not be cited in an outgoing message. Return JSON only, with no chain-of-thought.\n"
        "Required object:\n"
        '{"organ":"...","step":0,"summary":"...",'
        '"probabilities":{"improve":0.0,"stable":0.0,"worsen":0.0},'
        '"evidence_ids":["..."],"outgoing_messages":['
        '{"receiver":"...","mechanism_id":"...","source_variable":"...",'
        '"target_variable":"...","direction":"improve|stable|worsen",'
        '"strength":0.0,"lag":0,"confidence":0.0,'
        '"evidence_ids":["..."],"counterevidence_ids":[]}]}.'
    )
    user = (
        f"FORECAST_STEP={current_step}\n"
        f"PUBLIC_CASE=\n{_json(public_case)}\n"
        f"LOCAL_EVIDENCE=\n{_json(evidence)}\n"
        f"OWN_PRIOR_PREDICTIONS=\n{_json(safe_history)}\n"
        f"ALLOWED_OUTGOING_MECHANISM_CARDS=\n{_json(cards)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_arbitration_prompt(
    *,
    organ: str,
    current_step: int,
    initial_probabilities: Mapping[str, float],
    local_evidence: Sequence[Any],
    incoming_messages: Sequence[Any],
    relevant_cards: Sequence[Any],
) -> list[dict[str, str]]:
    """Build the receiver's post-gate arbitration prompt."""

    evidence = evidence_public_view(local_evidence, current_step=current_step)
    messages = [_sanitize_nested(_plain(message)) for message in incoming_messages]
    messages.sort(key=lambda row: str(row.get("message_id", "")))
    cards = [mechanism_card_public_view(card) for card in relevant_cards]
    cards.sort(key=lambda row: str(row.get("mechanism_id", "")))
    system = (
        f"You are the receiving physiological agent for organ '{organ}'. "
        "All incoming messages already passed deterministic gates. Decide each message independently "
        "as accept, partially_accept, or reject. Give a short auditable reason, not hidden reasoning. "
        "Then provide one conditioned local probability distribution. Return JSON only.\n"
        "Required object:\n"
        '{"verdicts":[{"message_id":"...","verdict":"accept|partially_accept|reject",'
        '"reason":"...","weight":0.0,"conflict_evidence_ids":[]}],'
        '"conditioned_probabilities":{"improve":0.0,"stable":0.0,"worsen":0.0}}.'
    )
    user = (
        f"FORECAST_STEP={current_step}\n"
        f"INITIAL_LOCAL_PROBABILITIES=\n{_json(dict(initial_probabilities))}\n"
        f"LOCAL_EVIDENCE=\n{_json(evidence)}\n"
        f"INCOMING_GATED_MESSAGES=\n{_json(messages)}\n"
        f"RELEVANT_MECHANISM_CARDS=\n{_json(cards)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_global_forecast_prompt(
    *,
    case: Any,
    organs: Sequence[str],
    current_step: int,
    history: Sequence[Mapping[str, Any]] = (),
    retrieved_cards: Sequence[Any] = (),
    intervention: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the A/B global-agent prompt with the same public case boundary."""

    public_case = public_case_view(case, current_step=current_step)
    cards = [mechanism_card_public_view(card) for card in retrieved_cards]
    cards.sort(key=lambda row: str(row.get("mechanism_id", "")))
    safe_intervention = _sanitize_nested(dict(intervention or {}))
    system = (
        "You are one global longitudinal clinical forecasting agent. Read the complete "
        "public trajectory prefix and predict every listed organ directly; do not simulate "
        "organ agents or hidden communication. Return JSON only and no chain-of-thought.\n"
        "Required object:\n"
        '{"organ_probabilities":{"organ":{"improve":0.0,"stable":0.0,"worsen":0.0}},'
        '"summaries":{"organ":"short reason"}}.'
    )
    user = (
        f"FORECAST_STEP={current_step}\n"
        f"ORGANS=\n{_json(sorted(str(item) for item in organs))}\n"
        f"PUBLIC_CASE=\n{_json(public_case)}\n"
        f"OWN_PRIOR_PREDICTIONS=\n{_json([_sanitize_nested(dict(item)) for item in history])}\n"
        f"RETRIEVED_MECHANISM_CARDS=\n{_json(cards)}\n"
        f"WHAT_IF_EDIT=\n{_json(safe_intervention)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_self_review_prompt(
    *,
    organ: str,
    current_step: int,
    initial_probabilities: Mapping[str, float],
    local_evidence: Sequence[Any],
    history: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    """Build the compute-matched no-graph self-review prompt."""

    evidence = evidence_public_view(local_evidence, current_step=current_step)
    system = (
        f"You are the local physiological agent for organ '{organ}'. Self-review your own "
        "estimate using only the same local evidence and prior local predictions. You receive "
        "no neighbor message, mechanism edge, or other organ's judgement. Return JSON only.\n"
        'Required object: {"conditioned_probabilities":'
        '{"improve":0.0,"stable":0.0,"worsen":0.0},"reason":"short audit reason"}.'
    )
    user = (
        f"FORECAST_STEP={current_step}\n"
        f"INITIAL_LOCAL_PROBABILITIES=\n{_json(dict(initial_probabilities))}\n"
        f"LOCAL_EVIDENCE=\n{_json(evidence)}\n"
        f"OWN_PRIOR_PREDICTIONS=\n{_json([_sanitize_nested(dict(item)) for item in history])}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_free_text_message_prompt(
    *,
    organ: str,
    current_step: int,
    local_evidence: Sequence[Any],
    receivers: Sequence[str],
    local_probabilities: Mapping[str, float],
) -> list[dict[str, str]]:
    """Build the untyped communication baseline's sender prompt."""

    evidence = evidence_public_view(local_evidence, current_step=current_step)
    system = (
        f"You are the local physiological agent for organ '{organ}'. Send a brief free-text "
        "clinical note to any listed neighboring organ that may need it. Do not use mechanism "
        "IDs, typed direction fields, lag fields, or card schemas. Return JSON only.\n"
        'Required object: {"messages":[{"receiver":"...","text":"...",'
        '"evidence_ids":["..."]}]}.'
    )
    user = (
        f"FORECAST_STEP={current_step}\n"
        f"ALLOWED_RECEIVERS=\n{_json(sorted(str(item) for item in receivers))}\n"
        f"LOCAL_PROBABILITIES=\n{_json(dict(local_probabilities))}\n"
        f"LOCAL_EVIDENCE=\n{_json(evidence)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_free_text_revision_prompt(
    *,
    organ: str,
    current_step: int,
    initial_probabilities: Mapping[str, float],
    local_evidence: Sequence[Any],
    incoming_messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build the free-text baseline receiver prompt without card semantics."""

    evidence = evidence_public_view(local_evidence, current_step=current_step)
    messages = [_sanitize_nested(dict(item)) for item in incoming_messages]
    system = (
        f"You are the receiving physiological agent for organ '{organ}'. Revise your local "
        "distribution after reading untyped neighbor notes. There are no mechanism cards or "
        "program gates in this baseline. Return JSON only.\n"
        'Required object: {"conditioned_probabilities":'
        '{"improve":0.0,"stable":0.0,"worsen":0.0},"reason":"short audit reason"}.'
    )
    user = (
        f"FORECAST_STEP={current_step}\n"
        f"INITIAL_LOCAL_PROBABILITIES=\n{_json(dict(initial_probabilities))}\n"
        f"LOCAL_EVIDENCE=\n{_json(evidence)}\n"
        f"UNTYPED_NEIGHBOR_NOTES=\n{_json(messages)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def assert_prompt_has_no_private_fields(messages: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed if a private benchmark field enters a rendered prompt."""

    rendered = "\n".join(str(message.get("content", "")) for message in messages).lower()
    forbidden = (
        '"gold"',
        '"ground_truth"',
        '"answer"',
        '"design_card"',
        '"expected_outcome"',
        '"future_nodes"',
        '"probe"',
        '"fingerprint"',
        '"stratum"',
    )
    leaked = [token for token in forbidden if token in rendered]
    if leaked:
        raise ValueError(f"private benchmark fields leaked into prompt: {leaked}")


__all__ = [
    "assert_prompt_has_no_private_fields",
    "build_arbitration_prompt",
    "build_global_forecast_prompt",
    "build_free_text_message_prompt",
    "build_free_text_revision_prompt",
    "build_local_forecast_prompt",
    "build_self_review_prompt",
    "evidence_public_view",
    "mechanism_card_public_view",
    "public_case_view",
]
