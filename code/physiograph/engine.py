"""Configuration-only PhysioCircuit inference engine.

The engine follows the paper's causal order for every forecast step:

1. independent local organ estimates;
2. typed mechanism-message emission;
3. fourteen deterministic gates;
4. receiver arbitration;
5. bounded residual revision;
6. deterministic patient-level aggregation;
7. autoregressive roll-forward.

Gold labels are neither accepted by the engine nor serialized by the prompt
builders.  Evaluation is intentionally a separate concern.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .aggregation import (
    DIRECTIONS,
    aggregate_estimates,
    bounded_residual_update,
    normalize_probabilities,
)
from .llm import BaseLLMClient, LLMResponse
from .prompts import (
    assert_prompt_has_no_private_fields,
    build_arbitration_prompt,
    build_free_text_message_prompt,
    build_free_text_revision_prompt,
    build_global_forecast_prompt,
    build_local_forecast_prompt,
    build_self_review_prompt,
)
from .schema import (
    ArbitrationVerdict,
    Direction,
    EvidenceRef,
    GateDecision,
    GateReport,
    GraphConfig,
    MechanismCard,
    MechanismMessage,
    OrganEstimate,
    PatientCase,
    Verdict,
)


GateValidator = Callable[..., GateReport]
MessageTransform = Callable[[Sequence[MechanismMessage], int], Sequence[MechanismMessage]]


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def _clip(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, low), high)


def _gate_decision(number: int, name: str, passed: bool, reason: str, layer: str) -> GateDecision:
    """Construct against both the initial and final GateDecision contracts."""

    try:
        return GateDecision(number, name, passed, reason, layer=layer)  # type: ignore[call-arg]
    except TypeError:
        return GateDecision(number, name, passed, reason)


def _synthetic_gate_report(message_id: str, *, failure_number: int, reason: str) -> GateReport:
    names = (
        "json_object",
        "required_fields",
        "sender_active",
        "receiver_active",
        "edge_active",
        "mechanism_allowed",
        "source_variable_allowed",
        "target_variable_allowed",
        "direction_allowed",
        "lag_allowed",
        "evidence_traceable",
        "counterevidence_traceable",
        "message_novel",
        "no_private_label_leakage",
    )
    decisions = []
    for number, name in enumerate(names, 1):
        passed = number < failure_number
        gate_reason = "passed before structural failure" if passed else (
            reason if number == failure_number else "not evaluated after prior failure"
        )
        decisions.append(
            _gate_decision(
                number,
                name,
                passed,
                gate_reason,
                "structural" if number <= 2 else "semantic",
            )
        )
    return GateReport(message_id=message_id, decisions=tuple(decisions))


class PhysioCircuitEngine:
    """Run one frozen organ graph with one pluggable LLM backbone."""

    def __init__(
        self,
        *,
        graph: GraphConfig,
        cards: Sequence[MechanismCard],
        llm: BaseLLMClient,
        gate_validator: Optional[GateValidator] = None,
        message_transform: Optional[MessageTransform] = None,
        alpha_max: float = 0.6,
        allow_unbounded: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> None:
        if not isinstance(graph, GraphConfig):
            raise TypeError("graph must be GraphConfig")
        ceiling = 1.0 if allow_unbounded else 0.6
        if not 0.0 <= float(alpha_max) <= ceiling:
            raise ValueError(f"alpha_max must lie in [0, {ceiling}]")
        self.graph = graph
        self.cards = tuple(sorted(cards, key=lambda card: card.mechanism_id))
        self.card_index = {card.mechanism_id: card for card in self.cards}
        if len(self.card_index) != len(self.cards):
            raise ValueError("mechanism IDs must be unique")
        for card in self.cards:
            if card.source not in graph.nodes or card.target not in graph.nodes:
                raise ValueError(f"card {card.mechanism_id} refers to an unknown graph node")
            if not graph.allows(card.source, card.target, card.mechanism_id):
                raise ValueError(f"graph does not allow card {card.mechanism_id}")
        self.llm = llm
        self._gate_validator = gate_validator
        self.message_transform = message_transform
        self.alpha_max = float(alpha_max)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

    @property
    def gate_validator(self) -> GateValidator:
        if self._gate_validator is None:
            from .gates import validate_message

            self._gate_validator = validate_message
        return self._gate_validator

    def _cards_from(self, organ: str) -> tuple[MechanismCard, ...]:
        return tuple(card for card in self.cards if card.source == organ)

    def _evidence_for(self, organ: str, evidence_index: Mapping[str, EvidenceRef], step: int) -> tuple[EvidenceRef, ...]:
        shared_names = {"patient", "shared", "global", "public", "system"}
        values = [
            evidence
            for evidence in evidence_index.values()
            if evidence.step <= step and (evidence.organ == organ or evidence.organ in shared_names)
        ]
        return tuple(sorted(values, key=lambda item: (item.step, item.evidence_id)))

    @staticmethod
    def _initial_evidence(case: PatientCase) -> dict[str, EvidenceRef]:
        evidence: dict[str, EvidenceRef] = {}
        for node in case.observed_nodes:
            for item in node.evidence:
                if item.evidence_id in evidence:
                    raise ValueError(f"duplicate evidence ID across case nodes: {item.evidence_id}")
                evidence[item.evidence_id] = item
        return evidence

    def _call(
        self,
        prompts: Sequence[Mapping[str, Any]],
        *,
        role: str,
        organ: str,
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert_prompt_has_no_private_fields(prompts)
        response = self.llm.complete_json(
            prompts,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        if response.parsed is None:
            raise ValueError("JSON completion did not produce a parsed object")
        audit = {
            "role": role,
            "organ": organ,
            "step": step,
            "prompt": [dict(message) for message in prompts],
            "content": response.content,
            "parsed": _plain(response.parsed),
            "metadata": _plain(response.metadata),
            "raw_response": _plain(response.raw_response),
        }
        return dict(response.parsed), audit

    def _local_forecast(
        self,
        *,
        case: PatientCase,
        organ: str,
        step: int,
        evidence_index: Mapping[str, EvidenceRef],
        history: Sequence[Mapping[str, Any]],
        allow_outgoing_messages: bool = True,
    ) -> tuple[OrganEstimate, str, list[Mapping[str, Any]], dict[str, Any]]:
        local_evidence = self._evidence_for(organ, evidence_index, step)
        prompts = build_local_forecast_prompt(
            case=case,
            organ=organ,
            current_step=step,
            local_evidence=local_evidence,
            outgoing_cards=self._cards_from(organ) if allow_outgoing_messages else (),
            history=history,
        )
        data, audit = self._call(prompts, role="independent_local_forecast", organ=organ, step=step)
        probabilities = normalize_probabilities(data.get("probabilities", data))
        valid_evidence = set(evidence_index)
        evidence_ids = tuple(
            sorted({str(item) for item in data.get("evidence_ids", []) if str(item) in valid_evidence})
        )
        estimate = OrganEstimate(
            organ=organ,
            step=step,
            p_improve=probabilities["improve"],
            p_stable=probabilities["stable"],
            p_worsen=probabilities["worsen"],
            evidence_ids=evidence_ids,
        )
        outgoing = data.get("outgoing_messages", [])
        if not isinstance(outgoing, list):
            outgoing = []
        if not allow_outgoing_messages:
            outgoing = []
        summary = str(data.get("summary", ""))
        return estimate, summary, outgoing, audit

    @staticmethod
    def _raw_message_sort_key(item: tuple[int, Any]) -> tuple[str, str, str, str, int]:
        index, raw = item
        if not isinstance(raw, Mapping):
            return ("", "", "", "", index)
        return (
            str(raw.get("receiver", raw.get("target", ""))),
            str(raw.get("mechanism_id", "")),
            str(raw.get("source_variable", "")),
            str(raw.get("target_variable", "")),
            index,
        )

    def _parse_messages(
        self,
        *,
        case_id: str,
        sender: str,
        step: int,
        raw_messages: Sequence[Any],
    ) -> tuple[list[MechanismMessage], list[dict[str, Any]]]:
        messages: list[MechanismMessage] = []
        failures: list[dict[str, Any]] = []
        ordered = sorted(enumerate(raw_messages), key=self._raw_message_sort_key)
        for ordinal, (_, raw) in enumerate(ordered):
            message_id = f"{case_id}:s{step}:{sender}:m{ordinal:03d}"
            if not isinstance(raw, Mapping):
                report = _synthetic_gate_report(
                    message_id, failure_number=1, reason="outgoing message is not a JSON object"
                )
                failures.append({"raw": _plain(raw), "gate_report": report.to_dict()})
                continue
            payload = {
                "message_id": message_id,
                "sender": sender,
                "receiver": raw.get("receiver", raw.get("target")),
                "mechanism_id": raw.get("mechanism_id"),
                "source_variable": raw.get("source_variable"),
                "target_variable": raw.get("target_variable"),
                "direction": raw.get("direction"),
                "strength": raw.get("strength"),
                "lag": raw.get("lag", raw.get("expected_lag_steps")),
                "evidence_ids": raw.get("evidence_ids", raw.get("evidence_node_ids", [])),
                "counterevidence_ids": raw.get("counterevidence_ids", raw.get("counterevidence_node_ids", [])),
                "confidence": raw.get("confidence", 1.0),
                "created_at_step": step,
            }
            try:
                messages.append(MechanismMessage.from_dict(payload))
            except (TypeError, ValueError, KeyError) as exc:
                report = _synthetic_gate_report(message_id, failure_number=2, reason=str(exc))
                failures.append({"raw": _plain(raw), "gate_report": report.to_dict()})
        messages.sort(key=lambda message: message.message_id)
        return messages, failures

    def _validate_and_enqueue(
        self,
        *,
        messages: Sequence[MechanismMessage],
        step: int,
        horizon_end: int,
        evidence_index: Mapping[str, EvidenceRef],
        seen: Mapping[str, set[str]],
        queue: dict[int, list[MechanismMessage]],
    ) -> list[dict[str, Any]]:
        audits: list[dict[str, Any]] = []
        batch_seen = {organ: set(values) for organ, values in seen.items()}
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)
        available_steps = tuple(sorted({item.step for item in evidence_index.values() if item.step <= step}))
        for message in sorted(messages, key=lambda item: item.message_id):
            card = self.card_index.get(message.mechanism_id)
            if card is None:
                report = _synthetic_gate_report(
                    message.message_id,
                    failure_number=6,
                    reason="mechanism_id is absent from the frozen card library",
                )
            else:
                report = self.gate_validator(
                    message=message,
                    card=card,
                    graph=self.graph,
                    evidence_index=evidence_index,
                    receiver_seen_fingerprints=frozenset(batch_seen.get(message.receiver, set())),
                    current_step=step,
                    available_steps=available_steps,
                    edge_message_count=edge_counts[(message.sender, message.receiver)],
                    max_messages_per_edge=1,
                )
                if not isinstance(report, GateReport):
                    raise TypeError("gate_validator must return GateReport")
            due_step = step + message.lag
            if report.eligible:
                queue[due_step].append(message)
                batch_seen.setdefault(message.receiver, set()).add(message.fingerprint)
            audits.append(
                {
                    "message": message.to_dict(),
                    "gate_report": report.to_dict(),
                    "due_step": due_step,
                    "within_rollout": due_step <= horizon_end,
                }
            )
            edge_counts[(message.sender, message.receiver)] += 1
        return audits

    @staticmethod
    def _verdict_name(value: Any) -> str:
        aliases = {
            "accepted": "accept",
            "partially_accepted": "partially_accept",
            "partial": "partially_accept",
            "rejected": "reject",
        }
        text = str(value or "reject").strip().lower()
        return aliases.get(text, text)

    def _arbitrate(
        self,
        *,
        organ: str,
        step: int,
        initial: OrganEstimate,
        messages: Sequence[MechanismMessage],
        evidence_index: Mapping[str, EvidenceRef],
    ) -> tuple[OrganEstimate, list[ArbitrationVerdict], float, dict[str, Any]]:
        ordered = tuple(sorted(messages, key=lambda item: item.message_id))
        local_evidence = self._evidence_for(organ, evidence_index, step)
        cards = tuple(self.card_index[message.mechanism_id] for message in ordered)
        initial_prob = {
            "improve": initial.p_improve,
            "stable": initial.p_stable,
            "worsen": initial.p_worsen,
        }
        prompts = build_arbitration_prompt(
            organ=organ,
            current_step=step,
            initial_probabilities=initial_prob,
            local_evidence=local_evidence,
            incoming_messages=ordered,
            relevant_cards=cards,
        )
        data, audit = self._call(prompts, role="receiver_arbitration", organ=organ, step=step)
        raw_verdicts = data.get("verdicts", [])
        if not isinstance(raw_verdicts, list):
            raw_verdicts = []
        by_id: dict[str, Mapping[str, Any]] = {}
        for raw in raw_verdicts:
            if isinstance(raw, Mapping):
                message_id = str(raw.get("message_id", ""))
                if message_id and message_id not in by_id:
                    by_id[message_id] = raw

        evidence_ids = set(evidence_index)
        verdicts: list[ArbitrationVerdict] = []
        for message in ordered:
            raw = by_id.get(message.message_id)
            if raw is None:
                verdict = Verdict.REJECT
                reason = "receiver omitted a verdict"
                weight = 0.0
                conflicts: tuple[str, ...] = ()
            else:
                try:
                    verdict = Verdict(self._verdict_name(raw.get("verdict")))
                except ValueError:
                    verdict = Verdict.REJECT
                reason = str(raw.get("reason", "invalid or unsupported verdict")) or "no reason supplied"
                requested = _clip(raw.get("weight", 1.0 if verdict is Verdict.ACCEPT else 0.5))
                if verdict is Verdict.REJECT:
                    weight = 0.0
                elif verdict is Verdict.PARTIALLY_ACCEPT:
                    weight = min(requested, 0.5)
                else:
                    weight = requested
                conflicts = tuple(
                    sorted(
                        {
                            str(item)
                            for item in raw.get("conflict_evidence_ids", [])
                            if str(item) in evidence_ids
                        }
                    )
                )
            verdicts.append(
                ArbitrationVerdict(
                    message_id=message.message_id,
                    receiver=organ,
                    verdict=verdict,
                    reason=reason,
                    weight=weight,
                    conflict_evidence_ids=conflicts,
                )
            )

        accepted_weight = max((verdict.weight for verdict in verdicts), default=0.0)
        alpha = self.alpha_max * accepted_weight
        conditioned = normalize_probabilities(data.get("conditioned_probabilities", initial_prob))
        final_prob = bounded_residual_update(
            initial_prob,
            conditioned,
            alpha,
            alpha_max=self.alpha_max,
        )
        final = OrganEstimate(
            organ=organ,
            step=step,
            p_improve=final_prob["improve"],
            p_stable=final_prob["stable"],
            p_worsen=final_prob["worsen"],
            evidence_ids=initial.evidence_ids,
        )
        audit["effective_alpha"] = alpha
        audit["conditioned_probabilities"] = conditioned
        return final, verdicts, alpha, audit

    def run(self, case: PatientCase, *, horizon: Optional[int] = None) -> dict[str, Any]:
        """Autoregressively forecast one case without reading any gold field."""

        if not isinstance(case, PatientCase):
            raise TypeError("case must be PatientCase")
        if case.domain != self.graph.domain:
            raise ValueError(f"case domain {case.domain!r} does not match graph {self.graph.domain!r}")
        forecast_horizon = case.forecast_horizon if horizon is None else int(horizon)
        if forecast_horizon <= 0:
            raise ValueError("horizon must be positive")
        observed = case.observed_nodes
        start_step = max(node.step for node in observed)
        end_step = start_step + forecast_horizon
        evidence_index = self._initial_evidence(case)
        queue: dict[int, list[MechanismMessage]] = defaultdict(list)
        seen_fingerprints: dict[str, set[str]] = {organ: set() for organ in self.graph.nodes}
        organ_history: dict[str, list[dict[str, Any]]] = {organ: [] for organ in self.graph.nodes}
        previous_phase: Optional[str] = None
        step_records: list[dict[str, Any]] = []

        for step in range(start_step + 1, end_step + 1):
            initial: dict[str, OrganEstimate] = {}
            summaries: dict[str, str] = {}
            raw_outgoing: dict[str, list[Mapping[str, Any]]] = {}
            llm_audits: list[dict[str, Any]] = []
            parse_failures: list[dict[str, Any]] = []
            candidate_messages: list[MechanismMessage] = []

            for organ in sorted(self.graph.nodes):
                estimate, summary, outgoing, audit = self._local_forecast(
                    case=case,
                    organ=organ,
                    step=step,
                    evidence_index=evidence_index,
                    history=organ_history[organ],
                )
                initial[organ] = estimate
                summaries[organ] = summary
                raw_outgoing[organ] = list(outgoing)
                llm_audits.append(audit)
                parsed, failed = self._parse_messages(
                    case_id=case.case_id,
                    sender=organ,
                    step=step,
                    raw_messages=outgoing,
                )
                candidate_messages.extend(parsed)
                parse_failures.extend({"sender": organ, **item} for item in failed)

            if self.message_transform is not None:
                transformed = tuple(self.message_transform(tuple(candidate_messages), step))
                if any(not isinstance(item, MechanismMessage) for item in transformed):
                    raise TypeError("message_transform must return MechanismMessage values")
                transformed_ids = [item.message_id for item in transformed]
                if len(transformed_ids) != len(set(transformed_ids)):
                    raise ValueError("message_transform returned duplicate message IDs")
                candidate_messages = list(transformed)

            gate_audits = self._validate_and_enqueue(
                messages=candidate_messages,
                step=step,
                horizon_end=end_step,
                evidence_index=evidence_index,
                seen=seen_fingerprints,
                queue=queue,
            )
            due = sorted(queue.pop(step, []), key=lambda message: message.message_id)
            incoming_by_receiver: dict[str, list[MechanismMessage]] = defaultdict(list)
            for message in due:
                incoming_by_receiver[message.receiver].append(message)

            final: dict[str, OrganEstimate] = {}
            verdict_records: list[dict[str, Any]] = []
            alpha_by_organ: dict[str, float] = {}
            for organ in sorted(self.graph.nodes):
                incoming = incoming_by_receiver.get(organ, [])
                if incoming:
                    revised, verdicts, alpha, audit = self._arbitrate(
                        organ=organ,
                        step=step,
                        initial=initial[organ],
                        messages=incoming,
                        evidence_index=evidence_index,
                    )
                    final[organ] = revised
                    alpha_by_organ[organ] = alpha
                    llm_audits.append(audit)
                    verdict_records.extend(verdict.to_dict() for verdict in verdicts)
                    seen_fingerprints[organ].update(message.fingerprint for message in incoming)
                else:
                    final[organ] = initial[organ]
                    alpha_by_organ[organ] = 0.0

            aggregate = aggregate_estimates(
                final,
                graph=self.graph,
                previous_phase=previous_phase,
                step=step,
            )
            previous_phase = aggregate.phase

            for organ in sorted(self.graph.nodes):
                probabilities = {
                    "improve": final[organ].p_improve,
                    "stable": final[organ].p_stable,
                    "worsen": final[organ].p_worsen,
                }
                organ_history[organ].append(
                    {
                        "step": step,
                        "predicted": True,
                        "local_probabilities": probabilities,
                        "macro_phase": aggregate.phase,
                    }
                )
                evidence_id = f"pred:{case.case_id}:s{step}:{organ}"
                evidence_index[evidence_id] = EvidenceRef(
                    evidence_id=evidence_id,
                    organ=organ,
                    variable="direction_distribution",
                    step=step,
                    text=(
                        f"predicted improve={probabilities['improve']:.6f}, "
                        f"stable={probabilities['stable']:.6f}, worsen={probabilities['worsen']:.6f}"
                    ),
                    concept_code="predicted_direction",
                    predicted=True,
                )

            step_records.append(
                {
                    "step": step,
                    "initial_estimates": {organ: initial[organ].to_dict() for organ in sorted(initial)},
                    "local_summaries": {organ: summaries[organ] for organ in sorted(summaries)},
                    "message_parse_failures": parse_failures,
                    "gate_audits": gate_audits,
                    "delivered_messages": [message.to_dict() for message in due],
                    "arbitration_verdicts": sorted(verdict_records, key=lambda row: row["message_id"]),
                    "alpha_by_organ": {organ: alpha_by_organ[organ] for organ in sorted(alpha_by_organ)},
                    "final_estimates": {organ: final[organ].to_dict() for organ in sorted(final)},
                    "aggregate": aggregate.to_dict(),
                    "llm_calls": llm_audits,
                }
            )

        pending = [
            {"due_step": due_step, "message": message.to_dict()}
            for due_step in sorted(queue)
            for message in sorted(queue[due_step], key=lambda item: item.message_id)
        ]
        return {
            "case_id": case.case_id,
            "domain": case.domain,
            "model": getattr(self.llm, "model", "unknown"),
            "start_step": start_step,
            "forecast_horizon": forecast_horizon,
            "alpha_max": self.alpha_max,
            "steps": step_records,
            "pending_messages": pending,
        }


def _snapshot_for_case(
    graph: GraphConfig,
    cards: Sequence[MechanismCard],
    case: PatientCase,
    *,
    messages: Sequence[MechanismMessage] = (),
):
    from .schema import InterventionSnapshot

    evidence = tuple(item for node in case.observed_nodes for item in node.evidence)
    return InterventionSnapshot(
        graph=graph,
        cards=tuple(cards),
        messages=tuple(messages),
        evidence=evidence,
    )


def _normalise_intervention_kind(value: Mapping[str, Any]) -> str:
    raw = str(value.get("kind", value.get("type", ""))).strip().lower()
    aliases = {
        "lag": "extend_lag",
        "delay": "extend_lag",
        "extend lag": "extend_lag",
        "inject": "inject_wrong_message",
        "wrong_message": "inject_wrong_message",
        "mask": "mask_local_evidence",
        "replace_card": "replace",
    }
    return aliases.get(raw, raw)


def _mechanism_target(value: Mapping[str, Any]) -> str:
    target = value.get("mechanism_id", value.get("edge", value.get("target_id", "")))
    return str(target).strip()


def _case_with_evidence(case: PatientCase, evidence: Sequence[EvidenceRef]) -> PatientCase:
    allowed = {item.evidence_id for item in evidence}
    nodes = tuple(
        dataclasses.replace(
            node,
            evidence=tuple(item for item in node.evidence if item.evidence_id in allowed),
        )
        for node in case.nodes
    )
    return dataclasses.replace(case, nodes=nodes)


def _wrong_message_from_spec(
    spec: Mapping[str, Any],
    *,
    case: PatientCase,
    graph: GraphConfig,
    cards: Sequence[MechanismCard],
) -> MechanismMessage:
    raw = spec.get("message", spec.get("wrong_message"))
    start_step = max(node.step for node in case.observed_nodes) + 1
    if isinstance(raw, Mapping):
        payload = dict(raw)
        payload.setdefault("message_id", f"{case.case_id}:intervention:wrong")
        payload.setdefault("created_at_step", int(spec.get("step", start_step)))
        return MechanismMessage.from_dict(payload)

    mechanism_id = _mechanism_target(spec)
    card = next((item for item in cards if item.mechanism_id == mechanism_id), None)
    if card is None:
        raise ValueError("inject_wrong_message requires a known mechanism_id or a complete message")
    forbidden = next(
        (direction for direction in Direction if direction not in card.allowed_directions),
        None,
    )
    direction = forbidden or card.allowed_directions[0]
    lag = card.lag_steps[0] if forbidden is not None else max(card.lag_steps) + 1
    evidence = tuple(item for node in case.observed_nodes for item in node.evidence)
    candidate = next(
        (
            item
            for item in evidence
            if item.organ in {card.source, "patient", "shared", "global", "public", "system"}
        ),
        None,
    )
    evidence_ids = (candidate.evidence_id,) if candidate is not None else ("missing:evidence",)
    return MechanismMessage(
        message_id=f"{case.case_id}:intervention:wrong",
        sender=card.source,
        receiver=card.target,
        mechanism_id=card.mechanism_id,
        source_variable=card.source_variables[0],
        target_variable=card.target_variables[0],
        direction=direction,
        strength=1.0,
        lag=lag,
        evidence_ids=evidence_ids,
        confidence=1.0,
        created_at_step=int(spec.get("step", start_step)),
    )


def _prepare_intervention(
    *,
    graph: GraphConfig,
    cards: Sequence[MechanismCard],
    case: PatientCase,
    intervention: Mapping[str, Any],
) -> tuple[GraphConfig, tuple[MechanismCard, ...], PatientCase, Optional[MessageTransform], list[Any]]:
    """Apply one paper intervention without exposing its expected outcome."""

    from .interventions import (
        delete_mechanism_message,
        extend_lag,
        inject_wrong_message,
        mask_local_evidence,
        replace_mechanism,
        reverse_direction,
    )

    kind = _normalise_intervention_kind(intervention)
    snapshot = _snapshot_for_case(graph, cards, case)
    audits: list[Any] = []
    transform: Optional[MessageTransform] = None

    if kind == "replace":
        mechanism_id = _mechanism_target(intervention)
        raw_card = intervention.get("replacement_card", intervention.get("card"))
        if not isinstance(raw_card, Mapping):
            raise ValueError("replace requires replacement_card")
        result = replace_mechanism(snapshot, mechanism_id, MechanismCard.from_dict(raw_card))
        snapshot = result.snapshot
        audits.append(_plain(result.audit))
    elif kind == "reverse":
        result = reverse_direction(snapshot, _mechanism_target(intervention))
        snapshot = result.snapshot
        audits.append(_plain(result.audit))
    elif kind == "extend_lag":
        additional = intervention.get("additional_steps", intervention.get("delta", 1))
        result = extend_lag(snapshot, _mechanism_target(intervention), int(additional))
        snapshot = result.snapshot
        audits.append(_plain(result.audit))
    elif kind == "mask_local_evidence":
        mechanism_id = _mechanism_target(intervention)
        organ = str(intervention.get("organ", intervention.get("source", ""))).strip()
        if not organ and mechanism_id:
            card = next((item for item in snapshot.cards if item.mechanism_id == mechanism_id), None)
            organ = card.source if card is not None else ""
        raw_ids = intervention.get("evidence_ids")
        evidence_ids = tuple(str(item) for item in raw_ids) if isinstance(raw_ids, list) else None
        result = mask_local_evidence(snapshot, organ, evidence_ids)
        snapshot = result.snapshot
        case = _case_with_evidence(case, snapshot.evidence)
        audits.append(_plain(result.audit))
    elif kind == "inject_wrong_message":
        message = _wrong_message_from_spec(
            intervention,
            case=case,
            graph=snapshot.graph,
            cards=snapshot.cards,
        )
        result = inject_wrong_message(snapshot, message)
        snapshot = result.snapshot
        audits.append(_plain(result.audit))
        injected = result.snapshot.messages[-1]
        injected_once = False

        def inject_transform(
            messages: Sequence[MechanismMessage], step: int
        ) -> Sequence[MechanismMessage]:
            nonlocal injected_once
            if not injected_once and step == injected.created_at_step:
                injected_once = True
                return tuple(messages) + (injected,)
            return tuple(messages)

        transform = inject_transform
    elif kind == "delete":
        target_message_id = str(intervention.get("message_id", "")).strip()
        target_mechanism = _mechanism_target(intervention)
        requested_step = intervention.get("step")
        deleted_once = False

        def delete_transform(
            messages: Sequence[MechanismMessage], step: int
        ) -> Sequence[MechanismMessage]:
            nonlocal deleted_once
            if deleted_once or (requested_step is not None and int(requested_step) != step):
                return tuple(messages)
            selected = next(
                (
                    item
                    for item in sorted(messages, key=lambda candidate: candidate.message_id)
                    if (target_message_id and item.message_id == target_message_id)
                    or (target_mechanism and item.mechanism_id == target_mechanism)
                ),
                None,
            )
            if selected is None:
                return tuple(messages)
            runtime = _snapshot_for_case(
                snapshot.graph,
                snapshot.cards,
                case,
                messages=messages,
            )
            result = delete_mechanism_message(runtime, selected.message_id)
            audits.append(_plain(result.audit))
            deleted_once = True
            return result.snapshot.messages

        transform = delete_transform
    else:
        raise ValueError(f"unknown intervention kind: {kind!r}")

    return snapshot.graph, snapshot.cards, case, transform, audits


def _macro_burden(result: Mapping[str, Any]) -> float:
    steps = result.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("result has no forecast steps")
    aggregate = steps[-1].get("aggregate", {})
    probabilities = aggregate.get("probabilities", {}) if isinstance(aggregate, Mapping) else {}
    if not isinstance(probabilities, Mapping):
        raise ValueError("result has no aggregate probabilities")
    return float(probabilities.get("worsen", 0.0)) - float(probabilities.get("improve", 0.0))


def _append_predicted_evidence(
    *,
    case: PatientCase,
    step: int,
    estimates: Mapping[str, OrganEstimate],
    phase: str,
    evidence_index: dict[str, EvidenceRef],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    for organ in sorted(estimates):
        estimate = estimates[organ]
        probabilities = {
            "improve": estimate.p_improve,
            "stable": estimate.p_stable,
            "worsen": estimate.p_worsen,
        }
        histories[organ].append(
            {
                "step": step,
                "predicted": True,
                "local_probabilities": probabilities,
                "macro_phase": phase,
            }
        )
        evidence_id = f"pred:{case.case_id}:s{step}:{organ}"
        evidence_index[evidence_id] = EvidenceRef(
            evidence_id=evidence_id,
            organ=organ,
            variable="direction_distribution",
            step=step,
            text=(
                f"predicted improve={probabilities['improve']:.6f}, "
                f"stable={probabilities['stable']:.6f}, "
                f"worsen={probabilities['worsen']:.6f}"
            ),
            concept_code="predicted_direction",
            predicted=True,
        )


def _run_independent_organs(
    engine: PhysioCircuitEngine,
    case: PatientCase,
    *,
    self_review: bool,
) -> Mapping[str, Any]:
    """Run organ decomposition with no cross-organ message channel."""

    if case.domain != engine.graph.domain:
        raise ValueError("case and graph domains do not match")
    evidence_index = engine._initial_evidence(case)
    histories: dict[str, list[dict[str, Any]]] = {
        organ: [] for organ in engine.graph.nodes
    }
    start_step = max(node.step for node in case.observed_nodes)
    previous_phase: Optional[str] = None
    records: list[dict[str, Any]] = []
    for step in range(start_step + 1, start_step + case.forecast_horizon + 1):
        initial: dict[str, OrganEstimate] = {}
        final: dict[str, OrganEstimate] = {}
        summaries: dict[str, str] = {}
        llm_calls: list[dict[str, Any]] = []
        alpha_by_organ: dict[str, float] = {}
        for organ in sorted(engine.graph.nodes):
            estimate, summary, _, audit = engine._local_forecast(
                case=case,
                organ=organ,
                step=step,
                evidence_index=evidence_index,
                history=histories[organ],
                allow_outgoing_messages=False,
            )
            initial[organ] = estimate
            summaries[organ] = summary
            llm_calls.append(audit)
            if not self_review:
                final[organ] = estimate
                alpha_by_organ[organ] = 0.0
                continue
            prior = {
                "improve": estimate.p_improve,
                "stable": estimate.p_stable,
                "worsen": estimate.p_worsen,
            }
            prompts = build_self_review_prompt(
                organ=organ,
                current_step=step,
                initial_probabilities=prior,
                local_evidence=engine._evidence_for(organ, evidence_index, step),
                history=histories[organ],
            )
            data, review_audit = engine._call(
                prompts,
                role="compute_matched_self_review",
                organ=organ,
                step=step,
            )
            conditioned = normalize_probabilities(data.get("conditioned_probabilities", prior))
            revised = bounded_residual_update(
                prior,
                conditioned,
                engine.alpha_max,
                alpha_max=engine.alpha_max,
            )
            final[organ] = OrganEstimate(
                organ=organ,
                step=step,
                p_improve=revised["improve"],
                p_stable=revised["stable"],
                p_worsen=revised["worsen"],
                evidence_ids=estimate.evidence_ids,
            )
            alpha_by_organ[organ] = engine.alpha_max
            review_audit["effective_alpha"] = engine.alpha_max
            review_audit["conditioned_probabilities"] = conditioned
            llm_calls.append(review_audit)

        aggregate = aggregate_estimates(
            final,
            graph=engine.graph,
            previous_phase=previous_phase,
            step=step,
        )
        previous_phase = aggregate.phase
        _append_predicted_evidence(
            case=case,
            step=step,
            estimates=final,
            phase=aggregate.phase,
            evidence_index=evidence_index,
            histories=histories,
        )
        records.append(
            {
                "step": step,
                "initial_estimates": {key: initial[key].to_dict() for key in sorted(initial)},
                "local_summaries": {key: summaries[key] for key in sorted(summaries)},
                "message_parse_failures": [],
                "gate_audits": [],
                "delivered_messages": [],
                "arbitration_verdicts": [],
                "alpha_by_organ": alpha_by_organ,
                "final_estimates": {key: final[key].to_dict() for key in sorted(final)},
                "aggregate": aggregate.to_dict(),
                "llm_calls": llm_calls,
            }
        )
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "model": getattr(engine.llm, "model", "unknown"),
        "start_step": start_step,
        "forecast_horizon": case.forecast_horizon,
        "alpha_max": engine.alpha_max,
        "baseline_mode": "compute_matched_self_review" if self_review else "independent_organs",
        "steps": records,
        "pending_messages": [],
    }


def _retrieve_cards(case: PatientCase, cards: Sequence[MechanismCard], limit: int = 6) -> tuple[MechanismCard, ...]:
    text = " ".join(
        item.text.lower()
        for node in case.observed_nodes
        for item in node.evidence
    )
    patient_tokens = set(text.replace("_", " ").split())

    def score(card: MechanismCard) -> tuple[int, str]:
        card_text = " ".join(
            card.source_variables
            + card.target_variables
            + card.activation_evidence
            + card.counterevidence
            + (card.mechanism_description,)
        ).lower()
        card_tokens = set(card_text.replace("_", " ").split())
        return (-len(patient_tokens.intersection(card_tokens)), card.mechanism_id)

    return tuple(sorted(cards, key=score)[: max(0, int(limit))])


def _run_global_agent(
    engine: PhysioCircuitEngine,
    case: PatientCase,
    *,
    use_rag: bool,
    intervention: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if case.domain != engine.graph.domain:
        raise ValueError("case and graph domains do not match")
    start_step = max(node.step for node in case.observed_nodes)
    previous_phase: Optional[str] = None
    history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    retrieved = _retrieve_cards(case, engine.cards) if use_rag else ()
    for step in range(start_step + 1, start_step + case.forecast_horizon + 1):
        prompts = build_global_forecast_prompt(
            case=case,
            organs=engine.graph.nodes,
            current_step=step,
            history=history,
            retrieved_cards=retrieved,
            intervention=intervention,
        )
        data, audit = engine._call(
            prompts,
            role="global_forecast_rag" if use_rag else "global_forecast",
            organ="patient",
            step=step,
        )
        raw_probabilities = data.get("organ_probabilities", data.get("organs", {}))
        raw_probabilities = raw_probabilities if isinstance(raw_probabilities, Mapping) else {}
        estimates: dict[str, OrganEstimate] = {}
        for organ in sorted(engine.graph.nodes):
            raw = raw_probabilities.get(organ, {"stable": 1.0})
            if isinstance(raw, Mapping) and isinstance(raw.get("probabilities"), Mapping):
                raw = raw["probabilities"]
            probabilities = normalize_probabilities(raw if isinstance(raw, Mapping) else {})
            estimates[organ] = OrganEstimate(
                organ=organ,
                step=step,
                p_improve=probabilities["improve"],
                p_stable=probabilities["stable"],
                p_worsen=probabilities["worsen"],
                evidence_ids=(),
            )
        aggregate = aggregate_estimates(
            estimates,
            graph=engine.graph,
            previous_phase=previous_phase,
            step=step,
        )
        previous_phase = aggregate.phase
        history.append(
            {
                "step": step,
                "predicted": True,
                "organ_probabilities": {
                    organ: {
                        "improve": estimates[organ].p_improve,
                        "stable": estimates[organ].p_stable,
                        "worsen": estimates[organ].p_worsen,
                    }
                    for organ in sorted(estimates)
                },
                "macro_phase": aggregate.phase,
            }
        )
        records.append(
            {
                "step": step,
                "initial_estimates": {organ: estimates[organ].to_dict() for organ in sorted(estimates)},
                "local_summaries": data.get("summaries", {}),
                "message_parse_failures": [],
                "gate_audits": [],
                "delivered_messages": [],
                "arbitration_verdicts": [],
                "alpha_by_organ": {organ: 0.0 for organ in sorted(estimates)},
                "final_estimates": {organ: estimates[organ].to_dict() for organ in sorted(estimates)},
                "aggregate": aggregate.to_dict(),
                "llm_calls": [audit],
            }
        )
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "model": getattr(engine.llm, "model", "unknown"),
        "start_step": start_step,
        "forecast_horizon": case.forecast_horizon,
        "alpha_max": 0.0,
        "baseline_mode": "global_card_rag" if use_rag else "global_no_rag",
        "retrieved_mechanism_ids": [card.mechanism_id for card in retrieved],
        "steps": records,
        "pending_messages": [],
    }


def _run_free_text_graph(
    engine: PhysioCircuitEngine,
    case: PatientCase,
) -> Mapping[str, Any]:
    """Run topology-preserving, untyped communication with no program gates."""

    if case.domain != engine.graph.domain:
        raise ValueError("case and graph domains do not match")
    evidence_index = engine._initial_evidence(case)
    histories: dict[str, list[dict[str, Any]]] = {
        organ: [] for organ in engine.graph.nodes
    }
    start_step = max(node.step for node in case.observed_nodes)
    previous_phase: Optional[str] = None
    records: list[dict[str, Any]] = []
    for step in range(start_step + 1, start_step + case.forecast_horizon + 1):
        initial: dict[str, OrganEstimate] = {}
        summaries: dict[str, str] = {}
        llm_calls: list[dict[str, Any]] = []
        for organ in sorted(engine.graph.nodes):
            estimate, summary, _, audit = engine._local_forecast(
                case=case,
                organ=organ,
                step=step,
                evidence_index=evidence_index,
                history=histories[organ],
                allow_outgoing_messages=False,
            )
            initial[organ] = estimate
            summaries[organ] = summary
            llm_calls.append(audit)

        delivered: list[dict[str, Any]] = []
        incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sender in sorted(engine.graph.nodes):
            receivers = tuple(
                sorted(
                    {
                        edge.target
                        for edge in engine.graph.edges
                        if edge.active and edge.source == sender
                    }
                )
            )
            if not receivers:
                continue
            estimate = initial[sender]
            prompts = build_free_text_message_prompt(
                organ=sender,
                current_step=step,
                local_evidence=engine._evidence_for(sender, evidence_index, step),
                receivers=receivers,
                local_probabilities={
                    "improve": estimate.p_improve,
                    "stable": estimate.p_stable,
                    "worsen": estimate.p_worsen,
                },
            )
            data, audit = engine._call(
                prompts,
                role="free_text_sender",
                organ=sender,
                step=step,
            )
            llm_calls.append(audit)
            raw_messages = data.get("messages", [])
            if not isinstance(raw_messages, list):
                continue
            used_receivers: set[str] = set()
            for raw in raw_messages:
                if not isinstance(raw, Mapping):
                    continue
                receiver = str(raw.get("receiver", ""))
                text = str(raw.get("text", "")).strip()
                if receiver not in receivers or receiver in used_receivers or not text:
                    continue
                used_receivers.add(receiver)
                evidence_ids = sorted(
                    {
                        str(item)
                        for item in raw.get("evidence_ids", [])
                        if str(item) in evidence_index
                    }
                )
                message = {
                    "message_id": f"{case.case_id}:s{step}:{sender}:free:{len(delivered):03d}",
                    "sender": sender,
                    "receiver": receiver,
                    "text": text,
                    "evidence_ids": evidence_ids,
                }
                delivered.append(message)
                incoming[receiver].append(message)

        final: dict[str, OrganEstimate] = {}
        alpha_by_organ: dict[str, float] = {}
        arbitration: list[dict[str, Any]] = []
        for organ in sorted(engine.graph.nodes):
            messages = incoming.get(organ, [])
            if not messages:
                final[organ] = initial[organ]
                alpha_by_organ[organ] = 0.0
                continue
            prior = {
                "improve": initial[organ].p_improve,
                "stable": initial[organ].p_stable,
                "worsen": initial[organ].p_worsen,
            }
            prompts = build_free_text_revision_prompt(
                organ=organ,
                current_step=step,
                initial_probabilities=prior,
                local_evidence=engine._evidence_for(organ, evidence_index, step),
                incoming_messages=messages,
            )
            data, audit = engine._call(
                prompts,
                role="free_text_receiver",
                organ=organ,
                step=step,
            )
            conditioned = normalize_probabilities(data.get("conditioned_probabilities", prior))
            revised = bounded_residual_update(
                prior,
                conditioned,
                engine.alpha_max,
                alpha_max=engine.alpha_max,
            )
            final[organ] = OrganEstimate(
                organ=organ,
                step=step,
                p_improve=revised["improve"],
                p_stable=revised["stable"],
                p_worsen=revised["worsen"],
                evidence_ids=initial[organ].evidence_ids,
            )
            alpha_by_organ[organ] = engine.alpha_max
            audit["effective_alpha"] = engine.alpha_max
            audit["conditioned_probabilities"] = conditioned
            llm_calls.append(audit)
            arbitration.extend(
                {
                    "message_id": message["message_id"],
                    "receiver": organ,
                    "verdict": "free_text_consumed",
                    "reason": str(data.get("reason", "untyped neighbor note reviewed")),
                    "weight": 1.0,
                }
                for message in messages
            )

        aggregate = aggregate_estimates(
            final,
            graph=engine.graph,
            previous_phase=previous_phase,
            step=step,
        )
        previous_phase = aggregate.phase
        _append_predicted_evidence(
            case=case,
            step=step,
            estimates=final,
            phase=aggregate.phase,
            evidence_index=evidence_index,
            histories=histories,
        )
        records.append(
            {
                "step": step,
                "initial_estimates": {organ: initial[organ].to_dict() for organ in sorted(initial)},
                "local_summaries": summaries,
                "message_parse_failures": [],
                "gate_audits": [],
                "delivered_messages": delivered,
                "arbitration_verdicts": arbitration,
                "alpha_by_organ": alpha_by_organ,
                "final_estimates": {organ: final[organ].to_dict() for organ in sorted(final)},
                "aggregate": aggregate.to_dict(),
                "llm_calls": llm_calls,
            }
        )
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "model": getattr(engine.llm, "model", "unknown"),
        "start_step": start_step,
        "forecast_horizon": case.forecast_horizon,
        "alpha_max": engine.alpha_max,
        "baseline_mode": "free_text_graph",
        "steps": records,
        "pending_messages": [],
    }


def _pass_all_gates(**kwargs: Any) -> GateReport:
    from .gates import DeterministicGateEngine

    message = kwargs["message"]
    decisions = tuple(
        GateDecision(
            gate_number=index,
            name=name,
            passed=True,
            reason="gate bypassed by explicit ablation",
            layer="structural" if index in {1, 2, 9, 10, 13} else "semantic",
        )
        for index, name in enumerate(DeterministicGateEngine.gate_names, 1)
    )
    return GateReport(message_id=message.message_id, decisions=decisions)


def _cards_for_graph(
    graph: GraphConfig,
    cards: Sequence[MechanismCard],
) -> tuple[MechanismCard, ...]:
    index = {card.mechanism_id: card for card in cards}
    output: list[MechanismCard] = []
    for edge in graph.edges:
        for mechanism_id in edge.mechanism_ids:
            card = index.get(mechanism_id)
            if card is None:
                raise ValueError(f"graph variant references missing card {mechanism_id}")
            output.append(dataclasses.replace(card, source=edge.source, target=edge.target))
    return tuple(output)


def _mutate_graph_bundle(
    graph: GraphConfig,
    cards: Sequence[MechanismCard],
    *,
    mutation: str,
    seed: int,
) -> Mapping[str, Any]:
    from .schema import Edge

    mutation = str(mutation)
    edges = list(graph.edges)
    if mutation == "reversed":
        edges = [
            Edge(edge.target, edge.source, edge.mechanism_ids, edge.active)
            for edge in edges
        ]
    elif mutation == "edge_deleted":
        if not edges:
            raise ValueError("cannot delete an edge from an empty graph")
        edges = sorted(edges, key=lambda edge: (edge.source, edge.target))[1:]
    elif mutation == "random_degree_matched":
        rng = random.Random(int(seed))
        attempts = max(20, 20 * len(edges))
        for _ in range(attempts):
            if len(edges) < 2:
                break
            left_index, right_index = rng.sample(range(len(edges)), 2)
            left, right = edges[left_index], edges[right_index]
            proposed = ((left.source, right.target), (right.source, left.target))
            if proposed[0][0] == proposed[0][1] or proposed[1][0] == proposed[1][1]:
                continue
            other_pairs = {
                (edge.source, edge.target)
                for index, edge in enumerate(edges)
                if index not in {left_index, right_index}
            }
            if proposed[0] in other_pairs or proposed[1] in other_pairs or proposed[0] == proposed[1]:
                continue
            edges[left_index] = Edge(
                left.source, right.target, left.mechanism_ids, left.active
            )
            edges[right_index] = Edge(
                right.source, left.target, right.mechanism_ids, right.active
            )
    elif mutation == "fully_connected":
        existing = {(edge.source, edge.target): edge for edge in edges}
        card_index = {card.mechanism_id: card for card in cards}
        expanded_cards = list(cards)
        expanded_edges: list[Edge] = []
        template = sorted(cards, key=lambda card: card.mechanism_id)[0]
        for source in sorted(graph.nodes):
            for target in sorted(graph.nodes):
                if source == target:
                    continue
                current = existing.get((source, target))
                if current is not None:
                    expanded_edges.append(current)
                    continue
                mechanism_id = f"FC:{source}:{target}"
                synthetic = dataclasses.replace(
                    template,
                    mechanism_id=mechanism_id,
                    source=source,
                    target=target,
                    mechanism_description=(
                        "Topology-control synthetic edge; physiology intentionally not curated."
                    ),
                )
                card_index[mechanism_id] = synthetic
                expanded_cards.append(synthetic)
                expanded_edges.append(Edge(source, target, (mechanism_id,), True))
        variant = dataclasses.replace(graph, edges=tuple(expanded_edges))
        referenced = {item for edge in variant.edges for item in edge.mechanism_ids}
        selected = tuple(card for card in expanded_cards if card.mechanism_id in referenced)
        return {"graph": variant, "cards": selected, "mutation": mutation, "seed": int(seed)}
    else:
        raise ValueError(f"unknown graph mutation: {mutation}")

    variant = dataclasses.replace(graph, edges=tuple(edges))
    selected = _cards_for_graph(variant, cards)
    return {"graph": variant, "cards": selected, "mutation": mutation, "seed": int(seed)}


def _aggregate_local_result(
    result: Mapping[str, Any],
    graph: GraphConfig,
    weights: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    active_graph = graph
    if weights:
        active_graph = dataclasses.replace(
            graph,
            aggregation_weights=tuple(sorted((str(key), float(value)) for key, value in weights.items())),
        )
    output = dict(result)
    previous_phase: Optional[str] = None
    steps: list[dict[str, Any]] = []
    for step_record in result.get("steps", []):
        record = dict(step_record)
        estimates = record.get("final_estimates", {})
        aggregate = aggregate_estimates(
            estimates,
            graph=active_graph,
            previous_phase=previous_phase,
            step=int(record.get("step", 0)),
        )
        previous_phase = aggregate.phase
        record["aggregate"] = aggregate.to_dict()
        steps.append(record)
    output["steps"] = steps
    return output


def build_adapter(
    *,
    graph: GraphConfig,
    cards: Sequence[MechanismCard],
    llm: BaseLLMClient,
    alpha_max: float = 0.6,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    **_: Any,
) -> Mapping[str, Any]:
    """Build the runner adapter, including executable intervention semantics."""

    base_cards = tuple(cards)

    def make_engine(
        active_graph: GraphConfig = graph,
        active_cards: Sequence[MechanismCard] = base_cards,
        *,
        gate_validator: Optional[GateValidator] = None,
        message_transform: Optional[MessageTransform] = None,
        active_alpha_max: Optional[float] = None,
        allow_unbounded: bool = False,
    ) -> PhysioCircuitEngine:
        return PhysioCircuitEngine(
            graph=active_graph,
            cards=tuple(active_cards),
            llm=llm,
            gate_validator=gate_validator,
            message_transform=message_transform,
            alpha_max=alpha_max if active_alpha_max is None else active_alpha_max,
            allow_unbounded=allow_unbounded,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def run_pg(
        case: PatientCase,
        intervention: Optional[Mapping[str, Any]] = None,
        **__: Any,
    ) -> Mapping[str, Any]:
        active_graph = graph
        active_cards = base_cards
        active_case = case
        transform: Optional[MessageTransform] = None
        audits: list[Any] = []
        if intervention is not None:
            active_graph, active_cards, active_case, transform, audits = _prepare_intervention(
                graph=graph,
                cards=base_cards,
                case=case,
                intervention=intervention,
            )
        active_engine = make_engine(
            active_graph,
            active_cards,
            message_transform=transform,
        )
        result = dict(active_engine.run(active_case))
        if intervention is not None:
            result["intervention"] = {
                "spec": {key: value for key, value in intervention.items() if "expected" not in key.lower()},
                "audits": audits,
            }
        return result

    def run_global(
        case: PatientCase,
        *,
        use_rag: bool = False,
        intervention: Optional[Mapping[str, Any]] = None,
        **__: Any,
    ) -> Mapping[str, Any]:
        return _run_global_agent(
            make_engine(),
            case,
            use_rag=bool(use_rag),
            intervention=intervention,
        )

    def run_organs(case: PatientCase, **__: Any) -> Mapping[str, Any]:
        return _run_independent_organs(make_engine(), case, self_review=False)

    def aggregate(
        *,
        estimates: Mapping[str, Any],
        weights: Optional[Mapping[str, Any]] = None,
        **__: Any,
    ) -> Mapping[str, Any]:
        return _aggregate_local_result(estimates, graph, weights)

    def run_compute_matched(case: PatientCase, **__: Any) -> Mapping[str, Any]:
        return _run_independent_organs(make_engine(), case, self_review=True)

    def mutate_graph(
        *,
        graph: GraphConfig,
        mutation: str,
        seed: int = 0,
        **__: Any,
    ) -> Mapping[str, Any]:
        return _mutate_graph_bundle(
            graph,
            base_cards,
            mutation=mutation,
            seed=seed,
        )

    def run_graph(
        case: PatientCase,
        *,
        graph: Any,
        cards: Sequence[MechanismCard] = base_cards,
        message_mode: str = "typed",
        gate_mode: str = "full",
        intervention: Optional[Mapping[str, Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
        **__: Any,
    ) -> Mapping[str, Any]:
        if isinstance(graph, Mapping):
            active_graph = graph.get("graph")
            active_cards = tuple(graph.get("cards", cards))
        else:
            active_graph = graph
            active_cards = tuple(cards)
        if not isinstance(active_graph, GraphConfig):
            raise TypeError("run_graph requires GraphConfig or a graph variant bundle")
        # Filter/remap the supplied library when a topology control changed
        # endpoints or removed an edge.
        if not all(active_graph.allows(card.source, card.target, card.mechanism_id) for card in active_cards):
            active_cards = _cards_for_graph(active_graph, active_cards)
        active_case = case
        transform: Optional[MessageTransform] = None
        audits: list[Any] = []
        if intervention is not None:
            active_graph, active_cards, active_case, transform, audits = _prepare_intervention(
                graph=active_graph,
                cards=active_cards,
                case=case,
                intervention=intervention,
            )
        option_values = dict(options or {})
        requested_alpha = float(option_values.get("alpha_max", alpha_max))
        if message_mode == "free_text":
            result = dict(
                _run_free_text_graph(
                    make_engine(
                        active_graph,
                        active_cards,
                        active_alpha_max=requested_alpha,
                        allow_unbounded=requested_alpha > 0.6,
                    ),
                    active_case,
                )
            )
            result["ablation"] = {
                "message_mode": message_mode,
                "gate_mode": "none",
                "alpha_max": requested_alpha,
            }
            return result
        bypass = gate_mode in {"none", "fail_open"}
        active_engine = make_engine(
            active_graph,
            active_cards,
            gate_validator=_pass_all_gates if bypass else None,
            message_transform=transform,
            active_alpha_max=requested_alpha,
            allow_unbounded=requested_alpha > 0.6,
        )
        result = dict(active_engine.run(active_case))
        result["ablation"] = {
            "message_mode": message_mode,
            "gate_mode": gate_mode,
            "alpha_max": requested_alpha,
        }
        if intervention is not None:
            result["intervention"] = {"spec": dict(intervention), "audits": audits}
        return result

    def compare_intervention(*, base: Any, edited: Any, **__: Any) -> str:
        base_result = getattr(base, "predictions", base)
        edited_result = getattr(edited, "predictions", edited)
        if not isinstance(base_result, Mapping) or not isinstance(edited_result, Mapping):
            raise TypeError("intervention comparison requires mapping predictions")
        delta = _macro_burden(edited_result) - _macro_burden(base_result)
        if delta > 1e-9:
            return "worsen"
        if delta < -1e-9:
            return "improve"
        return "stable"

    return {
        "run_pg": run_pg,
        "run_global": run_global,
        "run_organs": run_organs,
        "aggregate": aggregate,
        "run_compute_matched": run_compute_matched,
        "run_graph": run_graph,
        "mutate_graph": mutate_graph,
        "compare_intervention": compare_intervention,
    }


__all__ = ["PhysioCircuitEngine", "build_adapter"]
