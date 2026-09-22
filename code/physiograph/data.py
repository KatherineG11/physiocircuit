"""Patient-case loading with a hard public/private boundary.

The legacy benchmark stores public observations and private labels in the
same JSON file.  This module splits them immediately.  Model-facing code only
receives :class:`PatientCase`; the scorer receives the separate ``GoldRecord``.
No prompt builder is given the original mapping.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .schema import EvidenceRef, ObservationNode, PatientCase


class DataError(ValueError):
    """Raised for malformed or label-leaking case records."""


PRIVATE_KEYS = frozenset(
    {
        "gold",
        "design_card",
        "answer",
        "expected",
        "expect",
        "future",
        "future_nodes",
        "mechanisms",  # per-case active mechanisms are truth/design metadata
        "probe",
    }
)


@dataclass(frozen=True)
class GoldRecord:
    case_id: str
    steps: tuple[Mapping[str, Any], ...]
    turning_points: tuple[Any, ...]
    design_card: Mapping[str, Any]
    interventions: tuple[Mapping[str, Any], ...]
    probe: Mapping[str, Any]
    source_file: str

    @classmethod
    def from_legacy(cls, case_id: str, payload: Mapping[str, Any], source_file: str) -> "GoldRecord":
        gold = payload.get("gold", {})
        if not isinstance(gold, Mapping):
            gold = {}
        return cls(
            case_id=case_id,
            steps=tuple(copy.deepcopy(list(gold.get("steps", [])))),
            turning_points=tuple(copy.deepcopy(list(gold.get("turning_points", [])))),
            design_card=copy.deepcopy(dict(gold.get("design_card", {}))),
            interventions=tuple(copy.deepcopy(list(gold.get("interventions", [])))),
            probe=copy.deepcopy(dict(gold.get("probe", {}))),
            source_file=source_file,
        )


@dataclass(frozen=True)
class CaseBundle:
    public: PatientCase
    gold: GoldRecord
    source_file: str


def _normalise_domain(payload: Mapping[str, Any]) -> str:
    raw = str(payload.get("system") or payload.get("field") or "unknown").lower()
    if raw in {"cardiovascular_sepsis", "cardiovascular", "sepsis"}:
        return "cardio"
    if raw in {"respiratory", "pulmonary", "pneumonia"}:
        return "pneumonia"
    return raw


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_identifier(value: Any, fallback: str) -> str:
    """Turn a legacy free-text variable name into a schema-safe identifier."""

    raw = str(value).strip().lower()
    cooked = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in raw
    )
    cooked = "_".join(part for part in cooked.split("_") if part)
    return cooked or fallback


_LAB_ROUTES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "digestive": (
        (("bilirubin", "alp", "ggt", "bile"), "gallbladder"),
        (("gastric", "pepsin"), "stomach"),
        (("albumin", "fecal_fat", "absorption"), "intestine"),
    ),
    "pneumonia": (
        (("creatinine", "bun", "egfr", "urine"), "kidney"),
        (("troponin", "bnp", "nt_probnp"), "heart"),
        (("crp", "procalcitonin", "wbc", "cytokine"), "systemic_inflammation"),
        (("d_dimer", "pulmonary_pressure"), "pulmonary_vessels"),
        (("pao2", "pf_ratio", "oxygenation", "spo2"), "alveoli"),
        (("pco2", "peak_flow", "airway"), "airway"),
    ),
    "cardio": (
        (("troponin", "bnp", "nt_probnp", "cardiac_output"), "heart"),
        (("map", "lactate", "svr", "blood_pressure"), "vasculature"),
        (("creatinine", "bun", "egfr", "urine"), "kidney"),
        (("alt", "ast", "bilirubin", "albumin"), "liver"),
        (("inr", "platelet", "d_dimer", "fibrinogen"), "coagulation"),
        (("pao2", "pf_ratio", "spo2", "oxygenation"), "lung"),
    ),
}


_VITAL_ROUTES: dict[str, Mapping[str, str]] = {
    "pneumonia": {"spo2": "alveoli", "rr": "airway", "hr": "heart"},
    "cardio": {"spo2": "lung", "map": "vasculature", "hr": "heart"},
}


def _route_variable(domain: str, variable: str) -> str:
    """Route an observed variable without consulting labels or future states."""

    normalized = _safe_identifier(variable, "unknown")
    for tokens, organ in _LAB_ROUTES.get(domain, ()):
        if any(token in normalized for token in tokens):
            return organ
    return "patient"


def _evidence_for_node(
    payload: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    domain: str,
    index: int,
    include_static_context: bool,
) -> tuple[EvidenceRef, ...]:
    """Convert one legacy observation into traceable, prompt-safe evidence.

    Legacy L0 files do not assign each laboratory value to an organ.  A frozen
    public lookup routes only unambiguous variables (for example creatinine to
    kidney); anything else is labelled ``patient`` and shared.  The router uses
    no labels, future observations, or patient-specific mechanism truth.
    """

    step = int(item.get("t", index))
    evidence: list[EvidenceRef] = []
    if include_static_context:
        patient = payload.get("patient", {})
        presentation = payload.get("presentation", {})
        if patient:
            evidence.append(
                EvidenceRef(
                    evidence_id="patient:profile",
                    organ="patient",
                    variable="profile",
                    step=step,
                    text=_text(patient),
                    predicted=False,
                )
            )
        if presentation:
            presentation_public = dict(presentation) if isinstance(presentation, Mapping) else presentation
            if isinstance(presentation_public, dict):
                presentation_public.pop("vital_signs", None)
            evidence.append(
                EvidenceRef(
                    evidence_id="presentation:summary",
                    organ="patient",
                    variable="presentation",
                    step=step,
                    text=_text(presentation_public),
                    predicted=False,
                )
            )
            if isinstance(presentation, Mapping):
                vitals = presentation.get("vital_signs", {})
                if isinstance(vitals, Mapping):
                    vital_routes = _VITAL_ROUTES.get(domain, {})
                    for name in sorted(vitals):
                        variable = _safe_identifier(name, "vital_sign")
                        evidence.append(
                            EvidenceRef(
                                evidence_id=f"presentation:vital:{variable}",
                                organ=vital_routes.get(variable, "patient"),
                                variable=variable,
                                step=step,
                                text=f"{name}={vitals[name]}",
                                predicted=False,
                            )
                        )
    if item.get("note"):
        evidence.append(
            EvidenceRef(
                evidence_id=f"obs:{step}:note",
                organ="patient",
                variable="clinical_note",
                step=step,
                text=str(item["note"]),
                predicted=False,
            )
        )
    labs = item.get("labs", {})
    if isinstance(labs, Mapping):
        for name in sorted(labs):
            variable = _safe_identifier(name, "laboratory_value")
            evidence.append(
                EvidenceRef(
                    evidence_id=f"obs:{step}:lab:{variable}",
                    organ=_route_variable(domain, variable),
                    variable=variable,
                    step=step,
                    text=f"{name}={labs[name]}",
                    predicted=False,
                )
            )
    treatments = item.get("treatments", [])
    sequence = treatments if isinstance(treatments, list) else [treatments]
    for treatment_index, treatment in enumerate(sequence):
        evidence.append(
            EvidenceRef(
                evidence_id=f"obs:{step}:treatment:{treatment_index}",
                organ="patient",
                variable="treatment",
                step=step,
                text=f"treatment={_text(treatment)}",
                predicted=False,
            )
        )
    return tuple(evidence)


def split_legacy_case(payload: Mapping[str, Any], source_file: str = "<memory>") -> CaseBundle:
    """Split one legacy L0 case into prompt-safe public input and private gold."""

    case_id = str(payload.get("case_id", "")).strip()
    if not case_id:
        raise DataError(f"case_id is required: {source_file}")
    if "observed" in payload and "counterfactual" in payload:
        raise DataError(f"counterfactual pair is not a main-case record: {source_file}")
    observations: list[ObservationNode] = []
    domain = _normalise_domain(payload)
    raw_window = payload.get("observed_window", []) or []
    if not raw_window:
        # PatientCase requires an observed node.  A context-only node preserves
        # the public input without manufacturing a clinical measurement.
        raw_window = [{"t": 0, "note": "No longitudinal observation supplied."}]
    for index, item in enumerate(raw_window):
        if not isinstance(item, Mapping):
            raise DataError(f"observed_window[{index}] must be an object: {source_file}")
        step = int(item.get("t", index))
        observations.append(
            ObservationNode(
                step=step,
                time_label=str(item.get("time_label") or item.get("note") or f"t={step}"),
                evidence=_evidence_for_node(
                    payload,
                    item,
                    domain=domain,
                    index=index,
                    include_static_context=index == 0,
                ),
                elapsed_steps=max(0, int(item.get("elapsed_steps", item.get("elapsed", 1)))),
                predicted=False,
            )
        )
    metadata = {
        "source_file": source_file,
        "split": str(payload.get("split", "unknown")),
        "stratum": str(payload.get("stratum", "unknown")),
    }
    if payload.get("fingerprint") is not None:
        metadata["fingerprint"] = str(payload["fingerprint"])
    public = PatientCase(
        case_id=case_id,
        domain=domain,
        nodes=tuple(observations),
        forecast_horizon=int(payload.get("horizon", 1)),
        metadata=tuple(sorted(metadata.items())),
    )
    gold = GoldRecord.from_legacy(case_id, payload, source_file)
    return CaseBundle(public=public, gold=gold, source_file=source_file)


def load_case_file(path: str | Path) -> CaseBundle:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DataError(f"case file root must be an object: {source}")
    return split_legacy_case(value, str(source))


def iter_case_files(path: str | Path) -> Iterator[Path]:
    source = Path(path)
    if source.is_file():
        yield source
        return
    if not source.exists():
        raise DataError(f"case path does not exist: {source}")
    for item in sorted(source.glob("*.json")):
        if "-CF-" not in item.stem and not item.stem.startswith("PG-CF-"):
            yield item


def load_cases(
    path: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
) -> list[CaseBundle]:
    source = Path(path)
    if source.is_file() and source.suffix.lower() == ".jsonl":
        return load_jsonl_cases(source, split=split, limit=limit)
    result: list[CaseBundle] = []
    for case_file in iter_case_files(path):
        bundle = load_case_file(case_file)
        metadata = dict(bundle.public.metadata)
        if split and metadata.get("split") != split:
            continue
        result.append(bundle)
        if limit is not None and len(result) >= limit:
            break
    return result


def load_jsonl_cases(
    public_path: str | Path,
    *,
    gold_path: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> list[CaseBundle]:
    """Load physically separated ``public.jsonl`` and ``gold.jsonl`` files.

    The two streams are joined by ``case_id`` in memory only.  Public records
    are passed through :func:`split_legacy_case` after the join, so callers still
    receive a ``PatientCase`` that cannot contain private labels.  This is the
    format emitted by :mod:`physiograph.import_mimiciv`.
    """

    public_source = Path(public_path)
    if gold_path is None:
        gold_source = public_source.with_name("gold.jsonl")
    else:
        gold_source = Path(gold_path)
    if not public_source.is_file() or not gold_source.is_file():
        raise DataError(f"public/gold JSONL pair is incomplete: {public_source}, {gold_source}")

    def read_lines(source: Path) -> dict[str, Mapping[str, Any]]:
        values: dict[str, Mapping[str, Any]] = {}
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"invalid JSONL at {source}:{line_number}") from exc
                if not isinstance(value, Mapping) or not value.get("case_id"):
                    raise DataError(f"JSONL row needs case_id at {source}:{line_number}")
                case_id = str(value["case_id"])
                if case_id in values:
                    raise DataError(f"duplicate case_id {case_id!r} in {source}")
                values[case_id] = value
        return values

    public_values = read_lines(public_source)
    gold_values = read_lines(gold_source)
    if set(public_values) != set(gold_values):
        missing_public = sorted(set(gold_values) - set(public_values))[:5]
        missing_gold = sorted(set(public_values) - set(gold_values))[:5]
        raise DataError(f"public/gold case_id mismatch; missing public={missing_public}, missing gold={missing_gold}")
    result: list[CaseBundle] = []
    for case_id in sorted(public_values):
        public = dict(public_values[case_id])
        gold = dict(gold_values[case_id])
        public["gold"] = gold
        bundle = split_legacy_case(public, f"{public_source}::{case_id}")
        metadata = dict(bundle.public.metadata)
        if split and metadata.get("split") != split:
            continue
        result.append(bundle)
        if limit is not None and len(result) >= limit:
            break
    return result


def prompt_safe_payload(case: PatientCase) -> dict[str, Any]:
    """Return the only mapping that may be serialised into an LLM prompt."""

    value = case.to_dict()
    leaked = PRIVATE_KEYS.intersection(_recursive_keys(value))
    if leaked:
        raise DataError(f"public case contains private keys: {sorted(leaked)}")
    return value


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            keys.update(_recursive_keys(nested))
    return keys


def write_public_private_dataset(bundles: Iterable[CaseBundle], public_path: Path, private_path: Path) -> None:
    """Export physically separated JSONL files for reproducible experiments."""

    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with public_path.open("w", encoding="utf-8") as public_file, private_path.open("w", encoding="utf-8") as gold_file:
        for bundle in bundles:
            public_file.write(json.dumps(prompt_safe_payload(bundle.public), ensure_ascii=False) + "\n")
            gold_file.write(
                json.dumps(
                    {
                        "case_id": bundle.gold.case_id,
                        "steps": list(bundle.gold.steps),
                        "turning_points": list(bundle.gold.turning_points),
                        "design_card": dict(bundle.gold.design_card),
                        "interventions": list(bundle.gold.interventions),
                        "probe": dict(bundle.gold.probe),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
