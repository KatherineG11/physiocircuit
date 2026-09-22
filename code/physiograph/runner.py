"""Experiment execution, JSONL audit logs, scoring, and interventions.

The runner never reads private gold during inference.  Scoring is a separate
entry point and refuses legacy L1 manifests.  Engine integration uses duck-typed
hooks from :mod:`physiograph.baselines`, keeping the runner independent of the
concrete dataclass schemas and inference implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .baselines import (
    BaselineOutcome,
    check_availability,
    normalize_method_id,
    run_method,
)
from .legacy import API_RUN_SOURCE, assert_api_run
from .metrics import (
    cost_metrics,
    direction_accuracy,
    expected_calibration_error,
    full_trajectory_accuracy,
    intervention_direction_agreement,
    mean_and_sample_std,
    multiclass_brier_score,
    negative_log_likelihood,
    stability_metrics,
    turning_point_f1,
)


PUBLIC_CASE_FIELDS = frozenset(
    {
        "case_id",
        "split",
        "field",
        "domain",
        "system",
        "stratum",
        "horizon",
        "patient",
        "presentation",
        "observed_window",
        "observation_nodes",
        "observations",
        "treatments",
        "robustness_variant",
        "metadata",
    }
)
PRIVATE_TOP_LEVEL_FIELDS = frozenset(
    {
        "gold",
        "future",
        "future_nodes",
        "target",
        "targets",
        "label",
        "answer",
        "design_card",
        "expected",
        "probe",
        # Patient-specific active mechanisms reveal truth.  The frozen domain
        # card library is loaded separately as a resource.
        "mechanisms",
    }
)


@dataclass(frozen=True)
class MethodRequest:
    method_id: str
    variant: str | None = None

    @classmethod
    def parse(cls, value: str | "MethodRequest") -> "MethodRequest":
        if isinstance(value, MethodRequest):
            return value
        method, separator, variant = str(value).partition(":")
        return cls(normalize_method_id(method), variant if separator else None)

    @property
    def key(self) -> str:
        return f"{self.method_id}:{self.variant}" if self.variant else self.method_id


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(child) for child in value]
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON, JSONL, or directory of per-case JSON files."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(source.glob("*.json")):
            # Metadata files are not cases unless they contain a cases array.
            payload = json.loads(child.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "case_id" in payload:
                rows.append(payload)
            elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
                rows.extend(item for item in payload["cases"] if isinstance(item, dict))
        return rows
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for collection_key in ("cases", "records", "interventions"):
            if isinstance(payload.get(collection_key), list):
                return [item for item in payload[collection_key] if isinstance(item, dict)]
    if isinstance(payload, dict) and "case_id" in payload:
        return [payload]
    raise ValueError(f"cannot find case records in {source}")


def public_case(case: Any) -> Any:
    """Return the prompt-safe representation of a case.

    Schema objects may implement ``to_public_dict``.  Plain mappings are copied
    through a strict top-level allowlist; unknown fields are not sent to a model.
    """

    if hasattr(case, "to_public_dict") and callable(case.to_public_dict):
        return case.to_public_dict()
    if hasattr(case, "public_view") and callable(case.public_view):
        return case.public_view()
    if not isinstance(case, Mapping):
        return case
    return {key: _jsonable(case[key]) for key in PUBLIC_CASE_FIELDS if key in case}


def validate_public_cases(cases: Iterable[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id", ""))
        if not case_id:
            errors.append(f"case[{index}] has no case_id")
        elif case_id in seen:
            errors.append(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        leaked = sorted(PRIVATE_TOP_LEVEL_FIELDS.intersection(case))
        if leaked:
            errors.append(f"case {case_id or index} contains private fields: {', '.join(leaked)}")
        if int(case.get("horizon", 0)) <= 0:
            errors.append(f"case {case_id or index} has invalid horizon")
    return errors


class RunLogger:
    """Append-only stream logger for one run directory."""

    STREAMS = {
        "status": "status.jsonl",
        "predictions": "predictions.jsonl",
        "messages": "messages.jsonl",
        "gates": "gates.jsonl",
        "arbitration": "arbitration.jsonl",
        "macro": "macro.jsonl",
        "usage": "usage.jsonl",
        "raw_responses": "raw_responses.jsonl",
        "failures": "failures.jsonl",
        "interventions": "intervention_predictions.jsonl",
    }

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def append(self, stream: str, value: Any) -> None:
        if stream not in self.STREAMS:
            raise KeyError(f"unknown log stream: {stream}")
        append_jsonl(self.run_dir / self.STREAMS[stream], value)

    def read(self, stream: str) -> list[dict[str, Any]]:
        return read_jsonl(self.run_dir / self.STREAMS[stream])


def _config_snapshot(config: Any) -> dict[str, Any]:
    if hasattr(config, "redacted_snapshot") and callable(config.redacted_snapshot):
        return _jsonable(config.redacted_snapshot())
    if isinstance(config, Mapping):
        snapshot = _jsonable(config)
        # Defensive redaction for ad-hoc mappings.
        for model in snapshot.get("models", {}).values():
            if isinstance(model, dict):
                for key in list(model):
                    if "key" in key.lower() or "secret" in key.lower():
                        model[key] = "<redacted>"
        return snapshot
    return {"type": type(config).__name__}


def _case_id(case: Any) -> str:
    if isinstance(case, Mapping):
        return str(case.get("case_id", ""))
    return str(getattr(case, "case_id", ""))


def _stream_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _outcome_streams(outcome: BaselineOutcome) -> tuple[Any, dict[str, Any]]:
    predictions = outcome.predictions
    artifacts = dict(outcome.artifacts)
    if isinstance(predictions, Mapping):
        known = {
            "predictions",
            "messages",
            "gates",
            "arbitration",
            "macro",
            "usage",
            "raw_responses",
        }
        if any(key in predictions for key in known):
            payload = predictions.get("predictions")
            for key in known - {"predictions"}:
                if key in predictions and key not in artifacts:
                    artifacts[key] = predictions[key]
            predictions = payload
        elif isinstance(predictions.get("steps"), list):
            # Native PhysioCircuitEngine audit output.  Preserve absolute engine
            # time in ``actual_step`` while scoring against the benchmark's
            # relative future index 1..H.
            flat_predictions: list[dict[str, Any]] = []
            messages: list[dict[str, Any]] = []
            gates: list[dict[str, Any]] = []
            arbitration: list[dict[str, Any]] = []
            macro: list[dict[str, Any]] = []
            raw_responses: list[dict[str, Any]] = []
            usage: list[dict[str, Any]] = []
            for forecast_index, step_record in enumerate(predictions["steps"], 1):
                actual_step = step_record.get("step")
                estimates = step_record.get("final_estimates", {})
                if isinstance(estimates, Mapping):
                    for organ, estimate in estimates.items():
                        if not isinstance(estimate, Mapping):
                            continue
                        raw_probabilities = estimate.get("probabilities", estimate)
                        if not isinstance(raw_probabilities, Mapping):
                            raw_probabilities = estimate
                        probabilities = {
                            "improve": float(raw_probabilities.get("improve", estimate.get("p_improve", 0.0))),
                            "stable": float(raw_probabilities.get("stable", estimate.get("p_stable", 0.0))),
                            "worsen": float(raw_probabilities.get("worsen", estimate.get("p_worsen", 0.0))),
                        }
                        direction = max(
                            ("improve", "stable", "worsen"),
                            key=lambda label: (probabilities[label], -("improve", "stable", "worsen").index(label)),
                        )
                        flat_predictions.append(
                            {
                                "step": forecast_index,
                                "actual_step": actual_step,
                                "organ": str(organ),
                                "pred_direction": direction,
                                "probabilities": probabilities,
                            }
                        )
                messages.extend(
                    {"step": forecast_index, "actual_step": actual_step, **row}
                    for row in step_record.get("delivered_messages", [])
                    if isinstance(row, Mapping)
                )
                gates.extend(
                    {"step": forecast_index, "actual_step": actual_step, **row}
                    for row in step_record.get("gate_audits", [])
                    if isinstance(row, Mapping)
                )
                arbitration.extend(
                    {"step": forecast_index, "actual_step": actual_step, **row}
                    for row in step_record.get("arbitration_verdicts", [])
                    if isinstance(row, Mapping)
                )
                aggregate = step_record.get("aggregate")
                if isinstance(aggregate, Mapping):
                    macro.append(
                        {
                            "step": forecast_index,
                            "actual_step": actual_step,
                            **aggregate,
                        }
                    )
                for audit in step_record.get("llm_calls", []):
                    if not isinstance(audit, Mapping):
                        continue
                    raw_responses.append(
                        {"step": forecast_index, "actual_step": actual_step, **audit}
                    )
                    metadata = audit.get("metadata", {})
                    if isinstance(metadata, Mapping):
                        provider_usage = metadata.get("usage", {})
                        provider_usage = provider_usage if isinstance(provider_usage, Mapping) else {}
                        usage.append(
                            {
                                "step": forecast_index,
                                "actual_step": actual_step,
                                "role": audit.get("role"),
                                "organ": audit.get("organ"),
                                "calls": 1,
                                "input_tokens": provider_usage.get(
                                    "prompt_tokens", provider_usage.get("input_tokens", 0)
                                ),
                                "output_tokens": provider_usage.get(
                                    "completion_tokens", provider_usage.get("output_tokens", 0)
                                ),
                                "cached_tokens": provider_usage.get("cached_tokens", 0),
                                "latency_ms": metadata.get("latency_ms", 0.0),
                                "cost_usd": metadata.get("cost_usd", 0.0),
                                "resolved_model": metadata.get("resolved_model"),
                                "request_id": metadata.get("request_id"),
                            }
                        )
            predictions = flat_predictions
            artifacts.setdefault("messages", messages)
            artifacts.setdefault("gates", gates)
            artifacts.setdefault("arbitration", arbitration)
            artifacts.setdefault("macro", macro)
            artifacts.setdefault("raw_responses", raw_responses)
            artifacts.setdefault("usage", usage)
    return predictions, artifacts


def _log_outcome(
    logger: RunLogger,
    outcome: BaselineOutcome,
    common: Mapping[str, Any],
) -> None:
    status = {**common, "status": outcome.status, "reason": outcome.reason, "missing": outcome.missing}
    logger.append("status", status)
    if outcome.status != "ok":
        if outcome.status == "error":
            logger.append("failures", status)
        return
    predictions, artifacts = _outcome_streams(outcome)
    for row in _stream_rows(predictions):
        logger.append("predictions", {**common, **(_jsonable(row) if isinstance(_jsonable(row), dict) else {"prediction": _jsonable(row)})})
    mapping = {
        "messages": "messages",
        "gates": "gates",
        "arbitration": "arbitration",
        "macro": "macro",
        "usage": "usage",
        "raw_responses": "raw_responses",
    }
    for artifact_name, stream in mapping.items():
        for row in _stream_rows(artifacts.get(artifact_name)):
            encoded = _jsonable(row)
            logger.append(stream, {**common, **(encoded if isinstance(encoded, dict) else {"value": encoded})})


def run_experiment(
    *,
    config: Any,
    cases: Iterable[Any],
    adapter: Any,
    resources: Mapping[str, Any],
    run_dir: str | Path,
    domain: str,
    methods: Iterable[str | MethodRequest],
    model_name: str,
    run_indices: Iterable[int] = (0,),
    case_limit: int | None = None,
    resume: bool = False,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute cases and write an append-only, provenance-marked API run."""

    selected_cases = list(cases)
    if case_limit is not None:
        selected_cases = selected_cases[: max(0, int(case_limit))]
    if not selected_cases:
        raise ValueError("no cases selected")
    requests = [MethodRequest.parse(value) for value in methods]
    if not requests:
        raise ValueError("no methods selected")
    runs = [int(value) for value in run_indices]
    if not runs:
        raise ValueError("no run indices selected")

    target = Path(run_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    snapshot = _config_snapshot(config)
    config_hash = sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"run already exists: {target}; use resume=True")
        if existing.get("source_kind") != API_RUN_SOURCE:
            raise ValueError("cannot resume a non-API run directory")
        if existing.get("config_sha256") != config_hash:
            raise ValueError("resume configuration does not match existing manifest")
        manifest = existing
    else:
        manifest = {
            "schema_version": 1,
            "source_kind": API_RUN_SOURCE,
            "status": "running",
            "created_at": utc_now(),
            "domain": domain,
            "model_name": model_name,
            "methods": [asdict(request) for request in requests],
            "run_indices": runs,
            "case_count": len(selected_cases),
            "config": snapshot,
            "config_sha256": config_hash,
            "python": sys.version,
            "platform": platform.platform(),
        }
        write_json(manifest_path, manifest)

    logger = RunLogger(target)
    completed = {
        (
            str(row.get("case_id")),
            str(row.get("method_id")),
            row.get("variant"),
            int(row.get("run_index", 0)),
        )
        for row in logger.read("status")
        if row.get("status") in {"ok", "unavailable", "invalid", "skipped"}
    }
    status_counts: dict[str, int] = {}
    for request in requests:
        availability = check_availability(
            request.method_id, adapter, resources, variant=request.variant
        )
        if availability.status != "ready":
            manifest.setdefault("method_availability", {})[request.key] = availability.to_dict()

    try:
        for run_index in runs:
            for case in selected_cases:
                case_id = _case_id(case)
                if not case_id:
                    raise ValueError("every case must expose case_id")
                safe_case = public_case(case)
                for request in requests:
                    key = (case_id, request.method_id, request.variant, run_index)
                    if key in completed:
                        continue
                    common = {
                        "timestamp": utc_now(),
                        "case_id": case_id,
                        "domain": domain,
                        "method_id": request.method_id,
                        "variant": request.variant,
                        "model_name": model_name,
                        "run_index": run_index,
                    }
                    outcome = run_method(
                        request.method_id,
                        safe_case,
                        adapter,
                        resources=resources,
                        variant=request.variant,
                        run_seed=run_index,
                        options=options,
                    )
                    _log_outcome(logger, outcome, common)
                    status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["status_counts"] = status_counts
        write_json(manifest_path, manifest)
        raise
    manifest["finished_at"] = utc_now()
    manifest["status_counts"] = status_counts
    write_json(manifest_path, manifest)
    return manifest


def _flatten_gold(cases_or_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in cases_or_rows:
        if {"case_id", "step", "organ"}.issubset(item):
            direction = item.get("gold_direction", item.get("direction", item.get("gold")))
            output.append({**item, "gold_direction": direction})
            continue
        case_id = str(item.get("case_id", ""))
        domain = item.get("domain", item.get("field", item.get("system")))
        stratum = item.get("stratum")
        gold = item.get("gold", item if "steps" in item else {})
        if not isinstance(gold, Mapping):
            continue
        turning_points = {int(value) for value in gold.get("turning_points", []) if str(value).isdigit()}
        for step_record in gold.get("steps", []):
            step = int(step_record.get("t", step_record.get("step")))
            directions = step_record.get("direction", {})
            if not isinstance(directions, Mapping):
                continue
            for organ, direction in directions.items():
                output.append(
                    {
                        "case_id": case_id,
                        "domain": domain,
                        "stratum": stratum,
                        "step": step,
                        "organ": str(organ),
                        "gold_direction": direction,
                        "gold_turning": step in turning_points,
                        "gold_phase": step_record.get("phase"),
                    }
                )
    if not output:
        raise ValueError("gold source contains no scoreable direction records")
    return output


def _flatten_prediction_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "step" in row and "organ" in row:
        prediction = row.get("pred_direction", row.get("direction", row.get("prediction")))
        if isinstance(prediction, Enum):
            prediction = prediction.value
        return [{**row, "pred_direction": prediction}]
    payload = row.get("prediction")
    if isinstance(payload, Mapping):
        merged = {**row, **payload}
        merged.pop("prediction", None)
        return _flatten_prediction_row(merged)
    steps = row.get("steps")
    if isinstance(steps, list):
        output: list[dict[str, Any]] = []
        for forecast_index, step_record in enumerate(steps, 1):
            actual_step = step_record.get("t", step_record.get("step"))
            step = step_record.get("forecast_step", forecast_index)
            directions = step_record.get("direction", step_record.get("directions", {}))
            probabilities = step_record.get("probabilities", {})
            if isinstance(directions, Mapping):
                for organ, direction in directions.items():
                    output.append(
                        {
                            **{key: row.get(key) for key in ("case_id", "domain", "method_id", "variant", "model_name", "run_index")},
                            "step": step,
                            "actual_step": actual_step,
                            "organ": organ,
                            "pred_direction": direction,
                            "probabilities": probabilities.get(organ) if isinstance(probabilities, Mapping) else None,
                            "pred_turning": step_record.get("turning_point"),
                        }
                    )
            estimates = step_record.get("final_estimates", {})
            if isinstance(estimates, Mapping):
                for organ, estimate in estimates.items():
                    if not isinstance(estimate, Mapping):
                        continue
                    raw_probabilities = estimate.get("probabilities", estimate)
                    if not isinstance(raw_probabilities, Mapping):
                        raw_probabilities = estimate
                    organ_probabilities = {
                        "improve": float(raw_probabilities.get("improve", estimate.get("p_improve", 0.0))),
                        "stable": float(raw_probabilities.get("stable", estimate.get("p_stable", 0.0))),
                        "worsen": float(raw_probabilities.get("worsen", estimate.get("p_worsen", 0.0))),
                    }
                    direction = max(
                        ("improve", "stable", "worsen"),
                        key=lambda label: organ_probabilities[label],
                    )
                    output.append(
                        {
                            **{key: row.get(key) for key in ("case_id", "domain", "method_id", "variant", "model_name", "run_index")},
                            "step": step,
                            "actual_step": actual_step,
                            "organ": organ,
                            "pred_direction": direction,
                            "probabilities": organ_probabilities,
                        }
                    )
        return output
    return []


def flatten_predictions(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.extend(_flatten_prediction_row(row))
    return output


def _group_key(row: Mapping[str, Any]) -> tuple[str, str | None, int, str]:
    return (
        str(row.get("method_id")),
        row.get("variant"),
        int(row.get("run_index", 0)),
        str(row.get("domain", "")),
    )


def score_api_run(
    run_dir: str | Path,
    *,
    gold_path: str | Path,
    intervention_gold_path: str | Path | None = None,
    output_path: str | Path | None = None,
    ece_bins: int = 5,
) -> dict[str, Any]:
    """Join API predictions with private gold and compute run-level metrics."""

    target = Path(run_dir).expanduser().resolve()
    manifest = assert_api_run(target)
    predictions = flatten_predictions(read_jsonl(target / "predictions.jsonl"))
    gold_rows = _flatten_gold(load_records(gold_path))
    groups: dict[tuple[str, str | None, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[_group_key(row)].append(row)

    run_scores: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        method, variant, run_index, domain = group
        predicted_case_ids = {str(row.get("case_id", "")) for row in rows}
        applicable_gold = [
            row
            for row in gold_rows
            if str(row.get("case_id", "")) in predicted_case_ids
            and (not domain or not row.get("domain") or str(row.get("domain")) == domain)
        ]
        gold_map = {
            (str(row["case_id"]), int(row["step"]), str(row["organ"])): row
            for row in applicable_gold
        }
        pred_map = {
            (str(row["case_id"]), int(row["step"]), str(row["organ"])): row
            for row in rows
        }
        aligned: list[dict[str, Any]] = []
        expected_items: dict[str, int] = {}
        for key, gold in gold_map.items():
            expected_items[key[0]] = expected_items.get(key[0], 0) + 1
            pred = pred_map.get(key, {})
            aligned.append(
                {
                    **gold,
                    **pred,
                    "pred_direction": pred.get("pred_direction"),
                    "gold_direction": gold["gold_direction"],
                }
            )
        if not aligned:
            run_scores.append(
                {
                    "method_id": method,
                    "variant": variant,
                    "run_index": run_index,
                    "domain": domain,
                    "status": "unavailable",
                    "reason": "no prediction keys matched private gold",
                }
            )
            continue
        metrics: dict[str, Any] = {
            "direction_accuracy": direction_accuracy(aligned),
            "full_trajectory_accuracy": full_trajectory_accuracy(
                aligned, expected_items_by_case=expected_items
            ),
            "n_items": len(aligned),
            "n_missing": sum(row.get("pred_direction") is None for row in aligned),
        }
        probability_rows = [row for row in aligned if row.get("probabilities") is not None]
        if len(probability_rows) == len(aligned):
            probabilities = [row["probabilities"] for row in aligned]
            gold = [str(row["gold_direction"]) for row in aligned]
            metrics.update(
                {
                    "brier": multiclass_brier_score(probabilities, gold),
                    "nll": negative_log_likelihood(probabilities, gold),
                    "ece": expected_calibration_error(probabilities, gold, n_bins=ece_bins),
                }
            )
        turning_rows = [row for row in aligned if row.get("pred_turning") is not None]
        if turning_rows:
            metrics["turning_f1"] = turning_point_f1(turning_rows)
        metrics["stability"] = stability_metrics(aligned)
        run_scores.append(
            {
                "method_id": method,
                "variant": variant,
                "run_index": run_index,
                "domain": domain,
                "status": "ok",
                "metrics": metrics,
            }
        )

    usage_rows = read_jsonl(target / "usage.jsonl")
    for score in run_scores:
        matching = [row for row in usage_rows if _group_key(row) == (
            score["method_id"], score.get("variant"), score["run_index"], score["domain"]
        )]
        if matching:
            score.setdefault("metrics", {})["cost"] = cost_metrics(matching)

    intervention_result: dict[str, Any] | None = None
    if intervention_gold_path is not None:
        intervention_result = score_interventions(target, intervention_gold_path)

    summary: dict[str, Any] = {}
    scalar_names = {
        "direction_accuracy",
        "full_trajectory_accuracy",
        "turning_f1",
        "brier",
        "nll",
        "ece",
    }
    summary_groups: dict[tuple[str, str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for score in run_scores:
        if score["status"] == "ok":
            summary_groups[(score["method_id"], score.get("variant"), score["domain"])].append(score)
    for key, values in summary_groups.items():
        name = ":".join(str(value) for value in key if value not in {None, ""})
        summary[name] = {}
        for metric in scalar_names:
            samples = [value["metrics"][metric] for value in values if metric in value["metrics"]]
            if samples:
                mean, sample_sd = mean_and_sample_std(samples)
                summary[name][metric] = {"mean": mean, "sample_sd": sample_sd, "runs": len(samples)}

    result = {
        "schema_version": 1,
        "source_kind": "api_score",
        "scored_run": str(target),
        "run_manifest_sha256": sha256_file(target / "manifest.json"),
        "created_at": utc_now(),
        "run_scores": run_scores,
        "summary": summary,
        "interventions": intervention_result,
        "manifest_status": manifest.get("status"),
    }
    destination = Path(output_path) if output_path else target / "scores.json"
    write_json(destination, result)
    return result


def _without_expected(value: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_fragments = ("expected", "gold", "answer", "label", "rater")
    return {
        key: _jsonable(child)
        for key, child in value.items()
        if not any(fragment in key.lower() for fragment in forbidden_fragments)
    }


def _compare_intervention(adapter: Any, base: BaselineOutcome, edited: BaselineOutcome, mutation: Any) -> Any:
    hook = adapter.get("compare_intervention") if isinstance(adapter, Mapping) else getattr(adapter, "compare_intervention", None)
    if callable(hook):
        return hook(base=base, edited=edited, intervention=mutation)
    for candidate in (edited.artifacts, edited.predictions if isinstance(edited.predictions, Mapping) else {}):
        if isinstance(candidate, Mapping) and "predicted_response" in candidate:
            return candidate["predicted_response"]
    return None


def run_interventions(
    *,
    config: Any,
    cases: Iterable[Any],
    interventions: Iterable[Mapping[str, Any]],
    adapter: Any,
    resources: Mapping[str, Any],
    run_dir: str | Path,
    domain: str,
    methods: Iterable[str | MethodRequest],
    model_name: str,
    run_indices: Iterable[int] = (0,),
) -> dict[str, Any]:
    """Run base/edited pairs without exposing expected responses to inference."""

    case_map = {_case_id(case): case for case in cases}
    requests = [MethodRequest.parse(value) for value in methods]
    logger = RunLogger(run_dir)
    manifest_path = Path(run_dir).resolve() / "manifest.json"
    snapshot = _config_snapshot(config)
    manifest = {
        "schema_version": 1,
        "source_kind": API_RUN_SOURCE,
        "experiment": "intervention",
        "status": "running",
        "created_at": utc_now(),
        "domain": domain,
        "model_name": model_name,
        "methods": [asdict(request) for request in requests],
        "config": snapshot,
        "config_sha256": sha256(canonical_json(snapshot).encode("utf-8")).hexdigest(),
    }
    write_json(manifest_path, manifest)
    count = 0
    for run_index in run_indices:
        for intervention in interventions:
            intervention_id = str(intervention.get("intervention_id", ""))
            case_id = str(intervention.get("case_id", ""))
            if not intervention_id or case_id not in case_map:
                logger.append(
                    "failures",
                    {
                        "intervention_id": intervention_id,
                        "case_id": case_id,
                        "status": "invalid",
                        "reason": "missing intervention_id or unknown case_id",
                    },
                )
                continue
            mutation = _without_expected(intervention)
            case = public_case(case_map[case_id])
            for request in requests:
                base = run_method(
                    request.method_id,
                    case,
                    adapter,
                    resources=resources,
                    variant=request.variant,
                    run_seed=int(run_index),
                )
                edited = run_method(
                    request.method_id,
                    case,
                    adapter,
                    resources=resources,
                    variant=request.variant,
                    run_seed=int(run_index),
                    intervention=mutation,
                )
                predicted_response = (
                    _compare_intervention(adapter, base, edited, mutation)
                    if base.status == edited.status == "ok"
                    else None
                )
                logger.append(
                    "interventions",
                    {
                        "timestamp": utc_now(),
                        "intervention_id": intervention_id,
                        "case_id": case_id,
                        "domain": domain,
                        "method_id": request.method_id,
                        "variant": request.variant,
                        "model_name": model_name,
                        "run_index": int(run_index),
                        "mutation": mutation,
                        "base_status": base.status,
                        "edited_status": edited.status,
                        "predicted_response": predicted_response,
                        "status": "ok" if predicted_response is not None else "unavailable",
                        "reason": None if predicted_response is not None else "engine did not expose a deterministic intervention comparator",
                    },
                )
                count += 1
    manifest.update({"status": "completed", "finished_at": utc_now(), "records": count})
    write_json(manifest_path, manifest)
    return manifest


def score_interventions(run_dir: str | Path, gold_path: str | Path) -> dict[str, Any]:
    target = Path(run_dir).expanduser().resolve()
    assert_api_run(target)
    predictions = read_jsonl(target / "intervention_predictions.jsonl")
    gold_rows = load_records(gold_path)
    gold_map = {
        str(row.get("intervention_id")): row.get(
            "expected_response", row.get("expected_macro_delta", row.get("expected"))
        )
        for row in gold_rows
    }
    groups: dict[tuple[str, str | None, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        expected = gold_map.get(str(row.get("intervention_id")))
        if expected is None:
            continue
        groups[_group_key(row)].append({**row, "expected_response": expected})
    results: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        method, variant, run_index, domain = group
        scoreable = [row for row in rows if row.get("predicted_response") is not None]
        results.append(
            {
                "method_id": method,
                "variant": variant,
                "run_index": run_index,
                "domain": domain,
                "status": "ok" if scoreable else "unavailable",
                "idar": intervention_direction_agreement(scoreable) if scoreable else None,
                "n": len(scoreable),
                "n_unavailable": len(rows) - len(scoreable),
            }
        )
    return {"source_kind": "api_intervention_score", "groups": results}


__all__ = [
    "MethodRequest",
    "PRIVATE_TOP_LEVEL_FIELDS",
    "PUBLIC_CASE_FIELDS",
    "RunLogger",
    "append_jsonl",
    "flatten_predictions",
    "load_records",
    "public_case",
    "read_jsonl",
    "run_experiment",
    "run_interventions",
    "score_api_run",
    "score_interventions",
    "sha256_file",
    "validate_public_cases",
    "write_json",
]
