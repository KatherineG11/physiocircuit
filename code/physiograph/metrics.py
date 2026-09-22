"""Metrics and paired statistical tests for PhysioCircuit experiments.

The module deliberately has no dependency on the inference implementation.  Every
function consumes ordinary Python mappings/sequences so that scoring can run in a
separate process which is allowed to read private gold labels.  Empty inputs raise
``ValueError`` instead of silently producing a plausible-looking zero.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb, isfinite, log, sqrt
import random
from statistics import fmean, stdev
from typing import Any, Callable, Iterable, Mapping, Sequence


DIRECTION_LABELS: tuple[str, ...] = ("improve", "stable", "worsen")


def _materialize(values: Iterable[Any], name: str) -> list[Any]:
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _get(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    raise KeyError(f"record is missing all of: {', '.join(keys)}")


def mean_and_sample_std(values: Iterable[float]) -> tuple[float, float]:
    """Return mean and sample SD (``ddof=1``); one value has SD 0."""

    data = [float(v) for v in _materialize(values, "values")]
    return fmean(data), stdev(data) if len(data) > 1 else 0.0


def direction_accuracy(
    records: Iterable[Mapping[str, Any]],
    *,
    gold_key: str = "gold_direction",
    pred_key: str = "pred_direction",
) -> float:
    """Micro accuracy over aligned ``(case, step, organ)`` records.

    A missing/``None`` prediction is retained in the denominator and is incorrect.
    This prevents parser failures from disappearing from reported accuracy.
    """

    rows = _materialize(records, "records")
    correct = 0
    for row in rows:
        gold = _get(row, gold_key, "gold", "target")
        pred = row.get(pred_key, row.get("prediction", row.get("pred")))
        correct += int(pred is not None and pred == gold)
    return correct / len(rows)


def turning_point_counts(
    gold: Iterable[bool], pred: Iterable[bool]
) -> dict[str, int]:
    gold_values, pred_values = list(gold), list(pred)
    if not gold_values:
        raise ValueError("gold must not be empty")
    if len(gold_values) != len(pred_values):
        raise ValueError("gold and pred must have the same length")
    tp = sum(bool(g) and bool(p) for g, p in zip(gold_values, pred_values))
    fp = sum((not bool(g)) and bool(p) for g, p in zip(gold_values, pred_values))
    fn = sum(bool(g) and (not bool(p)) for g, p in zip(gold_values, pred_values))
    tn = len(gold_values) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def turning_point_f1(
    gold_or_records: Iterable[Any],
    pred: Iterable[bool] | None = None,
    *,
    gold_key: str = "gold_turning",
    pred_key: str = "pred_turning",
) -> float:
    """Exact-step binary F1 for turning points.

    The first argument may be a boolean sequence (with ``pred`` supplied) or a
    sequence of aligned mappings.  When there are no positive labels or positive
    predictions, F1 is defined as 1 only when both sets are empty.
    """

    first = _materialize(gold_or_records, "gold_or_records")
    if pred is None:
        if not isinstance(first[0], Mapping):
            raise TypeError("pred is required when the first argument is not records")
        gold_values = [bool(_get(row, gold_key, "gold")) for row in first]
        pred_values = [bool(row.get(pred_key, row.get("prediction", False))) for row in first]
    else:
        gold_values = [bool(v) for v in first]
        pred_values = [bool(v) for v in pred]
    counts = turning_point_counts(gold_values, pred_values)
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    if denominator == 0:
        return 1.0
    return 2 * counts["tp"] / denominator


def full_trajectory_accuracy(
    records: Iterable[Mapping[str, Any]],
    *,
    case_key: str = "case_id",
    gold_key: str = "gold_direction",
    pred_key: str = "pred_direction",
    expected_items_by_case: Mapping[str, int] | None = None,
) -> float:
    """Fraction of cases whose every future step and organ is correct.

    ``expected_items_by_case`` should be supplied when predictions can be missing
    entirely (rather than represented as rows with a null prediction).
    """

    rows = _materialize(records, "records")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(_get(row, case_key))].append(row)
    correct_cases = 0
    for case_id, case_rows in by_case.items():
        complete = expected_items_by_case is None or len(case_rows) == expected_items_by_case.get(
            case_id, len(case_rows)
        )
        all_correct = complete and all(
            row.get(pred_key, row.get("prediction", row.get("pred")))
            == _get(row, gold_key, "gold", "target")
            for row in case_rows
        )
        correct_cases += int(all_correct)
    if expected_items_by_case:
        # Cases with no prediction rows must remain in the denominator.
        return correct_cases / len(expected_items_by_case)
    return correct_cases / len(by_case)


def _probability_vector(
    probabilities: Mapping[str, Any] | Sequence[float], labels: Sequence[str]
) -> list[float]:
    if isinstance(probabilities, Mapping):
        values = [float(probabilities.get(label, 0.0)) for label in labels]
    else:
        values = [float(value) for value in probabilities]
        if len(values) != len(labels):
            raise ValueError("probability vector length does not match labels")
    if any((not isfinite(value)) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    # Small API rounding errors are normalized; large malformed values are not
    # silently accepted as-is.
    return [value / total for value in values]


def multiclass_brier_score(
    probabilities: Iterable[Mapping[str, Any] | Sequence[float]],
    gold: Iterable[str],
    *,
    labels: Sequence[str] = DIRECTION_LABELS,
    normalize_by_classes: bool = False,
) -> float:
    """Mean multiclass Brier score (sum over classes by default)."""

    probs, targets = list(probabilities), list(gold)
    if not targets:
        raise ValueError("gold must not be empty")
    if len(probs) != len(targets):
        raise ValueError("probabilities and gold must have the same length")
    label_list = list(labels)
    if len(set(label_list)) != len(label_list):
        raise ValueError("labels must be unique")
    scores: list[float] = []
    for raw, target in zip(probs, targets):
        if target not in label_list:
            raise ValueError(f"unknown gold label: {target!r}")
        vector = _probability_vector(raw, label_list)
        score = sum(
            (value - float(label == target)) ** 2
            for label, value in zip(label_list, vector)
        )
        if normalize_by_classes:
            score /= len(label_list)
        scores.append(score)
    return fmean(scores)


def negative_log_likelihood(
    probabilities: Iterable[Mapping[str, Any] | Sequence[float]],
    gold: Iterable[str],
    *,
    labels: Sequence[str] = DIRECTION_LABELS,
    epsilon: float = 1e-15,
) -> float:
    """Mean multiclass negative log likelihood."""

    probs, targets = list(probabilities), list(gold)
    if not targets:
        raise ValueError("gold must not be empty")
    if len(probs) != len(targets):
        raise ValueError("probabilities and gold must have the same length")
    label_list = list(labels)
    losses: list[float] = []
    for raw, target in zip(probs, targets):
        if target not in label_list:
            raise ValueError(f"unknown gold label: {target!r}")
        vector = _probability_vector(raw, label_list)
        p = max(epsilon, min(1.0, vector[label_list.index(target)]))
        losses.append(-log(p))
    return fmean(losses)


def expected_calibration_error(
    probabilities: Iterable[Mapping[str, Any] | Sequence[float]],
    gold: Iterable[str],
    *,
    labels: Sequence[str] = DIRECTION_LABELS,
    n_bins: int = 5,
) -> float:
    """Top-label ECE with fixed equal-width confidence bins."""

    probs, targets = list(probabilities), list(gold)
    if not targets:
        raise ValueError("gold must not be empty")
    if len(probs) != len(targets):
        raise ValueError("probabilities and gold must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    label_list = list(labels)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for raw, target in zip(probs, targets):
        vector = _probability_vector(raw, label_list)
        index = max(range(len(vector)), key=vector.__getitem__)
        confidence = vector[index]
        bin_index = min(n_bins - 1, int(confidence * n_bins))
        bins[bin_index].append((confidence, float(label_list[index] == target)))
    total = len(targets)
    return sum(
        (len(bucket) / total)
        * abs(fmean(item[0] for item in bucket) - fmean(item[1] for item in bucket))
        for bucket in bins
        if bucket
    )


def intervention_direction_agreement(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_key: str = "expected_response",
    predicted_key: str = "predicted_response",
) -> float:
    """IDAR over unique intervention instances.

    Duplicate ``intervention_id`` values are rejected so repeated log fragments
    cannot accidentally inflate the denominator.
    """

    rows = _materialize(records, "records")
    seen: set[str] = set()
    correct = 0
    for index, row in enumerate(rows):
        intervention_id = str(row.get("intervention_id", index))
        if intervention_id in seen:
            raise ValueError(f"duplicate intervention_id: {intervention_id}")
        seen.add(intervention_id)
        expected = _get(row, expected_key, "expected", "gold")
        predicted = row.get(predicted_key, row.get("prediction", row.get("pred")))
        correct += int(predicted is not None and predicted == expected)
    return correct / len(rows)


def stability_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    labels: Sequence[str] = DIRECTION_LABELS,
    final_step: int | None = 5,
) -> dict[str, float]:
    """Compute flips, rollout error growth, and probability oscillation.

    Records require ``case_id``, ``step``, ``organ``, a prediction and gold.  A
    probability mapping/vector is optional; oscillation is NaN when unavailable.
    Error growth is ``error(final_step) - error(first observed step)``.  When the
    requested final step is absent, the latest common step is used.
    """

    rows = _materialize(records, "records")
    trajectories: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    cases: set[str] = set()
    by_step: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        case_id = str(_get(row, "case_id"))
        organ = str(_get(row, "organ"))
        step = int(_get(row, "step"))
        pred = row.get("pred_direction", row.get("prediction", row.get("pred")))
        gold = _get(row, "gold_direction", "gold", "target")
        trajectories[(case_id, organ)].append(row)
        cases.add(case_id)
        by_step[step].append(pred == gold)

    flips = 0
    oscillations: list[float] = []
    for trajectory in trajectories.values():
        ordered = sorted(trajectory, key=lambda row: int(_get(row, "step")))
        predictions = [
            row.get("pred_direction", row.get("prediction", row.get("pred")))
            for row in ordered
        ]
        flips += sum(left != right for left, right in zip(predictions, predictions[1:]))
        probability_rows = [row.get("probabilities") for row in ordered]
        for left, right in zip(probability_rows, probability_rows[1:]):
            if left is None or right is None:
                continue
            left_vector = _probability_vector(left, labels)
            right_vector = _probability_vector(right, labels)
            # Total variation distance is normalized to [0, 1].
            oscillations.append(0.5 * sum(abs(a - b) for a, b in zip(left_vector, right_vector)))

    steps = sorted(by_step)
    first = steps[0]
    selected_final = final_step if final_step in by_step else steps[-1]
    first_error = 1.0 - sum(by_step[first]) / len(by_step[first])
    final_error = 1.0 - sum(by_step[selected_final]) / len(by_step[selected_final])
    return {
        "flips_per_case": flips / len(cases),
        "first_step_error": first_error,
        "final_step_error": final_error,
        "error_growth": final_error - first_error,
        "oscillation": fmean(oscillations) if oscillations else float("nan"),
        "first_step": float(first),
        "final_step": float(selected_final),
    }


def cost_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate token, latency, call and monetary cost logs."""

    rows = _materialize(records, "records")
    case_ids = {str(row.get("case_id", "__unknown__")) for row in rows}
    calls = sum(int(row.get("calls", 1)) for row in rows)
    input_tokens = sum(int(row.get("input_tokens", 0)) for row in rows)
    output_tokens = sum(int(row.get("output_tokens", 0)) for row in rows)
    cached_tokens = sum(int(row.get("cached_tokens", 0)) for row in rows)
    latency_ms = sum(float(row.get("latency_ms", 0.0)) for row in rows)
    cost_usd = sum(float(row.get("cost_usd", 0.0)) for row in rows)
    denominator = len(case_ids)
    return {
        "cases": float(denominator),
        "calls": float(calls),
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "total_tokens": float(input_tokens + output_tokens),
        "cached_tokens": float(cached_tokens),
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "calls_per_case": calls / denominator,
        "tokens_per_case": (input_tokens + output_tokens) / denominator,
        "cost_usd_per_case": cost_usd / denominator,
    }


def paired_bootstrap(
    values_a: Iterable[float],
    values_b: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: Callable[[Sequence[float]], float] = fmean,
) -> dict[str, float]:
    """Case-paired bootstrap CI for ``statistic(a) - statistic(b)``."""

    a, b = [float(v) for v in values_a], [float(v) for v in values_b]
    if not a:
        raise ValueError("paired values must not be empty")
    if len(a) != len(b):
        raise ValueError("paired values must have the same length")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(a)
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(
            float(statistic([a[index] for index in indices]))
            - float(statistic([b[index] for index in indices]))
        )
    samples.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(samples) - 1)
        lower = int(position)
        upper = min(len(samples) - 1, lower + 1)
        fraction = position - lower
        return samples[lower] * (1.0 - fraction) + samples[upper] * fraction

    alpha = 1.0 - confidence
    return {
        "estimate": float(statistic(a)) - float(statistic(b)),
        "ci_low": percentile(alpha / 2.0),
        "ci_high": percentile(1.0 - alpha / 2.0),
        "confidence": confidence,
        "n_pairs": float(n),
        "n_resamples": float(n_resamples),
    }


def mcnemar_test(
    correct_a: Iterable[bool], correct_b: Iterable[bool]
) -> dict[str, float]:
    """Paired McNemar test with an exact two-sided binomial p-value."""

    a, b = [bool(v) for v in correct_a], [bool(v) for v in correct_b]
    if not a:
        raise ValueError("paired outcomes must not be empty")
    if len(a) != len(b):
        raise ValueError("paired outcomes must have the same length")
    a_only = sum(left and not right for left, right in zip(a, b))
    b_only = sum((not left) and right for left, right in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
        corrected_chi2 = 0.0
    else:
        tail = sum(comb(discordant, k) for k in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
        corrected_chi2 = (max(0, abs(a_only - b_only) - 1) ** 2) / discordant
    return {
        "a_correct_b_wrong": float(a_only),
        "a_wrong_b_correct": float(b_only),
        "discordant": float(discordant),
        "chi2_continuity_corrected": corrected_chi2,
        "p_exact_two_sided": p_value,
        "n_pairs": float(len(a)),
    }


def score_direction_records(
    records: Iterable[Mapping[str, Any]],
    *,
    labels: Sequence[str] = DIRECTION_LABELS,
    ece_bins: int = 5,
) -> dict[str, float]:
    """Convenience scorer for aligned direction records."""

    rows = _materialize(records, "records")
    probabilities = [row.get("probabilities") for row in rows]
    gold = [str(_get(row, "gold_direction", "gold", "target")) for row in rows]
    result = {
        "direction_accuracy": direction_accuracy(rows),
        "full_trajectory_accuracy": full_trajectory_accuracy(rows),
    }
    if all(probability is not None for probability in probabilities):
        result.update(
            {
                "brier": multiclass_brier_score(probabilities, gold, labels=labels),
                "nll": negative_log_likelihood(probabilities, gold, labels=labels),
                "ece": expected_calibration_error(
                    probabilities, gold, labels=labels, n_bins=ece_bins
                ),
            }
        )
    return result


# Short, conventional aliases used by report code and third-party scripts.
brier_score = multiclass_brier_score
nll = negative_log_likelihood
ece = expected_calibration_error
idar = intervention_direction_agreement
bootstrap_ci = paired_bootstrap
mcnemar = mcnemar_test


__all__ = [
    "DIRECTION_LABELS",
    "bootstrap_ci",
    "brier_score",
    "cost_metrics",
    "direction_accuracy",
    "ece",
    "expected_calibration_error",
    "full_trajectory_accuracy",
    "idar",
    "intervention_direction_agreement",
    "mcnemar",
    "mcnemar_test",
    "mean_and_sample_std",
    "multiclass_brier_score",
    "negative_log_likelihood",
    "nll",
    "paired_bootstrap",
    "score_direction_records",
    "stability_metrics",
    "turning_point_counts",
    "turning_point_f1",
]
