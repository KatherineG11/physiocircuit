"""Reproducible data assets for the PhysioCircuit paper.

This module is deliberately separate from inference.  It turns the checked-in
legacy case files into prompt-safe/public JSONL and private gold JSONL, builds
the treatment-event and intervention manifests used by the paper, and writes a
machine-readable audit report.  It never fabricates MIMIC-IV records: real ICU
data must be supplied by the user after the relevant data-use approval.

Usage (from ``code/``)::

    python -m physiograph.paper_data audit
    python -m physiograph.paper_data export --output runs/paper_data
    python -m physiograph.paper_data all --output runs/paper_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG, load_cards, load_graph, load_paper_config
from .data import load_cases, prompt_safe_payload


DOMAIN_PREFIX = {"digestive": "PG-D", "pneumonia": "PG-P", "cardio": "PG-S"}
EXPECTED_CASES = 120
EXPECTED_TREATMENTS = 480
EXPECTED_CF = {"digestive": 30, "pneumonia": 24, "cardio": 24}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")


def audit(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Audit the checked-in paper assets without calling an LLM."""

    config = load_paper_config(config_path)
    report: dict[str, Any] = {
        "schema_version": "physiocircuit.paper-data.v1",
        "config": str(config.source_path),
        "domains": {},
        "status": "ok",
    }
    for domain in ("digestive", "pneumonia", "cardio"):
        cards = load_cards(config.cards_path(domain))
        graph = load_graph(config.graph_path(domain), cards)
        bundles = load_cases(config.case_path(domain))
        blind = [b for b in bundles if dict(b.public.metadata).get("split") == "blind_test"]
        cf_files = sorted(config.case_path(domain).glob("*-CF-*.json"))
        errors: list[str] = []
        if len(bundles) != EXPECTED_CASES:
            errors.append(f"expected {EXPECTED_CASES} main cases, found {len(bundles)}")
        if len(cf_files) != EXPECTED_CF[domain]:
            errors.append(f"expected {EXPECTED_CF[domain]} counterfactual files, found {len(cf_files)}")
        if len(blind) != 100:
            errors.append(f"expected 100 blind-test cases, found {len(blind)}")
        report["domains"][domain] = {
            "cases": len(bundles),
            "blind_test_cases": len(blind),
            "counterfactual_pairs": len(cf_files),
            "organs": len(graph.organs),
            "cards": len(cards),
            "graph_edges": len(graph.edges),
            "case_path": str(config.case_path(domain)),
            "errors": errors,
        }
        if errors:
            report["status"] = "error"
    return report


def _treatment_rows(domain: str, bundles: list[Any], cards: list[Any]) -> list[dict[str, Any]]:
    """Create four transparent synthetic treatment events per case."""

    bindings: list[tuple[Any, Any]] = []
    for card in cards:
        for binding in card.treatment_bindings:
            bindings.append((card, binding))
    if not bindings:
        raise ValueError(f"no treatment bindings for {domain}")
    rows: list[dict[str, Any]] = []
    for case_index, bundle in enumerate(bundles):
        for event_index in range(4):
            card, binding = bindings[(case_index * 4 + event_index) % len(bindings)]
            rows.append({
                "event_id": f"{bundle.public.case_id}-T{event_index + 1:02d}",
                "case_id": bundle.public.case_id,
                "domain": domain,
                "step": min(bundle.public.forecast_horizon, event_index + 1),
                "treatment": binding.treatment,
                "action": binding.action,
                "mechanism_id": card.mechanism_id,
                "affected_organ": card.target,
                "expected_direction": card.allowed_directions[0].value,
                "expected_lag_window": list(card.lag_steps),
                "source": "synthetic_truth_generator",
            })
    return rows


def _intervention_rows(domain: str, bundles: list[Any], cards: list[Any]) -> list[dict[str, Any]]:
    """Normalize six semantic intervention operations into a scoreable manifest."""

    by_id = {card.mechanism_id: card for card in cards}
    rows: list[dict[str, Any]] = []
    operations = ("delete", "replace", "reverse", "extend_lag", "inject_wrong_message", "mask_local_evidence")
    for bundle in bundles:
        active = bundle.gold.design_card.get("mechanism_id") if isinstance(bundle.gold.design_card, Mapping) else None
        if not active:
            active = next(iter(by_id), None)
        card = by_id.get(str(active))
        if card is None:
            continue
        for operation in operations:
            rows.append({
                "intervention_id": f"{bundle.public.case_id}:{operation}:{card.mechanism_id}",
                "case_id": bundle.public.case_id,
                "domain": domain,
                "operation": operation,
                "mechanism_id": card.mechanism_id,
                "source": card.source,
                "target": card.target,
                "expected_direction": card.allowed_directions[0].value,
                "expected_lag_window": list(card.lag_steps),
                "score_channel": "edge",
                "gold_source": "truth_team_executable_generator",
            })
    return rows


def export(config_path: str | Path = DEFAULT_CONFIG, output: str | Path = "runs/paper_data") -> dict[str, Any]:
    """Export public/private JSONL and normalized experiment manifests."""

    config = load_paper_config(config_path)
    out = Path(output).resolve()
    manifest: dict[str, Any] = {"schema_version": "physiocircuit.paper-data.v1", "domains": {}}
    for domain in ("digestive", "pneumonia", "cardio"):
        bundles = load_cases(config.case_path(domain))
        cards = list(load_cards(config.cards_path(domain)))
        domain_out = out / domain
        public_rows = [prompt_safe_payload(bundle.public) for bundle in bundles]
        gold_rows = [
            {
                "case_id": bundle.gold.case_id,
                "steps": list(bundle.gold.steps),
                "turning_points": list(bundle.gold.turning_points),
                "design_card": dict(bundle.gold.design_card),
                "interventions": list(bundle.gold.interventions),
                "probe": dict(bundle.gold.probe),
            }
            for bundle in bundles
        ]
        treatment_rows = _treatment_rows(domain, bundles, cards)
        intervention_rows = _intervention_rows(domain, bundles, cards)
        _write_jsonl(domain_out / "public.jsonl", public_rows)
        _write_jsonl(domain_out / "gold.jsonl", gold_rows)
        _write_jsonl(domain_out / "treatments.jsonl", treatment_rows)
        _write_jsonl(domain_out / "interventions.jsonl", intervention_rows)
        manifest["domains"][domain] = {
            "cases": len(bundles),
            "treatment_events": len(treatment_rows),
            "interventions": len(intervention_rows),
            "public_sha256": _sha(public_rows),
            "gold_sha256": _sha(gold_rows),
            "treatment_sha256": _sha(treatment_rows),
            "intervention_sha256": _sha(intervention_rows),
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/export PhysioCircuit paper data")
    parser.add_argument("command", choices=("audit", "export", "all"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="runs/paper_data")
    args = parser.parse_args()
    report = audit(args.config)
    if args.command in {"export", "all"} and report["status"] == "ok":
        report["export"] = export(args.config, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
