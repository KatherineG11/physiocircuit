"""Shared: debugging and logging utilities.

Generates per-run artifacts:
  - output/log/{timestamp}/ directory
  - run.log: streaming terminal log with timestamps
  - case_{name}.json: full per-case debug record (LLM prompt / response / gate details / probability trajectory)
  - summary.json: final summary

All files are UTF-8 encoded.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


class RunLogger:
    """Logger for a run batch (multiple cases)."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log = self.log_dir / "run.log"
        self.cases_dir = self.log_dir / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self._case_records = {}
        self._open()

    def _open(self):
        self._fh = open(self.run_log, "a", encoding="utf-8")

    def _write(self, line):
        ts = datetime.now().strftime("%H:%M:%S")
        self._fh.write(f"[{ts}] {line}\n")
        self._fh.flush()

    def info(self, msg):
        self._write(msg)
        print(msg, flush=True)

    def debug(self, msg):
        self._write(msg)

    def new_case(self, case_name):
        """Start recording a new case; returns the case record dict."""
        rec = {
            "case_name": case_name,
            "steps": [],
            "llm_calls": [],
            "messages": [],
            "gates": [],
            "arbitrations": [],
            "prob_trajectory": [],
            "final": {},
        }
        self._case_records[case_name] = rec
        return rec

    def record_llm_call(self, case_name, round_no, organ, role, sys_prompt, user_prompt, response, usage):
        """Record one LLM call."""
        rec = self._case_records.setdefault(
            case_name,
            {"case_name": case_name, "steps": [], "llm_calls": [], "messages": [],
             "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}},
        )
        rec["llm_calls"].append({
            "round": round_no,
            "organ": organ,
            "role": role,
            "system_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "usage": usage,
        })

    def record_message(self, case_name, msg):
        self._case_records.setdefault(case_name, {"steps": [], "llm_calls": [], "messages": [], "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}})
        self._case_records[case_name]["messages"].append(msg)

    def record_gate(self, case_name, gate):
        self._case_records.setdefault(case_name, {"steps": [], "llm_calls": [], "messages": [], "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}})
        self._case_records[case_name]["gates"].append(gate)

    def record_arbitration(self, case_name, arb):
        self._case_records.setdefault(case_name, {"steps": [], "llm_calls": [], "messages": [], "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}})
        self._case_records[case_name]["arbitrations"].append(arb)

    def record_prob(self, case_name, round_no, organ, prob, alpha=None, note=""):
        self._case_records.setdefault(case_name, {"steps": [], "llm_calls": [], "messages": [], "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}})
        self._case_records[case_name]["prob_trajectory"].append({
            "round": round_no, "organ": organ, "prob": prob, "alpha": alpha, "note": note,
        })

    def record_step(self, case_name, step):
        self._case_records.setdefault(case_name, {"steps": [], "llm_calls": [], "messages": [], "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}})
        self._case_records[case_name]["steps"].append(step)

    def finalize_case(self, case_name, final):
        self._case_records.setdefault(case_name, {"steps": [], "llm_calls": [], "messages": [], "gates": [], "arbitrations": [], "prob_trajectory": [], "final": {}})
        self._case_records[case_name]["final"] = final
        # Flush immediately: save each case as it completes to avoid data loss on interruption
        self.flush_case(case_name)

    def flush_case(self, case_name):
        """Write a single case record to disk immediately."""
        rec = self._case_records.get(case_name)
        if rec is None:
            return
        path = self.cases_dir / f"{case_name}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self):
        """Write all case records to disk."""
        for name in self._case_records:
            self.flush_case(name)
        if self._fh and not self._fh.closed:
            self._fh.close()

    def write_summary(self, summary):
        path = self.log_dir / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def weighted_aggregation(organ_probs, weights):
    """Weighted deterministic aggregation (weighted average of probabilities, then normalised).

    organ_probs: list[dict], each organ {candidate: probability}
    weights: list[float], aligned with organ_probs
    Returns dict {candidate: probability}
    """
    assert len(organ_probs) == len(weights) and len(organ_probs) > 0
    keys = list(organ_probs[0].keys())
    n = len(keys)
    acc = [0.0] * n
    wsum = sum(w for w in weights if w > 0)
    if wsum <= 0:
        weights = [1.0] * len(organ_probs)
        wsum = sum(weights)
    for probs, w in zip(organ_probs, weights):
        for i, k in enumerate(keys):
            acc[i] += (w / wsum) * float(probs.get(k, 0.0))
    total = sum(acc)
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: v / total for k, v in zip(keys, acc)}
