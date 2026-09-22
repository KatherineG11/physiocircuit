"""Import the MIMIC-IV demo archive into PhysioCircuit's public/private case format.

The importer is intentionally dependency-free and streams CSV files directly
from the nested ``zip -> gzip -> csv`` archive.  It creates a retrospective
dataset; it does not modify the
paper source.

Example::

    python -m physiograph.import_mimiciv \
      --zip ..\\..\\agent\\code\\data\\MIMIC-IV\\MIMIC-IV-collaborative-research-database-demo-2.0.1.zip \
      --output runs/MIMIC-IV_PhysioCircuit --n 600
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DOMAINS = ("digestive", "pneumonia", "cardio")
ORGANS = {
    "digestive": ("stomach", "intestine", "gallbladder"),
    "pneumonia": ("airway", "alveoli", "pulmonary_vessels", "heart", "kidney", "systemic_inflammation"),
    "cardio": ("heart", "vasculature", "kidney", "liver", "coagulation", "lung"),
}
DOMAIN_TERMS = {
    "digestive": ("gastro", "bowel", "intestinal", "abdomen", "pancrea", "liver", "bile", "hepatic"),
    "pneumonia": ("pneumonia", "respirat", "pulmonary", "alveol", "pleural", "bronch", "oxygen"),
    "cardio": ("cardiac", "heart", "myocard", "shock", "sepsis", "vascular", "arrhythm", "coronary"),
}

LAB_NAMES = {
    "albumin", "alp", "alt", "ast", "bilirubin", "bun", "creatinine", "crp",
    "lactate", "platelet", "wbc", "inr", "pao2", "pco2", "ph", "procalcitonin",
}


def _number(value: Any) -> float | None:
    try:
        value = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _open_member(zf: zipfile.ZipFile, suffix: str) -> io.TextIOWrapper:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one archive member ending {suffix!r}, found {matches}")
    raw = zf.open(matches[0])
    return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8", errors="replace", newline="")


def _read_patient(zf: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    patients: dict[str, dict[str, Any]] = {}
    with _open_member(zf, "patient.csv.gz") as stream:
        for row in csv.DictReader(stream):
            sid = str(row.get("patientunitstayid", "")).strip()
            if sid:
                patients[sid] = {
                    "gender": row.get("gender", ""),
                    "age": row.get("age", ""),
                    "ethnicity": row.get("ethnicity", ""),
                    "unit_type": row.get("unittype", ""),
                    "admission_dx": row.get("apacheadmissiondx", ""),
                    "discharge_status": row.get("unitdischargestatus", row.get("hospitaldischargestatus", "")),
                    "discharge_offset": _number(row.get("unitdischargeoffset")),
                }
    return patients


def _read_diagnoses(zf: zipfile.ZipFile) -> dict[str, str]:
    values: defaultdict[str, list[str]] = defaultdict(list)
    with _open_member(zf, "diagnosis.csv.gz") as stream:
        for row in csv.DictReader(stream):
            sid = str(row.get("patientunitstayid", "")).strip()
            text = row.get("diagnosisstring", "")
            if sid and text:
                values[sid].append(str(text))
    return {sid: "; ".join(items[:12]) for sid, items in values.items()}


def _add(store: dict[str, dict[int, dict[str, list[float]]]], sid: str, offset: Any, key: str, value: Any) -> None:
    number = _number(value)
    minute = _number(offset)
    if number is None or minute is None:
        return
    if number < -1e6 or number > 1e6:
        return
    bucket = int(minute // 360)
    store[sid][bucket].setdefault(key, []).append(number)


def _read_observations(zf: zipfile.ZipFile) -> dict[str, dict[int, dict[str, list[float]]]]:
    observations: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    with _open_member(zf, "vitalPeriodic.csv.gz") as stream:
        for row in csv.DictReader(stream):
            sid = str(row.get("patientunitstayid", "")).strip()
            if not sid:
                continue
            for source, key in (("sao2", "sao2"), ("heartrate", "heart_rate"), ("respiration", "respiration"),
                                ("systemicmean", "map"), ("temperature", "temperature")):
                _add(observations, sid, row.get("observationoffset"), key, row.get(source))
    with _open_member(zf, "lab.csv.gz") as stream:
        for row in csv.DictReader(stream):
            name = _norm_text(row.get("labname"))
            key = next((candidate for candidate in LAB_NAMES if candidate in name), None)
            sid = str(row.get("patientunitstayid", "")).strip()
            if sid and key:
                _add(observations, sid, row.get("labresultoffset"), key, row.get("labresult"))
    return observations


def _read_treatments(zf: zipfile.ZipFile) -> dict[str, list[dict[str, Any]]]:
    treatments: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    with _open_member(zf, "treatment.csv.gz") as stream:
        for row in csv.DictReader(stream):
            sid = str(row.get("patientunitstayid", "")).strip()
            treatment = str(row.get("treatmentstring", "")).strip()
            offset = _number(row.get("treatmentoffset"))
            if sid and treatment and offset is not None:
                treatments[sid].append({"offset": int(offset), "treatment": treatment})
    return treatments


def _median(values: Iterable[float]) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    middle = len(data) // 2
    return data[middle] if len(data) % 2 else (data[middle - 1] + data[middle]) / 2.0


def _severity(key: str, value: float) -> float:
    bounds = {
        "sao2": (95.0, 75.0, True), "heart_rate": (80.0, 150.0, False), "respiration": (18.0, 40.0, False),
        "map": (75.0, 45.0, True), "temperature": (37.0, 40.0, False), "albumin": (4.0, 1.5, True),
        "alp": (100.0, 600.0, False), "alt": (40.0, 400.0, False), "ast": (40.0, 400.0, False),
        "bilirubin": (1.0, 8.0, False), "bun": (20.0, 100.0, False), "creatinine": (1.0, 6.0, False),
        "crp": (1.0, 20.0, False), "lactate": (1.0, 6.0, False), "platelet": (200.0, 20.0, True),
        "wbc": (10.0, 30.0, False), "inr": (1.1, 3.0, False), "pao2": (90.0, 40.0, True),
        "pco2": (45.0, 80.0, False), "ph": (7.4, 7.0, True), "procalcitonin": (0.5, 20.0, False),
    }
    good, bad, inverse = bounds.get(key, (0.0, 1.0, False))
    if inverse:
        score = (good - value) / max(1e-6, good - bad)
    else:
        score = (value - good) / max(1e-6, bad - good)
    return max(0.0, min(1.0, score))


def _organ_scores(domain: str, bucket: Mapping[str, list[float]]) -> dict[str, float]:
    values = {key: _median(items) for key, items in bucket.items() if items}
    def avg(keys: Iterable[str], fallback: float = 0.5) -> float:
        selected = [_severity(key, values[key]) for key in keys if key in values]
        return sum(selected) / len(selected) if selected else fallback
    systemic = avg(("heart_rate", "map", "temperature", "lactate"))
    if domain == "digestive":
        return {"stomach": avg(("lactate", "ph", "wbc"), systemic),
                "intestine": avg(("albumin", "wbc", "crp"), systemic),
                "gallbladder": avg(("bilirubin", "alp", "alt"), systemic)}
    if domain == "pneumonia":
        return {"airway": avg(("respiration", "pco2"), systemic),
                "alveoli": avg(("sao2", "pao2"), systemic),
                "pulmonary_vessels": avg(("map", "sao2"), systemic),
                "heart": avg(("heart_rate", "map", "lactate"), systemic),
                "kidney": avg(("creatinine", "bun"), systemic),
                "systemic_inflammation": avg(("wbc", "crp", "temperature", "procalcitonin"), systemic)}
    return {"heart": avg(("heart_rate", "lactate", "troponin"), systemic),
            "vasculature": avg(("map", "lactate"), systemic),
            "kidney": avg(("creatinine", "bun"), systemic),
            "liver": avg(("bilirubin", "alt", "ast", "albumin"), systemic),
            "coagulation": avg(("platelet", "inr"), systemic),
            "lung": avg(("sao2", "respiration", "pao2"), systemic)}


def _domain_for(text: str) -> str:
    scores = {domain: sum(term in text for term in terms) for domain, terms in DOMAIN_TERMS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "cardio"


def _direction(delta: float) -> str:
    if delta > 0.035:
        return "worsen"
    if delta < -0.035:
        return "improve"
    return "stable"


def build(zip_path: str | Path, output: str | Path, n: int = 600, seed: int = 20260922) -> dict[str, Any]:
    rng = random.Random(seed)
    output = Path(output).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        patients = _read_patient(zf)
        diagnoses = _read_diagnoses(zf)
        observations = _read_observations(zf)
        treatments = _read_treatments(zf)
    candidates: list[tuple[str, str]] = []
    for sid, patient in patients.items():
        bins = observations.get(sid, {})
        if len(bins) >= 9:
            candidates.append((sid, _domain_for(_norm_text(diagnoses.get(sid, "") + " " + patient.get("admission_dx", "")))))
    rng.shuffle(candidates)
    chosen: list[tuple[str, str]] = []
    per_domain = max(1, n // 3)
    for domain in DOMAINS:
        chosen.extend((sid, domain) for sid, detected in candidates if detected == domain and sum(d == domain for _, d in chosen) < per_domain)
    chosen_ids = {sid for sid, _ in chosen}
    for sid, detected in candidates:
        if len(chosen) >= n:
            break
        if sid not in chosen_ids:
            chosen.append((sid, detected)); chosen_ids.add(sid)
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} ICU stays have at least 9 six-hour bins; requested {n}")
    chosen = chosen[:n]
    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    treatment_rows: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for index, (sid, domain) in enumerate(chosen, 1):
        patient = patients[sid]
        bins = observations[sid]
        ordered = sorted(bins)[:9]
        scores = [_organ_scores(domain, bins[b]) for b in ordered]
        case_id = f"MIMIC-IV-{index:04d}"
        observed = []
        for step, bucket_id in enumerate(ordered[:3]):
            observed.append({"t": step, "note": f"MIMIC-IV offset bin {bucket_id * 360} minutes", "labs": {k: round(_median(v), 4) for k, v in bins[bucket_id].items()}})
        gold_steps = []
        for step in range(1, 7):
            before, after = scores[step + 1], scores[step + 2]
            directions = {organ: _direction(after[organ] - before[organ]) for organ in ORGANS[domain]}
            severity = {organ: round(after[organ], 4) for organ in ORGANS[domain]}
            gold_steps.append({"t": step, "direction": directions, "severity": severity, "phase": "transition"})
        public_rows.append({
            "case_id": case_id, "split": "blind_test", "field": domain, "source": "MIMIC-IV-demo",
            "patientunitstayid": sid, "stratum": "real_icu", "horizon": 6,
            "patient": {"age": patient["age"], "sex": patient["gender"], "ethnicity": patient["ethnicity"], "unit_type": patient["unit_type"]},
            "presentation": {"admission_diagnosis": patient["admission_dx"], "diagnoses": diagnoses.get(sid, "")},
            "observed_window": observed,
        })
        gold_rows.append({"case_id": case_id, "steps": gold_steps, "source": "MIMIC-IV-observed-future-window", "outcome": patient["discharge_status"]})
        for event_index, event in enumerate(sorted(treatments.get(sid, []), key=lambda item: item["offset"])[:4], 1):
            treatment_rows.append({"event_id": f"{case_id}-T{event_index:02d}", "case_id": case_id, "patientunitstayid": sid, **event, "source": "MIMIC-IV-demo"})
        counts[domain] += 1
    output.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        domain_public = [row for row in public_rows if row["field"] == domain]
        domain_ids = {row["case_id"] for row in domain_public}
        domain_gold = [row for row in gold_rows if row["case_id"] in domain_ids]
        domain_treat = [row for row in treatment_rows if row["case_id"] in domain_ids]
        target = output / domain; target.mkdir(parents=True, exist_ok=True)
        (target / "public.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in domain_public), encoding="utf-8")
        (target / "gold.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in domain_gold), encoding="utf-8")
        (target / "treatments.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in domain_treat), encoding="utf-8")
    manifest = {"schema_version": "physiocircuit.mimiciv.v1", "source": "MIMIC-IV Clinical Database demo 2.0.1", "requested_cases": n, "cases": len(public_rows), "cases_by_domain": dict(counts), "treatment_events": len(treatment_rows), "bin_minutes": 360, "observed_bins": 3, "future_bins": 6, "seed": seed, "note": "Retrospective MIMIC-IV validation data; not a synthetic truth generator."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Import MIMIC-IV demo data for PhysioCircuit")
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    print(json.dumps(build(args.zip_path, args.output, args.n, args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
