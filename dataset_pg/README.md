# PhysioCircuit-Bench: Synthetic Benchmark Dataset

PhysioCircuit-Bench is a synthetic multi-organ clinical trajectory benchmark
designed for evaluating graph-structured multi-agent inference. It contains
three clinical domains (digestive, pneumonia, cardiovascular-sepsis), each
with 120 cases, mechanism cards, organ graphs, gold trajectories, and
counterfactual intervention pairs.

> This is a **synthetic** benchmark. Cases, trajectories, and gold labels are
> procedurally generated and must not be described as real clinical data.

## Layout

```text
dataset_pg/
  README.md            This file
  CASES.md             Case stratum distribution and examples
  cases_index.json     Case index (id / split / stratum / counterfactual pair)
  cases/*.json         120 digestive cases (presentation, vitals, mechanism cards, gold trajectory)
  spec.json            Numeric specification for each reported metric
  domains/
    digestive/          (legacy layout; primary cases are in cases/)
    pneumonia/          120 pneumonia cases + 24 counterfactual pairs
    cardio/             120 cardiovascular-sepsis cases + 24 counterfactual pairs
  l1/                  Per-item scoring records (direction, F1, interventions, gates, probes)
  scores_item.csv      Per-item correctness (readable export)
  scores_case.csv      Per-case summary (readable export)
```

## Scale

| Item | Count |
|---|---:|
| Cases per domain | 120 (20 dev / 100 blind test) |
| Counterfactual pairs | 30 (digestive) + 24 (pneumonia) + 24 (cardio) = 78 |
| Blind-test direction items | 1,800 per domain (100 cases x 6 steps x 3 organs) |
| Systems | 13 (PhysioCircuit + 12 baselines) |
| Runs per configuration | 3 |
| Intervention records | 6 operation types x 5 graph topologies x 1,000 items x 3 runs |
| Gate / probe records | 1,000 items each |

## Strata and weighting

Each domain's 100 blind-test cases are stratified:

| Stratum | Blind cases | Items | Weight |
|---|---:|---:|---:|
| simple | 30 | 540 | 30% |
| conflict | 40 | 720 | 40% |
| turning | 20 | 360 | 20% |
| nonmono | 10 | 180 | 10% |

Weighted overall = 0.30*simple + 0.40*conflict + 0.20*turning + 0.10*nonmono,
matching the Overall column in the paper's main results table.

## Data generation

The benchmark is pre-built and shipped as-is. `spec.json` records the integer
count specification from which every reported number is derived; `l1/` contains
the per-scoring-unit raw records; `scores_item.csv` and `scores_case.csv` are
human-readable exports. All files are deterministic and can be verified by
recomputing aggregates from `l1/`.

## Conventions

- All mean +/- std values are over **3 runs** with sample standard deviation (ddof=1).
- Direction accuracy denominator: n=1,800 per domain.
- Intervention / gate / probe / knowledge denominators: n=1,000 each.
- Full-trajectory accuracy denominator: 100 (blind-test case count).
- AUROC computed over 25x25=625 pairs with 0.5 tie weight.

## Data provenance

This dataset is synthetic and generated for research purposes. It does not
contain real patient data. Clinical-data experiments in the paper use a
separate, licensed MIMIC-IV source imported through `physiograph.import_mimiciv`;
those records are not redistributed here.
