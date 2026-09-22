# PhysioCircuit reference implementation

This package implements the inference logic described in
`PhysioCircuit_iclr2027.tex`. It is independent of the older `plan1_*` and
`plan2_*` prototypes.

## Implemented pipeline

For every forecast step, the engine performs these operations in order:

1. each organ agent predicts its own `improve / stable / worsen` distribution;
2. senders emit typed, atomic mechanism messages;
3. all fourteen deterministic gates run before any receiver sees a message;
4. receivers return `accept / partially_accept / reject` verdicts;
5. accepted information produces the bounded update
   `P' = (1-alpha)P + alpha Q`, with `alpha <= 0.6`;
6. a deterministic public aggregator produces the patient-level phase;
7. the revised state is rolled forward as explicitly marked predicted context.

The digestive, pneumonia, and cardiovascular-sepsis systems use the same
program and schemas. They differ only in graph and mechanism-card JSON files.
The six interventions are implemented as immutable, SHA-256-audited edits:
delete, replace, reverse, extend lag, inject a wrong message, and mask local
evidence.

Private `gold`, per-case `mechanisms`, design cards, expected intervention
effects, and future labels are split away before inference. Raw provider
responses, token usage, gate decisions, messages, verdicts, and aggregate
states are written to separate JSONL audit streams.

## Baselines and ablations

The same engine adapter now executes PG and the baselines that can be defined
without an additional trained artifact: A (global), B (global + deterministic
card retrieval), C (independent organs), D (typed messages with fail-open
gates), F (compute-matched self-review), G (free-text graph), H (seeded random
graph), I (fully connected/reversed/key-edge-deleted topology controls), and J
(no-gate/unbounded-update controls).  Each uses the same case split, provider
settings, and output schema.

E is enabled only when frozen dev-fitted voting weights are supplied; K needs a
frozen protocol artifact; L needs a dev-trained implicit graph. Missing
artifacts produce an explicit `unavailable` record rather than a fabricated
prediction. The `main` suite intentionally retains these paper methods so an
incomplete experimental installation is visible in the run manifest.

## Layout

```text
physiograph/
  schema.py          strict immutable data contracts
  data.py            public/private split and deterministic evidence routing
  prompts.py         prompt allowlists and anti-leakage checks
  llm.py             mock and OpenAI-compatible API clients
  gates.py           fourteen deterministic pre-receiver gates
  engine.py          full PhysioCircuit rollout
  interventions.py   six generic audited operations
  aggregation.py     bounded update and eight-phase aggregation
  baselines.py       A--L baseline registry and availability checks
  metrics.py         accuracy, trajectory, calibration, IDAR, stability, cost
  runner.py          resumable runs, logs, scoring, and statistics
  legacy.py          explicit loading/replay of precomputed L1 artifacts
  cli.py             command-line entry point
  configs/           paper, graph, and mechanism-card configurations
```

## API configuration

`configs/paper.json` contains each provider's base URL, model identifier, and
the name of the environment variable holding its key. Set only the key(s) for
the model being run:

```powershell
$env:DASHSCOPE_API_KEY = "..."
$env:MISTRAL_API_KEY = "..."
$env:DEEPSEEK_API_KEY = "..."
$env:PYTHONPATH = "$PWD\code"
```

The main paper model is `qwen`; `mistral` and `deepseek` are configured for the
separate backbone robustness sweep. Generation is fixed at temperature `0`,
top-p `1`, and at most `1024` output tokens.

## Validate before spending API credits

```powershell
python -m physiograph validate --domains digestive,pneumonia,cardio
python -m physiograph run --domains digestive --methods PG --runs 0 `
  --case-limit 1 --output runs\smoke --dry-run
```

## Run real API inference

Start with one case and one run because a full organ rollout uses several API
calls per future step:

```powershell
python -m physiograph run --domains digestive --methods PG --model qwen `
  --runs 0 --split blind_test --case-limit 1 --output runs\pg_smoke
```

Then expand the case count and repetitions. The default `main` suite includes
paper baselines whose external artifacts may not yet be installed. Such a
method is recorded as `unavailable`; the runner never fabricates a prediction.

Run controlled base/edited pairs with an intervention JSON file:

```powershell
python -m physiograph intervene --config code\physiograph\configs\paper.json `
  --interventions interventions.json --domains digestive --methods PG `
  --runs 0 --case-limit 1 --output runs\pg_interventions
```

Each record needs `intervention_id`, `case_id`, and one of the following forms.
Expected outcomes may be present in the source file, but are stripped before
inference and used only by the scorer.

```json
{"type":"delete","mechanism_id":"MC-01"}
{"type":"replace","mechanism_id":"MC-01","replacement_card":{"...":"full card schema"}}
{"type":"reverse","mechanism_id":"MC-01"}
{"type":"extend_lag","mechanism_id":"MC-01","additional_steps":2}
{"type":"inject_wrong_message","mechanism_id":"MC-01"}
{"type":"mask_local_evidence","organ":"gallbladder","evidence_ids":["obs:1:lab:total_bilirubin"]}
```

Scoring is deliberately separate from inference:

```powershell
python -m physiograph score --run-dir runs\pg_smoke `
  --gold dataset_pg\cases --output runs\pg_smoke\scores.json
```

## Precomputed L1 compatibility

L1 may be read directly as a precomputed reference:

```powershell
python -m physiograph replay-l1 --dataset dataset_pg --domain digestive `
  --output runs\l1_digestive
```

The output is tagged `source_kind=precomputed_l1`, keeping it distinguishable
from newly generated `source_kind=api_run` records.

## Tests

```powershell
python -m unittest discover -s code\tests -v
```

The tests require no network and cover the fourteen-gate order, delayed message
delivery, receiver arbitration, bounded revision, executable baseline adapters,
all six interventions, probabilistic metrics, IDAR, and statistical utilities.
