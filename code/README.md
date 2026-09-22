# PhysioCircuit: Code

Configuration-only multi-organ inference engine for clinical trajectory forecasting.
Each organ is an LLM agent; expert-defined mechanism cards and a frozen graph govern
typed message passing, deterministic gating, arbitration, and bounded state updates.

## Layout

```text
code/
├── common/                  Shared utilities (LLM client, mechanisms, logging, gateway)
├── physiograph/             Core engine package
│   ├── configs/             JSON configs for paper experiments and MIMIC-IV validation
│   │   ├── cards/           Mechanism cards per domain (digestive, pneumonia, cardio)
│   │   ├── graphs/          Organ graph topologies per domain
│   │   ├── paper.json       Main paper experiment configuration
│   │   └── mimiciv_demo.json  Retrospective MIMIC-IV validation configuration
│   ├── engine.py            Multi-organ inference engine (frozen graph, typed messages)
│   ├── gates.py             Fourteen deterministic message gates
│   ├── aggregation.py       Deterministic public aggregation (organ directions + macro phases)
│   ├── interventions.py     Six mechanism intervention operations (delete, reverse, etc.)
│   ├── metrics.py           IDAR, direction agreement (DA), timing agreement (TA), ECE
│   ├── baselines.py         Baseline methods (global agent, implicit graph, Vital Trace-style, etc.)
│   ├── runner.py            Experiment runner, intervention runner, scorer
│   ├── data.py              Case loading, public/private split, prompt-safe payloads
│   ├── prompts.py           Prompt templates for organ agents and baselines
│   ├── schema.py            Data schemas (cases, cards, graphs, messages)
│   ├── config.py            Configuration loading and validation
│   ├── llm.py               OpenAI-compatible LLM client + mock client for testing
│   ├── paper_data.py        Paper benchmark data pipeline (audit, export)
│   ├── import_mimiciv.py    MIMIC-IV demo archive importer (retrospective ICU trajectories)
│   ├── cli.py               Command-line interface (validate / run / score / intervene)
│   └── legacy.py            Legacy L1 replay for provenance
└── tests/                   Unit tests (engine, gates, interventions, metrics, runner)
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

The API-backed experiments require provider credentials in environment variables
(see `physiograph/config.py` for the expected variable names). Do not commit `.env`
or API keys. Use `--dry-run` to check an experiment without making API calls.

## Quick start

Run from the `code/` directory.

### 1. Run unit tests

```powershell
python -m unittest discover -s tests -q
```

### 2. Validate config, cases, cards, and graphs

```powershell
python -m physiograph validate --config physiograph/configs/paper.json
```

This checks every domain: case count, mechanism card count, organ count, graph edges,
and prompt-safety of the public payload. It does not require an API key.

### 3. Audit and export the paper benchmark

```powershell
python -m physiograph.paper_data audit
python -m physiograph.paper_data all --output runs/paper_data
```

The pipeline verifies 120 cases/domain, 100 blind-test cases, counterfactual pairs,
and card/graph counts, then exports physically separated `public.jsonl` and `gold.jsonl`,
1,440 treatment events, and six semantic intervention lists. `manifest.json` records
SHA-256 hashes for every output file.

## CLI commands

All commands are accessed through `python -m physiograph <command>`.

### validate

Validate configuration, cases, cards, and graphs without API calls.

```powershell
python -m physiograph validate --config physiograph/configs/paper.json [--domains digestive,pneumonia,cardio]
```

### run

Run API inference on blind-test cases for one or more methods. Public prompts only;
private gold labels are never read.

```powershell
python -m physiograph run --config physiograph/configs/paper.json --output runs/main --suite main --runs 0,1,2
```

Available suites: `main`, `all`, `composition`, `topology`, `gates`, `alpha`, `intervention`.
Use `--methods PG,C,D` to select specific methods instead of a suite.
Use `--dry-run` to print the planned run without API calls.

### intervene

Run base/edited mechanism pairs for IDAR scoring.

```powershell
python -m physiograph intervene --config physiograph/configs/paper.json --interventions runs/paper_data/digestive/interventions.jsonl --output runs/intervene/digestive
```

### score

Score an API run against private gold labels (direction accuracy, turning-point F1,
full-trajectory accuracy, ECE).

```powershell
python -m physiograph score --run-dir runs/main/digestive --gold runs/paper_data/digestive/gold.jsonl --intervention-gold runs/paper_data/digestive/interventions.jsonl
```

### replay-l1

Replay legacy precomputed L1 records with non-API provenance.

```powershell
python -m physiograph replay-l1 --dataset ../dataset_pg --domain digestive --output runs/replay/digestive
```

## MIMIC-IV retrospective validation

The raw MIMIC-IV archive is not included. After obtaining the archive under its own
data-use terms, import 600 retrospective ICU trajectories:

```powershell
python -m physiograph.import_mimiciv --zip path\to\mimic-iv-clinical-database-demo-2.0.1.zip --output runs/mimiciv_physiocircuit --n 600
```

Output: `runs/mimiciv_physiocircuit/{digestive,pneumonia,cardio}/` with
`public.jsonl`, `gold.jsonl`, `treatments.jsonl`, and a top-level `manifest.json`.
Each case uses 6-hour bins, 3 observed windows, and 6 future windows. The gold labels
are computed from later real observations (retrospective), not from a synthetic truth
generator. Validate with:

```powershell
python -m physiograph validate --config physiograph/configs/mimiciv_demo.json
```

## Configuration

`physiograph/configs/paper.json` is the main experiment config. It defines:

- **domains**: digestive, pneumonia, cardiovascular-sepsis
- **models**: LLM provider endpoints (API keys from environment variables)
- **generation**: temperature, top-p, max tokens
- **inference**: alpha_max for bounded updates, gate settings
- **paths**: cases, cards, graphs per domain
- **baseline_resources**: retriever, voting weights, protocol, learned graph per domain

`physiograph/configs/mimiciv_demo.json` is the retrospective MIMIC-IV validation config.

## Output and logs

Each experiment run writes to its own output directory:

```text
runs/<experiment>/<domain>/
├── cases/              Per-case JSON (full prompt, response, trajectory, gate audit)
├── log/                Run log and summary
├── results.json        Aggregated results
└── manifest.json       Run metadata (model, config hash, timestamp, seed)
```

Every case is written immediately after completion. Case logs contain the full LLM
prompt, response, per-step trajectory, and program-gate audit record.

## License

See the top-level `LICENSE` file. The code license does not grant permission to
redistribute MIMIC-IV, MIMIC, or other third-party clinical data.
