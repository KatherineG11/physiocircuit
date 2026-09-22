# PhysioCircuit: Reproducible Code and Benchmark

This directory is the GitHub-ready release bundle for the PhysioCircuit 9.22
manuscript. It contains the configuration-only multi-organ inference engine,
mechanism cards and graphs, tests, synthetic benchmark assets, and scripts for
importing an externally supplied MIMIC-IV demo archive.

## Layout

```text
code/          Python package, tests, and package README
dataset_pg/    Synthetic benchmark cases, gold/L1 records, and manifests
docs/          Data and reproducibility notes
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd code
python -m unittest discover -s tests -q
python -m physiograph.paper_data audit
python -m physiograph validate --config physiograph/configs/paper.json
```

The API-backed experiments require provider credentials in environment
variables. Do not commit `.env` or keys. Use `--dry-run` to check an experiment
without making API calls.

## MIMIC-IV data

The raw MIMIC-IV archive is not included. After obtaining the archive under its
own data-use terms, run from `code/`:

```powershell
python -m physiograph.import_mimiciv `
  --zip path\to\mimic-iv-clinical-database-demo-2.0.1.zip `
  --output runs/mimiciv_physiocircuit `
  --n 600
```

The importer creates retrospective MIMIC-IV labels from later observations. These
records are retrospective MIMIC-IV labels, not a synthetic truth generator, and are not redistributed here.

## Data provenance

`dataset_pg/` is the public synthetic benchmark bundle. Its README describes
the case construction, score records, and verification scripts. The benchmark
must not be described as real clinical data. Clinical-data experiments require
the user's own licensed source archive and should be reported with the correct
database name.

## License

Code is released under the [MIT License](LICENSE). The code license does not
grant permission to redistribute MIMIC-IV, MIMIC, or other third-party clinical
data; those remain subject to their own data-use agreements.
