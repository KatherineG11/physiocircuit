# Reproducibility checklist

- Run the unit tests before any model call.
- Validate cards, graphs, and case files with the package CLI.
- Keep public prompts separate from private gold labels during scoring.
- Record model name, provider snapshot, temperature, top-p, token limit, seed,
  and data manifest hash for every run.
- Do not commit API credentials or raw clinical archives.
- Report the exact clinical database used (e.g., MIMIC-IV) in every experiment; do not substitute one dataset for another in claims or tables.
