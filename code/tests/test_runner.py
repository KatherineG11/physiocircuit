from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from physiograph.runner import run_experiment, score_api_run  # noqa: E402


class RunnerIntegrationTests(unittest.TestCase):
    def test_nested_engine_probabilities_and_case_subset_are_scored(self) -> None:
        def run_pg(case):
            return {
                "case_id": case["case_id"],
                "domain": "digestive",
                "steps": [
                    {
                        "step": 3,
                        "final_estimates": {
                            "stomach": {
                                "organ": "stomach",
                                "step": 3,
                                "probabilities": {
                                    "improve": 0.8,
                                    "stable": 0.1,
                                    "worsen": 0.1,
                                },
                            }
                        },
                        "llm_calls": [],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            gold_dir = root / "gold"
            gold_dir.mkdir()
            selected = {
                "case_id": "selected",
                "field": "digestive",
                "gold": {
                    "steps": [
                        {"t": 1, "direction": {"stomach": "improve"}, "phase": "baseline"}
                    ]
                },
            }
            unselected = {
                "case_id": "unselected",
                "field": "digestive",
                "gold": {
                    "steps": [
                        {"t": 1, "direction": {"stomach": "worsen"}, "phase": "decompensation"}
                    ]
                },
            }
            (gold_dir / "selected.json").write_text(json.dumps(selected), encoding="utf-8")
            (gold_dir / "unselected.json").write_text(json.dumps(unselected), encoding="utf-8")

            run_experiment(
                config={},
                cases=[{"case_id": "selected", "horizon": 1}],
                adapter={"run_pg": run_pg},
                resources={"graph": object(), "cards": object()},
                run_dir=run_dir,
                domain="digestive",
                methods=["PG"],
                model_name="mock",
                run_indices=[0],
            )
            score = score_api_run(run_dir, gold_path=gold_dir)

        record = score["run_scores"][0]
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["metrics"]["n_items"], 1)
        self.assertEqual(record["metrics"]["n_missing"], 0)
        self.assertEqual(record["metrics"]["direction_accuracy"], 1.0)
        self.assertAlmostEqual(record["metrics"]["brier"], 0.06)


if __name__ == "__main__":
    unittest.main()
