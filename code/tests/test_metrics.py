import math
import unittest

from physiograph.metrics import (
    cost_metrics,
    direction_accuracy,
    expected_calibration_error,
    full_trajectory_accuracy,
    intervention_direction_agreement,
    mcnemar_test,
    multiclass_brier_score,
    negative_log_likelihood,
    paired_bootstrap,
    stability_metrics,
    turning_point_f1,
)


class DirectionMetricsTests(unittest.TestCase):
    def test_direction_accuracy_counts_missing_prediction_as_wrong(self) -> None:
        rows = [
            {"gold_direction": "improve", "pred_direction": "improve"},
            {"gold_direction": "stable", "pred_direction": "worsen"},
            {"gold_direction": "worsen", "pred_direction": None},
        ]
        self.assertAlmostEqual(direction_accuracy(rows), 1 / 3)

    def test_full_trajectory_requires_every_item_and_missing_case(self) -> None:
        rows = [
            {"case_id": "a", "gold_direction": "stable", "pred_direction": "stable"},
            {"case_id": "a", "gold_direction": "worsen", "pred_direction": "worsen"},
            {"case_id": "b", "gold_direction": "stable", "pred_direction": "improve"},
        ]
        self.assertEqual(full_trajectory_accuracy(rows), 0.5)
        expected = {"a": 2, "b": 1, "c": 1}
        self.assertAlmostEqual(
            full_trajectory_accuracy(rows, expected_items_by_case=expected), 1 / 3
        )

    def test_turning_point_f1_and_empty_positive_convention(self) -> None:
        self.assertAlmostEqual(
            turning_point_f1([True, False, True, False], [True, True, False, False]),
            0.5,
        )
        self.assertEqual(turning_point_f1([False, False], [False, False]), 1.0)

    def test_empty_direction_input_is_not_silently_zero(self) -> None:
        with self.assertRaises(ValueError):
            direction_accuracy([])


class ProbabilisticMetricsTests(unittest.TestCase):
    def test_perfect_probabilities(self) -> None:
        probabilities = [
            {"improve": 1.0, "stable": 0.0, "worsen": 0.0},
            {"improve": 0.0, "stable": 1.0, "worsen": 0.0},
        ]
        gold = ["improve", "stable"]
        self.assertEqual(multiclass_brier_score(probabilities, gold), 0.0)
        self.assertEqual(negative_log_likelihood(probabilities, gold), 0.0)
        self.assertEqual(expected_calibration_error(probabilities, gold), 0.0)

    def test_brier_normalizes_rounded_probability_mass(self) -> None:
        # The vector sums to two and is normalized to [0.5, 0.5, 0].
        value = multiclass_brier_score([[1.0, 1.0, 0.0]], ["improve"])
        self.assertAlmostEqual(value, 0.5)

    def test_ece_uses_equal_width_top_label_bins(self) -> None:
        probabilities = [
            {"improve": 0.8, "stable": 0.1, "worsen": 0.1},  # correct
            {"improve": 0.8, "stable": 0.1, "worsen": 0.1},  # wrong
        ]
        # Both occupy the same bin: |mean confidence .8 - accuracy .5| = .3.
        self.assertAlmostEqual(
            expected_calibration_error(probabilities, ["improve", "stable"], n_bins=5),
            0.3,
        )


class InterventionStabilityCostTests(unittest.TestCase):
    def test_idar_and_duplicate_guard(self) -> None:
        rows = [
            {
                "intervention_id": "i1",
                "expected_response": "improve",
                "predicted_response": "improve",
            },
            {
                "intervention_id": "i2",
                "expected_response": "worsen",
                "predicted_response": "stable",
            },
        ]
        self.assertEqual(intervention_direction_agreement(rows), 0.5)
        with self.assertRaises(ValueError):
            intervention_direction_agreement([rows[0], rows[0]])

    def test_stability_metrics(self) -> None:
        rows = [
            {
                "case_id": "c1",
                "organ": "lung",
                "step": 1,
                "gold_direction": "improve",
                "pred_direction": "improve",
                "probabilities": {"improve": 0.8, "stable": 0.1, "worsen": 0.1},
            },
            {
                "case_id": "c1",
                "organ": "lung",
                "step": 2,
                "gold_direction": "stable",
                "pred_direction": "stable",
                "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
            },
            {
                "case_id": "c1",
                "organ": "lung",
                "step": 5,
                "gold_direction": "worsen",
                "pred_direction": "stable",
                "probabilities": {"improve": 0.1, "stable": 0.7, "worsen": 0.2},
            },
        ]
        result = stability_metrics(rows)
        self.assertEqual(result["flips_per_case"], 1.0)
        self.assertEqual(result["first_step_error"], 0.0)
        self.assertEqual(result["final_step_error"], 1.0)
        self.assertEqual(result["error_growth"], 1.0)
        self.assertTrue(0.0 < result["oscillation"] < 1.0)

    def test_cost_aggregation(self) -> None:
        result = cost_metrics(
            [
                {
                    "case_id": "a",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "latency_ms": 100,
                    "cost_usd": 0.1,
                },
                {
                    "case_id": "b",
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "latency_ms": 200,
                    "cost_usd": 0.2,
                },
            ]
        )
        self.assertEqual(result["total_tokens"], 40.0)
        self.assertEqual(result["tokens_per_case"], 20.0)
        self.assertAlmostEqual(result["cost_usd"], 0.3)


class StatisticalTests(unittest.TestCase):
    def test_paired_bootstrap_is_reproducible(self) -> None:
        first = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], n_resamples=500, seed=7)
        second = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], n_resamples=500, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 0.5)
        self.assertLessEqual(first["ci_low"], first["estimate"])
        self.assertGreaterEqual(first["ci_high"], first["estimate"])

    def test_mcnemar_exact_two_sided(self) -> None:
        # A-only=3, B-only=0: exact two-sided p=2*(1/2^3)=.25.
        result = mcnemar_test(
            [True, True, True, False],
            [False, False, False, False],
        )
        self.assertEqual(result["a_correct_b_wrong"], 3.0)
        self.assertEqual(result["a_wrong_b_correct"], 0.0)
        self.assertAlmostEqual(result["p_exact_two_sided"], 0.25)


if __name__ == "__main__":
    unittest.main()
