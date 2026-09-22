from __future__ import annotations

import unittest

from physiograph.gates import GateContext, DeterministicGateEngine, validate_message
from physiograph.schema import (
    Direction,
    EvidenceRef,
    GraphConfig,
    MechanismCard,
    MechanismMessage,
)


class GateFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphConfig.from_dict(
            {
                "domain": "pneumonia",
                "nodes": ["lung", "kidney"],
                "edges": [
                    {
                        "source": "lung",
                        "target": "kidney",
                        "mechanism_ids": ["hypoxic_renal_stress"],
                    }
                ],
                "aggregation_weights": {"kidney": 0.4, "lung": 0.6},
                "core_organs": ["lung"],
                "phase_thresholds": {"decompensation": 0.7},
            }
        )
        self.card = MechanismCard.from_dict(
            {
                "mechanism_id": "hypoxic_renal_stress",
                "source": "lung",
                "target": "kidney",
                "source_variables": ["oxygenation"],
                "target_variables": ["renal_stress"],
                "allowed_directions": ["worsen"],
                "lag_steps": [0, 2],
                "activation_evidence": ["documented hypoxemia"],
                "counterevidence": ["normal oxygenation"],
                "treatment_bindings": [
                    {"treatment": "oxygen", "action": "suppress"}
                ],
                "mechanism_description": "Hypoxemia increases renal stress.",
            }
        )
        self.evidence = EvidenceRef(
            evidence_id="ev1",
            organ="lung",
            variable="oxygenation",
            step=1,
            text="Oxygen saturation is falling.",
            concept_code="hypoxemia",
        )
        self.message = MechanismMessage(
            message_id="msg1",
            sender="lung",
            receiver="kidney",
            mechanism_id="hypoxic_renal_stress",
            source_variable="oxygenation",
            target_variable="renal_stress",
            direction=Direction.WORSEN,
            strength=0.7,
            lag=1,
            evidence_ids=("ev1",),
            confidence=0.8,
            created_at_step=2,
        )


class TestFourteenGates(GateFixture):
    def test_valid_message_passes_all_fourteen_pre_receiver_gates(self) -> None:
        report = validate_message(
            self.message,
            self.card,
            self.graph,
            {"ev1": self.evidence},
            current_step=2,
        )

        self.assertTrue(report.eligible)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.decisions), 14)
        self.assertEqual(
            tuple(item.gate_number for item in report.decisions), tuple(range(1, 15))
        )
        self.assertEqual(len({item.name for item in report.decisions}), 14)
        self.assertEqual(report.failed_gate_names, ())
        self.assertTrue(all(item.reason for item in report.decisions))
        self.assertEqual(report.to_dict()["eligible"], True)

    def test_direction_and_lag_are_independent_semantic_gates(self) -> None:
        invalid = MechanismMessage.from_dict(
            {
                **self.message.to_dict(),
                "direction": "stable",
                "lag": 4,
            }
        )
        report = validate_message(
            invalid, self.card, self.graph, (self.evidence,), current_step=2
        )

        self.assertFalse(report.eligible)
        self.assertIn("direction_allowed", report.failed_gate_names)
        self.assertIn("lag_allowed", report.failed_gate_names)
        by_name = {item.name: item for item in report.decisions}
        self.assertEqual(by_name["direction_allowed"].layer, "semantic")
        self.assertIn("forbidden", by_name["direction_allowed"].reason)

    def test_future_predicted_evidence_and_option_leakage_are_blocked(self) -> None:
        leaked = EvidenceRef(
            evidence_id="ev1",
            organ="lung",
            variable="oxygenation",
            step=3,
            text="Future target says option B is correct.",
            concept_code="hypoxemia",
            predicted=True,
        )
        report = validate_message(
            self.message, self.card, self.graph, (leaked,), current_step=2
        )

        self.assertIn("historical_evidence_only", report.failed_gate_names)
        self.assertIn("no_option_label_leakage", report.failed_gate_names)

    def test_novelty_and_atomic_edge_limit_are_pre_receiver_gates(self) -> None:
        report = validate_message(
            self.message,
            self.card,
            self.graph,
            (self.evidence,),
            receiver_seen_fingerprints=(self.message.fingerprint,),
            current_step=2,
            edge_message_count=1,
            max_messages_per_edge=1,
        )

        self.assertIn("message_novelty", report.failed_gate_names)
        self.assertIn("atomic_edge_message_limit", report.failed_gate_names)

    def test_treatment_binding_can_ground_counterevidence(self) -> None:
        oxygen = EvidenceRef(
            evidence_id="ev2",
            organ="lung",
            variable="treatment",
            step=1,
            text="treatment=oxygen",
        )
        message = MechanismMessage.from_dict(
            {
                **self.message.to_dict(),
                "counterevidence_ids": ["ev2"],
            }
        )
        report = validate_message(
            message,
            self.card,
            self.graph,
            (self.evidence, oxygen),
            current_step=2,
        )

        self.assertTrue(report.eligible)
        self.assertNotIn("counterevidence_traceability", report.failed_gate_names)

    def test_engine_accepts_explicit_immutable_context(self) -> None:
        context = GateContext(
            graph=self.graph,
            card=self.card,
            evidence=(self.evidence,),
            current_step=2,
            available_steps=frozenset({0, 1, 2}),
        )
        report = DeterministicGateEngine().evaluate(self.message, context)
        self.assertTrue(report.eligible)

    def test_card_json_range_is_inclusive_and_unknown_fields_are_rejected(self) -> None:
        self.assertEqual(self.card.lag_steps, (0, 1, 2))
        self.assertEqual(self.card.to_dict()["lag_steps"], [0, 2])
        data = self.graph.to_dict()
        data["unexpected"] = True
        with self.assertRaises(ValueError):
            GraphConfig.from_dict(data)


if __name__ == "__main__":
    unittest.main()
