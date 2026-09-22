from __future__ import annotations

import unittest

from physiograph.interventions import (
    apply_intervention,
    delete_mechanism_message,
    extend_lag,
    inject_wrong_message,
    mask_local_evidence,
    replace_mechanism,
    reverse_direction,
)
from physiograph.schema import (
    Direction,
    EvidenceRef,
    GraphConfig,
    InterventionKind,
    InterventionSnapshot,
    MechanismCard,
    MechanismMessage,
)


class InterventionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphConfig.from_dict(
            {
                "domain": "pneumonia",
                "nodes": ["lung", "kidney", "heart"],
                "edges": [
                    {
                        "source": "lung",
                        "target": "kidney",
                        "mechanism_ids": ["m1"],
                    },
                    {
                        "source": "kidney",
                        "target": "heart",
                        "mechanism_ids": ["m2"],
                    },
                ],
                "aggregation_weights": {
                    "heart": 0.2,
                    "kidney": 0.3,
                    "lung": 0.5,
                },
                "core_organs": ["lung"],
                "phase_thresholds": {},
            }
        )
        self.card = MechanismCard.from_dict(
            {
                "mechanism_id": "m1",
                "source": "lung",
                "target": "kidney",
                "source_variables": ["oxygenation"],
                "target_variables": ["renal_stress"],
                "allowed_directions": ["worsen"],
                "lag_steps": [0, 2],
                "activation_evidence": ["hypoxemia"],
                "counterevidence": [],
                "treatment_bindings": [],
                "mechanism_description": "test mechanism",
            }
        )
        self.card2 = MechanismCard.from_dict(
            {
                "mechanism_id": "m2",
                "source": "kidney",
                "target": "heart",
                "source_variables": ["volume"],
                "target_variables": ["load"],
                "allowed_directions": ["worsen"],
                "lag_steps": [1, 2],
                "activation_evidence": ["retention"],
                "counterevidence": [],
                "treatment_bindings": [],
                "mechanism_description": "second mechanism",
            }
        )
        self.evidence = EvidenceRef(
            evidence_id="e1",
            organ="lung",
            variable="oxygenation",
            step=1,
            text="Observed hypoxemia.",
            concept_code="hypoxemia",
        )
        self.other_evidence = EvidenceRef(
            evidence_id="e2",
            organ="kidney",
            variable="volume",
            step=1,
            text="Observed retention.",
            concept_code="retention",
        )
        self.message = MechanismMessage(
            message_id="msg1",
            sender="lung",
            receiver="kidney",
            mechanism_id="m1",
            source_variable="oxygenation",
            target_variable="renal_stress",
            direction=Direction.WORSEN,
            strength=0.8,
            lag=1,
            evidence_ids=("e1",),
            confidence=0.9,
            created_at_step=1,
        )
        self.snapshot = InterventionSnapshot(
            graph=self.graph,
            cards=(self.card, self.card2),
            messages=(self.message,),
            evidence=(self.evidence, self.other_evidence),
        )


class TestInterventions(InterventionFixture):
    def assertAuditedCopy(self, result, kind: InterventionKind) -> None:
        self.assertIsNot(result.snapshot, self.snapshot)
        self.assertEqual(result.audit.kind, kind)
        self.assertEqual(result.audit.before_digest, self.snapshot.digest)
        self.assertEqual(result.audit.after_digest, result.snapshot.digest)
        self.assertNotEqual(result.audit.before_digest, result.audit.after_digest)
        self.assertTrue(result.audit.changes)

    def test_delete_message_returns_copy_and_audit(self) -> None:
        result = delete_mechanism_message(self.snapshot, "msg1")
        self.assertAuditedCopy(result, InterventionKind.DELETE)
        self.assertEqual(result.snapshot.messages, ())
        self.assertEqual(self.snapshot.messages, (self.message,))
        with self.assertRaises(KeyError):
            delete_mechanism_message(self.snapshot, "missing")

    def test_replace_card_rewires_edge_without_mutating_input(self) -> None:
        replacement = MechanismCard.from_dict(
            {
                "mechanism_id": "m_new",
                "source": "heart",
                "target": "lung",
                "source_variables": ["load"],
                "target_variables": ["edema"],
                "allowed_directions": ["worsen"],
                "lag_steps": [1, 3],
                "activation_evidence": ["overload"],
                "counterevidence": [],
                "treatment_bindings": [],
                "mechanism_description": "replacement",
            }
        )
        result = replace_mechanism(self.snapshot, "m1", replacement)

        self.assertAuditedCopy(result, InterventionKind.REPLACE)
        self.assertTrue(result.snapshot.graph.allows("heart", "lung", "m_new"))
        self.assertFalse(result.snapshot.graph.allows("lung", "kidney", "m1"))
        self.assertEqual(result.snapshot.cards[0], replacement)
        self.assertEqual(self.snapshot.cards[0], self.card)
        self.assertTrue(result.audit.requires_regeneration)

    def test_reverse_swaps_endpoints_and_variable_roles(self) -> None:
        result = reverse_direction(self.snapshot, "m1")
        reversed_card = result.snapshot.cards[0]

        self.assertAuditedCopy(result, InterventionKind.REVERSE)
        self.assertEqual((reversed_card.source, reversed_card.target), ("kidney", "lung"))
        self.assertEqual(reversed_card.source_variables, ("renal_stress",))
        self.assertEqual(reversed_card.target_variables, ("oxygenation",))
        self.assertTrue(result.snapshot.graph.allows("kidney", "lung", "m1"))
        self.assertFalse(self.snapshot.graph.allows("kidney", "lung", "m1"))

    def test_extend_lag_shifts_entire_window(self) -> None:
        result = extend_lag(self.snapshot, "m1", additional_steps=2)
        self.assertAuditedCopy(result, InterventionKind.EXTEND_LAG)
        self.assertEqual(result.snapshot.cards[0].lag_steps, (2, 3, 4))
        self.assertEqual(self.card.lag_steps, (0, 1, 2))
        with self.assertRaises(ValueError):
            extend_lag(self.snapshot, "m1", additional_steps=0)

    def test_inject_wrong_message_requires_a_gate_failure(self) -> None:
        wrong = MechanismMessage(
            message_id="wrong1",
            sender="lung",
            receiver="kidney",
            mechanism_id="m1",
            source_variable="oxygenation",
            target_variable="renal_stress",
            direction=Direction.IMPROVE,
            strength=0.9,
            lag=1,
            evidence_ids=("e1",),
            created_at_step=1,
        )
        result = inject_wrong_message(self.snapshot, wrong)
        self.assertAuditedCopy(result, InterventionKind.INJECT_WRONG_MESSAGE)
        self.assertEqual(result.snapshot.messages[-1], wrong)

        valid_copy = MechanismMessage.from_dict(
            {**self.message.to_dict(), "message_id": "valid2"}
        )
        with self.assertRaises(ValueError):
            inject_wrong_message(self.snapshot, valid_copy)

    def test_mask_evidence_keeps_original_and_dangling_message_for_gate_audit(self) -> None:
        result = mask_local_evidence(self.snapshot, "lung", ("e1",))
        self.assertAuditedCopy(result, InterventionKind.MASK_LOCAL_EVIDENCE)
        self.assertEqual(tuple(item.evidence_id for item in result.snapshot.evidence), ("e2",))
        self.assertEqual(tuple(item.evidence_id for item in self.snapshot.evidence), ("e1", "e2"))
        self.assertEqual(result.snapshot.messages, self.snapshot.messages)
        self.assertTrue(result.audit.requires_regeneration)

    def test_generic_dispatch_uses_exact_six_operation_names(self) -> None:
        result = apply_intervention(
            "mask_local_evidence", self.snapshot, organ="lung"
        )
        self.assertEqual(result.audit.kind, InterventionKind.MASK_LOCAL_EVIDENCE)
        with self.assertRaises(ValueError):
            apply_intervention("unknown", self.snapshot)


if __name__ == "__main__":
    unittest.main()
