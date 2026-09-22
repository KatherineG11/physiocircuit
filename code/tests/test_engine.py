from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from physiograph.engine import PhysioCircuitEngine, build_adapter  # noqa: E402
from physiograph.baselines import run_method  # noqa: E402
from physiograph.llm import MockLLMClient  # noqa: E402
from physiograph.schema import (  # noqa: E402
    Direction,
    Edge,
    EvidenceRef,
    GateDecision,
    GateReport,
    GraphConfig,
    MechanismCard,
    ObservationNode,
    PatientCase,
)


def _decision(number: int, name: str) -> GateDecision:
    try:
        return GateDecision(number, name, True, "ok", layer="structural" if number <= 2 else "semantic")
    except TypeError:
        return GateDecision(number, name, True, "ok")


def pass_all_gates(**kwargs):
    message = kwargs["message"]
    return GateReport(
        message_id=message.message_id,
        decisions=tuple(_decision(index, f"gate_{index:02d}") for index in range(1, 15)),
    )


def graph_and_cards():
    graph = GraphConfig(
        domain="renal_cardiac",
        nodes=("heart", "kidney"),
        edges=(Edge(source="kidney", target="heart", mechanism_ids=("renal_load",)),),
        aggregation_weights=(("heart", 0.5), ("kidney", 0.5)),
        core_organs=("heart",),
    )
    card = MechanismCard(
        mechanism_id="renal_load",
        source="kidney",
        target="heart",
        source_variables=("fluid_retention",),
        target_variables=("cardiac_load",),
        allowed_directions=(Direction.WORSEN,),
        lag_steps=(1,),
        activation_evidence=("rising fluid balance",),
        counterevidence=("effective diuresis",),
        mechanism_description="Renal fluid retention increases cardiac load after one step.",
    )
    return graph, (card,)


def patient_case():
    node = ObservationNode(
        step=0,
        time_label="admission",
        evidence=(
            EvidenceRef(
                evidence_id="e-heart",
                organ="heart",
                variable="cardiac_load",
                step=0,
                text="cardiac load is currently stable",
            ),
            EvidenceRef(
                evidence_id="e-kidney",
                organ="kidney",
                variable="fluid_retention",
                step=0,
                text="fluid balance is rising",
            ),
        ),
    )
    return PatientCase(
        case_id="case-1",
        domain="renal_cardiac",
        nodes=(node,),
        forecast_horizon=2,
        metadata=(("gold", "DO_NOT_SHOW"), ("stratum", "conflict")),
    )


class EngineTests(unittest.TestCase):
    def test_lagged_message_is_arbitrated_then_bounded(self):
        graph, cards = graph_and_cards()
        llm = MockLLMClient(
            [
                # Step 1: stable independent heart estimate.
                {
                    "organ": "heart",
                    "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                    "evidence_ids": ["e-heart"],
                    "outgoing_messages": [],
                },
                # Step 1: kidney emits a one-step delayed typed message.
                {
                    "organ": "kidney",
                    "probabilities": {"improve": 0.05, "stable": 0.15, "worsen": 0.8},
                    "evidence_ids": ["e-kidney"],
                    "outgoing_messages": [
                        {
                            "receiver": "heart",
                            "mechanism_id": "renal_load",
                            "source_variable": "fluid_retention",
                            "target_variable": "cardiac_load",
                            "direction": "worsen",
                            "strength": 0.9,
                            "lag": 1,
                            "confidence": 0.95,
                            "evidence_ids": ["e-kidney"],
                            "counterevidence_ids": [],
                        }
                    ],
                },
                # Step 2 independent estimates occur before delivery/arbitration.
                {
                    "organ": "heart",
                    "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                    "evidence_ids": ["e-heart"],
                    "outgoing_messages": [],
                },
                {
                    "organ": "kidney",
                    "probabilities": {"improve": 0.1, "stable": 0.7, "worsen": 0.2},
                    "evidence_ids": ["e-kidney"],
                    "outgoing_messages": [],
                },
                # Heart accepts; alpha=0.6 and P'=(.4 P + .6 Q).
                {
                    "verdicts": [
                        {
                            "message_id": "case-1:s1:kidney:m000",
                            "verdict": "accept",
                            "reason": "traceable renal load is relevant",
                            "weight": 1.0,
                            "conflict_evidence_ids": [],
                        }
                    ],
                    "conditioned_probabilities": {
                        "improve": 0.05,
                        "stable": 0.05,
                        "worsen": 0.9,
                    },
                },
            ]
        )
        engine = PhysioCircuitEngine(
            graph=graph,
            cards=cards,
            llm=llm,
        )
        result = engine.run(patient_case())

        self.assertEqual(len(result["steps"]), 2)
        first, second = result["steps"]
        self.assertEqual(first["delivered_messages"], [])
        self.assertEqual(first["gate_audits"][0]["due_step"], 2)
        self.assertEqual(len(first["gate_audits"][0]["gate_report"]["decisions"]), 14)
        self.assertEqual(len(second["delivered_messages"]), 1)
        self.assertEqual(second["alpha_by_organ"]["heart"], 0.6)
        heart = second["final_estimates"]["heart"]["probabilities"]
        self.assertAlmostEqual(heart["improve"], 0.07)
        self.assertAlmostEqual(heart["stable"], 0.35)
        self.assertAlmostEqual(heart["worsen"], 0.58)
        self.assertEqual(result["pending_messages"], [])
        self.assertEqual(list(second["initial_estimates"]), ["heart", "kidney"])
        self.assertEqual(len(llm.calls), 5)

        rendered = "\n".join(
            str(message["content"])
            for call in llm.calls
            for message in call["messages"]
        )
        self.assertNotIn("DO_NOT_SHOW", rendered)
        self.assertNotIn('"gold"', rendered.lower())
        self.assertNotIn('"stratum"', rendered.lower())

    def test_malformed_message_fails_closed_before_delivery(self):
        graph, cards = graph_and_cards()
        llm = MockLLMClient(
            [
                {
                    "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                    "evidence_ids": ["e-heart"],
                    "outgoing_messages": [],
                },
                {
                    "probabilities": {"improve": 0.1, "stable": 0.2, "worsen": 0.7},
                    "evidence_ids": ["e-kidney"],
                    "outgoing_messages": [{"receiver": "heart", "mechanism_id": "renal_load"}],
                },
            ]
        )
        engine = PhysioCircuitEngine(
            graph=graph,
            cards=cards,
            llm=llm,
            gate_validator=pass_all_gates,
        )
        result = engine.run(patient_case(), horizon=1)
        step = result["steps"][0]
        self.assertEqual(step["delivered_messages"], [])
        self.assertEqual(len(step["message_parse_failures"]), 1)
        report = step["message_parse_failures"][0]["gate_report"]
        self.assertFalse(report["eligible"])
        self.assertEqual(len(report["decisions"]), 14)

    def test_alpha_cannot_exceed_paper_bound(self):
        graph, cards = graph_and_cards()
        with self.assertRaises(ValueError):
            PhysioCircuitEngine(
                graph=graph,
                cards=cards,
                llm=MockLLMClient([]),
                gate_validator=pass_all_gates,
                alpha_max=0.61,
            )

    def test_runner_adapter_injects_wrong_message_before_gates(self):
        graph, cards = graph_and_cards()
        llm = MockLLMClient(
            [
                {
                    "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                    "evidence_ids": ["e-heart"],
                    "outgoing_messages": [],
                },
                {
                    "probabilities": {"improve": 0.1, "stable": 0.2, "worsen": 0.7},
                    "evidence_ids": ["e-kidney"],
                    "outgoing_messages": [],
                },
            ]
        )
        adapter = build_adapter(graph=graph, cards=cards, llm=llm)
        one_step_case = dataclasses.replace(patient_case(), forecast_horizon=1)
        result = adapter["run_pg"](
            one_step_case,
            intervention={
                "type": "inject_wrong_message",
                "mechanism_id": "renal_load",
                "step": 1,
            },
        )

        gate_audits = result["steps"][0]["gate_audits"]
        self.assertEqual(len(gate_audits), 1)
        self.assertFalse(gate_audits[0]["gate_report"]["eligible"])
        self.assertEqual(len(result["intervention"]["audits"]), 1)

    def test_runner_adapter_deletes_one_generated_mechanism_message(self):
        graph, cards = graph_and_cards()
        llm = MockLLMClient(
            [
                {
                    "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                    "evidence_ids": ["e-heart"],
                    "outgoing_messages": [],
                },
                {
                    "probabilities": {"improve": 0.1, "stable": 0.2, "worsen": 0.7},
                    "evidence_ids": ["e-kidney"],
                    "outgoing_messages": [
                        {
                            "receiver": "heart",
                            "mechanism_id": "renal_load",
                            "source_variable": "fluid_retention",
                            "target_variable": "cardiac_load",
                            "direction": "worsen",
                            "strength": 0.9,
                            "lag": 1,
                            "evidence_ids": ["e-kidney"],
                        }
                    ],
                },
            ]
        )
        adapter = build_adapter(graph=graph, cards=cards, llm=llm)
        one_step_case = dataclasses.replace(patient_case(), forecast_horizon=1)
        result = adapter["run_pg"](
            one_step_case,
            intervention={"type": "delete", "mechanism_id": "renal_load"},
        )

        self.assertEqual(result["steps"][0]["gate_audits"], [])
        self.assertEqual(result["pending_messages"], [])
        self.assertEqual(len(result["intervention"]["audits"]), 1)

    def test_executable_baseline_adapter_hooks(self):
        graph, cards = graph_and_cards()

        def factory(messages, options, index):
            system = messages[0]["content"]
            if "one global longitudinal" in system:
                return {
                    "organ_probabilities": {
                        "heart": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                        "kidney": {"improve": 0.1, "stable": 0.7, "worsen": 0.2},
                    },
                    "summaries": {},
                }
            if "Self-review" in system:
                return {
                    "conditioned_probabilities": {
                        "improve": 0.1,
                        "stable": 0.8,
                        "worsen": 0.1,
                    },
                    "reason": "local evidence is stable",
                }
            if "brief free-text" in system:
                return {"messages": []}
            return {
                "probabilities": {"improve": 0.1, "stable": 0.8, "worsen": 0.1},
                "evidence_ids": [],
                "outgoing_messages": [],
            }

        adapter = build_adapter(graph=graph, cards=cards, llm=MockLLMClient(factory))
        case = dataclasses.replace(patient_case(), forecast_horizon=1)
        resources = {"graph": graph, "cards": cards}
        ready = [
            ("A", None),
            ("B", None),
            ("C", None),
            ("D", None),
            ("F", None),
            ("G", None),
            ("H", None),
            ("I", "fully_connected"),
            ("J", "no_gates"),
            ("J", "unbounded"),
        ]
        for method, variant in ready:
            with self.subTest(method=method, variant=variant):
                outcome = run_method(
                    method,
                    case,
                    adapter,
                    resources=resources,
                    variant=variant,
                )
                self.assertEqual(outcome.status, "ok", outcome.reason)
                self.assertEqual(len(outcome.predictions["steps"]), 1)



if __name__ == "__main__":
    unittest.main()
