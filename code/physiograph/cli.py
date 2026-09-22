"""Command-line interface for validation, API runs, scoring and legacy replay."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .aggregation import aggregate_estimates
from .baselines import MAIN_METHODS, list_methods
from .config import DEFAULT_CONFIG, PaperConfig, load_cards, load_graph, load_paper_config
from .data import load_cases, prompt_safe_payload
from .legacy import replay_l1
from .llm import MockLLMClient, OpenAICompatibleClient
from .runner import (
    MethodRequest,
    load_records,
    run_experiment,
    run_interventions,
    score_api_run,
)


SUITES: dict[str, tuple[str, ...]] = {
    "main": MAIN_METHODS,
    "all": ("PG", "A", "B", "C", "D", "E", "F", "G", "H", "I:fully_connected", "I:reversed", "I:edge_deleted", "J:no_gates", "J:unbounded", "K", "L"),
    "composition": ("PG", "C", "D", "E", "F", "G"),
    "topology": ("PG", "C", "H", "I:fully_connected", "I:reversed", "I:edge_deleted"),
    "gates": ("PG", "D", "J:no_gates"),
    "alpha": ("PG", "J:unbounded"),
    "intervention": ("PG", "A", "H", "I:fully_connected", "I:reversed"),
}


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _invoke(function: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def _read_optional_resource(config: PaperConfig, key: str, domain: str) -> Any:
    resources = config.raw.get("baseline_resources", {})
    value: Any = None
    if isinstance(resources, Mapping):
        candidate = resources.get(key)
        if isinstance(candidate, Mapping):
            value = candidate.get(domain)
        else:
            value = candidate
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        return value
    path = config.resolve_path(value)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resources(config: PaperConfig, domain: str) -> dict[str, Any]:
    cards = load_cards(config.cards_path(domain))
    graph = load_graph(config.graph_path(domain), cards)
    result: dict[str, Any] = {"cards": cards, "graph": graph}
    for key in ("retriever", "voting_weights", "protocol", "learned_graph"):
        value = _read_optional_resource(config, key, domain)
        if value is not None:
            result[key] = value
    return result


def _client(config: PaperConfig, model_name: str, *, mock: bool = False) -> Any:
    if mock:
        return MockLLMClient([], model="mock-physiograph")
    models = config.models
    if model_name not in models:
        raise ValueError(f"unknown model {model_name!r}; expected one of {sorted(models)}")
    model = models[model_name]
    if not model.api_key:
        raise ValueError(f"environment variable {model.api_key_env} is empty")
    raw_generation = config.raw.get("api", {})
    timeout = float(raw_generation.get("timeout", 180)) if isinstance(raw_generation, Mapping) else 180.0
    retries = int(raw_generation.get("retries", 3)) if isinstance(raw_generation, Mapping) else 3
    return OpenAICompatibleClient(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model,
        provider=model.provider,
        timeout=timeout,
        max_retries=retries,
    )


def _wrap_engine(engine: Any, resources: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only hooks genuinely implemented by the concrete engine."""

    adapter: dict[str, Any] = {}
    candidates = {
        "run_pg": ("run_pg", "run_case", "forecast", "run"),
        "run_global": ("run_global", "global_forecast"),
        "run_organs": ("run_organs", "predict_organs", "initial_estimates"),
        "run_graph": ("run_graph", "graph_forecast"),
        "run_compute_matched": ("run_compute_matched",),
        "mutate_graph": ("mutate_graph",),
        "run_protocol": ("run_protocol",),
        "run_implicit_graph": ("run_implicit_graph",),
        "compare_intervention": ("compare_intervention",),
    }
    for hook_name, names in candidates.items():
        for name in names:
            hook = getattr(engine, name, None)
            if callable(hook):
                adapter[hook_name] = hook
                break

    def aggregate(*, estimates: Any, weights: Any = None, **_: Any) -> Any:
        graph: Any = resources.get("graph")
        if weights:
            if hasattr(graph, "to_dict"):
                graph = graph.to_dict()
            graph = dict(graph) if isinstance(graph, Mapping) else {}
            graph["aggregation_weights"] = weights
        return aggregate_estimates(estimates, graph=graph).to_dict()

    adapter["aggregate"] = aggregate
    return adapter


def _build_adapter(
    config: PaperConfig,
    domain: str,
    client: Any,
    resources: Mapping[str, Any],
) -> Any:
    """Build an engine adapter without fixing the engine's constructor API."""

    try:
        module = importlib.import_module(".engine", package=__package__)
    except ModuleNotFoundError as exc:
        if exc.name not in {f"{__package__}.engine", "physiograph.engine"}:
            raise
        return {}
    kwargs = {
        "config": config,
        "domain": domain,
        "client": client,
        "llm": client,
        "llm_client": client,
        "graph": resources.get("graph"),
        "cards": resources.get("cards"),
        "inference": config.inference,
        "generation": config.generation,
        "alpha_max": float(config.inference.get("alpha_max", 0.6)),
        "temperature": float(config.generation.get("temperature", 0.0)),
        "top_p": float(config.generation.get("top_p", 1.0)),
        "max_tokens": int(config.generation.get("max_tokens", 1024)),
    }
    for name in ("build_adapter", "create_adapter"):
        builder = getattr(module, name, None)
        if callable(builder):
            value = _invoke(builder, **kwargs)
            return value
    for name in ("PhysioCircuitEngine", "Engine"):
        engine_type = getattr(module, name, None)
        if engine_type is not None:
            return _wrap_engine(_invoke(engine_type, **kwargs), resources)
    return {}


def _domains(config: PaperConfig, value: str | None) -> list[str]:
    selected = _split_csv(value)
    available = list(config.raw.get("domains", {}))
    if not selected:
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown domains: {unknown}; expected {available}")
    return selected


def _methods(args: argparse.Namespace) -> list[str]:
    explicit = _split_csv(getattr(args, "methods", None))
    if explicit:
        # Parse now so malformed variants fail before an expensive call.
        return [MethodRequest.parse(value).key for value in explicit]
    suite = getattr(args, "suite", "main")
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {sorted(SUITES)}")
    return list(SUITES[suite])


def command_validate(args: argparse.Namespace) -> int:
    config = load_paper_config(args.config)
    errors = config.validate(require_api_key=args.require_api_key)
    report: dict[str, Any] = {
        "status": "ok",
        "config": str(config.source_path),
        "domains": {},
        "methods": list_methods(),
        "errors": errors,
    }
    for domain in _domains(config, args.domains):
        domain_report: dict[str, Any] = {"errors": [], "warnings": []}
        try:
            cards = load_cards(config.cards_path(domain))
            graph = load_graph(config.graph_path(domain), cards)
            bundles = load_cases(config.case_path(domain), split=args.split, limit=args.case_limit)
            for bundle in bundles:
                prompt_safe_payload(bundle.public)
            domain_report.update(
                {
                    "cases": len(bundles),
                    "cards": len(cards),
                    "organs": len(graph.organs),
                    "graph_edges": len(graph.edges),
                    "legacy_files_split_in_memory": True,
                }
            )
        except Exception as exc:
            domain_report["errors"].append(f"{type(exc).__name__}: {exc}")
            errors.append(f"{domain}: {exc}")
        report["domains"][domain] = domain_report
    if errors:
        report["status"] = "error"
    _json_print(report)
    return 0 if not errors else 2


def command_run(args: argparse.Namespace) -> int:
    config = load_paper_config(args.config)
    domains = _domains(config, args.domains)
    methods = _methods(args)
    model_name = args.model or config.main_model
    runs = [int(value) for value in _split_csv(args.runs)]
    if args.dry_run:
        _json_print(
            {
                "status": "dry_run",
                "domains": domains,
                "methods": methods,
                "model": model_name,
                "runs": runs,
                "case_limit": args.case_limit,
            }
        )
        return 0
    client = _client(config, model_name)
    base_output = Path(args.output).expanduser().resolve()
    results: dict[str, Any] = {}
    for domain in domains:
        resources = _resources(config, domain)
        adapter = _build_adapter(config, domain, client, resources)
        bundles = load_cases(
            config.case_path(domain), split=args.split, limit=args.case_limit
        )
        target = base_output / domain if len(domains) > 1 else base_output
        results[domain] = run_experiment(
            config=config,
            cases=[bundle.public for bundle in bundles],
            adapter=adapter,
            resources=resources,
            run_dir=target,
            domain=domain,
            methods=methods,
            model_name=model_name,
            run_indices=runs,
            resume=args.resume,
            options={"generation": dict(config.generation), "inference": dict(config.inference)},
        )
    _json_print({"status": "completed", "runs": results})
    return 0


def command_score(args: argparse.Namespace) -> int:
    result = score_api_run(
        args.run_dir,
        gold_path=args.gold,
        intervention_gold_path=args.intervention_gold,
        output_path=args.output,
        ece_bins=args.ece_bins,
    )
    _json_print(result)
    return 0


def command_intervene(args: argparse.Namespace) -> int:
    config = load_paper_config(args.config)
    domains = _domains(config, args.domains)
    methods = _methods(args)
    model_name = args.model or config.main_model
    runs = [int(value) for value in _split_csv(args.runs)]
    interventions = load_records(args.interventions)
    if args.dry_run:
        _json_print(
            {
                "status": "dry_run",
                "domains": domains,
                "methods": methods,
                "model": model_name,
                "runs": runs,
                "interventions": len(interventions),
            }
        )
        return 0
    client = _client(config, model_name)
    base_output = Path(args.output).expanduser().resolve()
    results: dict[str, Any] = {}
    for domain in domains:
        resources = _resources(config, domain)
        adapter = _build_adapter(config, domain, client, resources)
        bundles = load_cases(config.case_path(domain), split=args.split, limit=args.case_limit)
        domain_interventions = [
            row
            for row in interventions
            if not row.get("domain") or str(row.get("domain")) == domain
        ]
        target = base_output / domain if len(domains) > 1 else base_output
        results[domain] = run_interventions(
            config=config,
            cases=[bundle.public for bundle in bundles],
            interventions=domain_interventions,
            adapter=adapter,
            resources=resources,
            run_dir=target,
            domain=domain,
            methods=methods,
            model_name=model_name,
            run_indices=runs,
        )
    _json_print({"status": "completed", "runs": results})
    return 0


def command_replay_l1(args: argparse.Namespace) -> int:
    target = replay_l1(
        args.dataset,
        args.output,
        domain=args.domain,
        names=_split_csv(args.names) or None,
        overwrite=args.force,
    )
    _json_print(
        {
            "status": "completed",
            "source_kind": "precomputed_l1",
            "scoreable_as_api_run": False,
            "output": str(target),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physiograph",
        description="Auditable PhysioCircuit experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate config, cases, cards and graphs")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate.add_argument("--domains")
    validate.add_argument("--split")
    validate.add_argument("--case-limit", type=int)
    validate.add_argument("--require-api-key", action="store_true")
    validate.set_defaults(handler=command_validate)

    run = subparsers.add_parser("run", help="run API inference without reading private gold")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--domains")
    run.add_argument("--suite", choices=sorted(SUITES), default="main")
    run.add_argument("--methods", help="comma-separated IDs, variants as I:reversed")
    run.add_argument("--model")
    run.add_argument("--runs", default="0,1,2")
    run.add_argument("--split", default="blind_test")
    run.add_argument("--case-limit", type=int)
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=command_run)

    score = subparsers.add_parser("score", help="score an API run against private gold")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--gold", required=True)
    score.add_argument("--intervention-gold")
    score.add_argument("--output")
    score.add_argument("--ece-bins", type=int, default=5)
    score.set_defaults(handler=command_score)

    intervene = subparsers.add_parser("intervene", help="run base/edited mechanism pairs")
    intervene.add_argument("--config", default=str(DEFAULT_CONFIG))
    intervene.add_argument("--interventions", required=True)
    intervene.add_argument("--domains")
    intervene.add_argument("--suite", choices=sorted(SUITES), default="intervention")
    intervene.add_argument("--methods")
    intervene.add_argument("--model")
    intervene.add_argument("--runs", default="0,1,2")
    intervene.add_argument("--split", default="blind_test")
    intervene.add_argument("--case-limit", type=int)
    intervene.add_argument("--output", required=True)
    intervene.add_argument("--dry-run", action="store_true")
    intervene.set_defaults(handler=command_intervene)

    replay = subparsers.add_parser(
        "replay-l1", help="read legacy precomputed L1 with non-API provenance"
    )
    replay.add_argument("--dataset", required=True)
    replay.add_argument("--domain", default="digestive")
    replay.add_argument("--names", help="optional comma-separated artifact names")
    replay.add_argument("--output", required=True)
    replay.add_argument("--force", action="store_true")
    replay.set_defaults(handler=command_replay_l1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        _json_print({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 2


__all__ = ["SUITES", "build_parser", "main"]
