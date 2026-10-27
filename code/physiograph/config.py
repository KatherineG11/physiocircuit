"""Configuration loading and validation for PhysioCircuit.

JSON is the canonical on-disk format.  YAML is accepted when PyYAML is
already installed, but it is never required by the package.  API secrets are
referenced by environment-variable name and are not copied into manifests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import GraphConfig, MechanismCard


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "paper.json"


class ConfigError(ValueError):
    """Raised when a paper configuration is incomplete or inconsistent."""


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigError("YAML requires PyYAML; use the canonical JSON config instead") from exc
        value = yaml.safe_load(text)
    else:
        raise ConfigError(f"unsupported configuration format: {path}")
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be an object: {path}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    expected_snapshot: str | None = None
    reasoning_mode: str = "disabled"

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> "ModelSpec":
        missing = [k for k in ("provider", "base_url", "model", "api_key_env") if not value.get(k)]
        if missing:
            raise ConfigError(f"model {name!r} is missing: {', '.join(missing)}")
        return cls(
            name=name,
            provider=str(value["provider"]),
            base_url=str(value["base_url"]).rstrip("/"),
            model=str(value["model"]),
            api_key_env=str(value["api_key_env"]),
            expected_snapshot=(str(value["expected_snapshot"]) if value.get("expected_snapshot") else None),
            reasoning_mode=str(value.get("reasoning_mode", "disabled")),
        )


@dataclass(frozen=True)
class PaperConfig:
    source_path: Path
    raw: Mapping[str, Any]

    @property
    def root(self) -> Path:
        configured = self.raw.get("project_root")
        if configured:
            p = Path(str(configured))
            return p if p.is_absolute() else (self.source_path.parent / p).resolve()
        # Installed layout: code/physiograph/configs/paper.json.
        return PACKAGE_DIR.parents[1]

    @property
    def main_model(self) -> str:
        return str(self.raw.get("main_model", "qwen"))

    @property
    def models(self) -> dict[str, ModelSpec]:
        values = self.raw.get("models", {})
        if not isinstance(values, Mapping):
            raise ConfigError("models must be an object")
        return {str(k): ModelSpec.from_dict(str(k), v) for k, v in values.items()}

    @property
    def generation(self) -> Mapping[str, Any]:
        return dict(self.raw.get("generation", {}))

    @property
    def inference(self) -> Mapping[str, Any]:
        return dict(self.raw.get("inference", {}))

    @property
    def data(self) -> Mapping[str, Any]:
        return dict(self.raw.get("data", {}))

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def graph_path(self, domain: str) -> Path:
        domains = self.raw.get("domains", {})
        if domain not in domains:
            raise ConfigError(f"unknown domain {domain!r}; expected one of {sorted(domains)}")
        return self.resolve_path(str(domains[domain]["graph"]))

    def cards_path(self, domain: str) -> Path:
        domains = self.raw.get("domains", {})
        if domain not in domains:
            raise ConfigError(f"unknown domain {domain!r}; expected one of {sorted(domains)}")
        return self.resolve_path(str(domains[domain]["cards"]))

    def case_path(self, domain: str) -> Path:
        domains = self.raw.get("domains", {})
        if domain not in domains:
            raise ConfigError(f"unknown domain {domain!r}; expected one of {sorted(domains)}")
        return self.resolve_path(str(domains[domain]["cases"]))

    def redacted_snapshot(self) -> dict[str, Any]:
        value = copy.deepcopy(dict(self.raw))
        for model in value.get("models", {}).values():
            model.pop("api_key", None)
        return value

    def validate(self, require_api_key: bool = False) -> list[str]:
        errors: list[str] = []
        models = self.models
        if self.main_model not in models:
            errors.append(f"main_model {self.main_model!r} is not present in models")
        generation = self.generation
        if float(generation.get("temperature", 0)) != 0.2:
            errors.append("paper main configuration requires temperature=0.2")
        if int(generation.get("max_tokens", 0)) != 1024:
            errors.append("paper main configuration requires max_tokens=1024")
        alpha_max = float(self.inference.get("alpha_max", 0.6))
        if not 0.0 <= alpha_max < 1.0:
            errors.append("alpha_max must be in [0, 1)")
        if require_api_key:
            for name, model in models.items():
                if not model.api_key:
                    errors.append(f"model {name!r}: environment variable {model.api_key_env} is empty")
        for domain in self.raw.get("domains", {}):
            for label, path in (("graph", self.graph_path(domain)), ("cards", self.cards_path(domain))):
                if not path.exists():
                    errors.append(f"domain {domain!r} {label} file does not exist: {path}")
        return errors


def load_paper_config(path: str | os.PathLike[str] | None = None, overrides: Mapping[str, Any] | None = None) -> PaperConfig:
    source = Path(path) if path else DEFAULT_CONFIG
    source = source.resolve()
    raw = _read_mapping(source)
    if overrides:
        raw = deep_merge(raw, overrides)
    config = PaperConfig(source_path=source, raw=raw)
    errors = config.validate(require_api_key=False)
    if errors:
        raise ConfigError("invalid configuration:\n- " + "\n- ".join(errors))
    return config


def load_cards(path: str | os.PathLike[str]) -> tuple[MechanismCard, ...]:
    payload = _read_mapping(Path(path))
    values = payload.get("cards")
    if not isinstance(values, list):
        raise ConfigError(f"mechanism card file must contain a cards array: {path}")
    cards = tuple(MechanismCard.from_dict(item) for item in values)
    ids = [card.mechanism_id for card in cards]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"duplicate mechanism_id in {path}")
    return cards


def load_graph(path: str | os.PathLike[str], cards: Iterable[MechanismCard]) -> GraphConfig:
    payload = _read_mapping(Path(path))
    graph = GraphConfig.from_dict(payload)
    card_map = {card.mechanism_id: card for card in cards}
    referenced = {mid for edge in graph.edges for mid in edge.mechanism_ids}
    missing = sorted(referenced - set(card_map))
    if missing:
        raise ConfigError(f"graph references missing mechanism cards: {missing}")
    for edge in graph.edges:
        for mid in edge.mechanism_ids:
            card = card_map[mid]
            if card.source != edge.source or card.target != edge.target:
                raise ConfigError(
                    f"edge {edge.source}->{edge.target} contains {mid}, but card endpoints are "
                    f"{card.source}->{card.target}"
                )
    return graph
