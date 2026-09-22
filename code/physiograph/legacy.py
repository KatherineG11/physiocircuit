"""Read-only support for historical precomputed L1 artifacts.

Legacy bitmaps were not produced by the API runner.  They are intentionally kept
in a separate namespace and carry a non-scoreable provenance marker so report
code cannot accidentally present them as new model executions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


API_RUN_SOURCE = "api_run"
PRECOMPUTED_L1_SOURCE = "precomputed_l1"
LEGACY_WARNING = (
    "Historical L1 values are precomputed artifacts, not raw API outputs. "
    "They may be replayed for compatibility but must not be reported as an API run."
)


@dataclass(frozen=True)
class LegacyBundle:
    root: Path
    domain: str
    files: dict[str, Any]
    hashes: dict[str, str]
    source_kind: str = PRECOMPUTED_L1_SOURCE
    scoreable_as_api_run: bool = False
    warning: str = LEGACY_WARNING

    def to_dict(self, *, include_payloads: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_kind": self.source_kind,
            "scoreable_as_api_run": self.scoreable_as_api_run,
            "warning": self.warning,
            "root": str(self.root),
            "domain": self.domain,
            "hashes": self.hashes,
        }
        if include_payloads:
            result["files"] = self.files
        else:
            result["file_names"] = sorted(self.files)
        return result


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_l1_directory(dataset_root: str | Path, domain: str = "digestive") -> Path:
    root = Path(dataset_root).expanduser().resolve()
    candidates: list[Path]
    normalized = domain.strip().lower()
    if root.name.lower() == "l1":
        candidates = [root]
    elif normalized in {"digestive", "digestion"}:
        candidates = [root / "l1"]
    else:
        aliases = {"cardiovascular": "cardio", "cardiovascular-sepsis": "cardio"}
        normalized = aliases.get(normalized, normalized)
        candidates = [root / "domains" / normalized / "l1", root / normalized / "l1"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"no L1 directory for domain {domain!r}; searched: {searched}")


def load_precomputed_l1(
    dataset_root: str | Path,
    *,
    domain: str = "digestive",
    names: Iterable[str] | None = None,
) -> LegacyBundle:
    """Load legacy JSON files without converting them to API predictions."""

    l1_dir = resolve_l1_directory(dataset_root, domain)
    requested = {name.removesuffix(".json") for name in names} if names else None
    files: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for path in sorted(l1_dir.glob("*.json")):
        if requested is not None and path.stem not in requested:
            continue
        with path.open("r", encoding="utf-8") as handle:
            files[path.stem] = json.load(handle)
        hashes[path.name] = _sha256(path)
    if not files:
        raise FileNotFoundError(f"no matching JSON artifacts in {l1_dir}")
    return LegacyBundle(l1_dir, domain, files, hashes)


def _bitmap_summary(value: Any, path: str, output: dict[str, dict[str, float]]) -> None:
    if isinstance(value, str) and value and set(value) <= {"0", "1"}:
        output[path] = {
            "n": float(len(value)),
            "ones": float(value.count("1")),
            "rate": value.count("1") / len(value),
        }
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _bitmap_summary(child, f"{path}.{key}" if path else str(key), output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _bitmap_summary(child, f"{path}[{index}]", output)


def summarize_precomputed_l1(bundle: LegacyBundle) -> dict[str, Any]:
    bitmaps: dict[str, dict[str, float]] = {}
    for name, payload in bundle.files.items():
        _bitmap_summary(payload, name, bitmaps)
    return {
        "status": "ok",
        "source_kind": PRECOMPUTED_L1_SOURCE,
        "scoreable_as_api_run": False,
        "warning": LEGACY_WARNING,
        "domain": bundle.domain,
        "input_hashes": bundle.hashes,
        "bitmap_summaries": bitmaps,
    }


def replay_l1(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    domain: str = "digestive",
    names: Iterable[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a provenance-marked compatibility summary of precomputed L1 data."""

    bundle = load_precomputed_l1(dataset_root, domain=domain, names=names)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "legacy_replay.json"
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}; pass overwrite=True")
    payload = summarize_precomputed_l1(bundle)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_run_manifest(run_dir_or_manifest: str | Path) -> dict[str, Any]:
    path = Path(run_dir_or_manifest).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    return manifest


def assert_api_run(run_dir_or_manifest: str | Path) -> dict[str, Any]:
    """Reject legacy replays at every API-run scoring/report boundary."""

    manifest = load_run_manifest(run_dir_or_manifest)
    source_kind = manifest.get("source_kind")
    if source_kind != API_RUN_SOURCE:
        raise ValueError(
            f"expected source_kind={API_RUN_SOURCE!r}, found {source_kind!r}; "
            "legacy artifacts cannot be scored as API runs"
        )
    return manifest


__all__ = [
    "API_RUN_SOURCE",
    "LEGACY_WARNING",
    "PRECOMPUTED_L1_SOURCE",
    "LegacyBundle",
    "assert_api_run",
    "load_precomputed_l1",
    "load_run_manifest",
    "replay_l1",
    "resolve_l1_directory",
    "summarize_precomputed_l1",
]
