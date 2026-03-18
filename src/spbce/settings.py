from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_CANDIDATES = [
    REPO_ROOT / "api.env",
    REPO_ROOT / "api,env",
    REPO_ROOT / "minimax_api.env",
]
RAW_ENV_PREFIX_MAP = {
    "minimax": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gpt": "OPENAI_API_KEY",
}
PROVIDER_ENV_KEYS = {
    "anthropic": {"ANTHROPIC_API_KEY"},
    "deepseek": {"DEEPSEEK_API_KEY"},
    "openai": {"OPENAI_API_KEY"},
}


def _parse_env_text(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    lines = text.splitlines()
    standard_entries = [line for line in lines if "=" in line]
    if standard_entries:
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", maxsplit=1)
            parsed[key.strip()] = value.strip().strip("'\"")
        return parsed

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_\-]+)\s*[:,]\s*(.+)$", line)
        if match:
            prefix = match.group(1).strip().lower()
            value = match.group(2).strip()
            env_key = RAW_ENV_PREFIX_MAP.get(prefix)
            if env_key and value:
                parsed[env_key] = value
                continue
        if line.startswith("sk-api-"):
            parsed.setdefault("ANTHROPIC_API_KEY", line)
    return parsed


def _provider_count(env_map: dict[str, str]) -> int:
    return sum(1 for keys in PROVIDER_ENV_KEYS.values() if any(key in env_map for key in keys))


def resolve_runtime_env_file(env_file: str | None = None) -> Path | None:
    candidate_paths: list[Path] = []
    if env_file:
        candidate_paths.append(Path(env_file))
    env_override = os.getenv("SPBCE_ENV_FILE")
    if env_override:
        candidate_paths.append(Path(env_override))
    candidate_paths.extend(ENV_FILE_CANDIDATES)

    best_path: Path | None = None
    best_score = -1
    seen: set[Path] = set()
    for path in candidate_paths:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        env_map = _parse_env_text(resolved.read_text(encoding="utf-8"))
        score = _provider_count(env_map)
        if score > best_score:
            best_score = score
            best_path = resolved
    return best_path


def get_provider_environment_summary(env_file: str | None = None) -> dict[str, Any]:
    path = resolve_runtime_env_file(env_file)
    if path is None:
        return {"path": None, "providers": []}
    env_map = _parse_env_text(path.read_text(encoding="utf-8"))
    providers = [
        provider
        for provider, keys in PROVIDER_ENV_KEYS.items()
        if any(key in env_map and env_map[key] for key in keys)
    ]
    return {"path": str(path), "providers": providers}


def initialize_runtime_env(env_file: str | None = None) -> None:
    path = resolve_runtime_env_file(env_file)
    if path is None:
        return
    env_map = _parse_env_text(path.read_text(encoding="utf-8"))
    for key, value in env_map.items():
        os.environ.setdefault(key, value)

    if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = DEFAULT_ANTHROPIC_BASE_URL
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("DEEPSEEK_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = DEFAULT_DEEPSEEK_BASE_URL
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = DEFAULT_OPENAI_BASE_URL


initialize_runtime_env()


@dataclass(slots=True)
class Settings:
    model_artifact: str = os.getenv(
        "SPBCE_MODEL_ARTIFACT", "data/processed/artifacts/demo_survey_model.joblib"
    )
    prompt_artifact: str = os.getenv(
        "SPBCE_BASELINE_ARTIFACT", "data/processed/artifacts/demo_prompt_baseline.joblib"
    )
    ood_artifact: str = os.getenv("SPBCE_OOD_ARTIFACT", "data/processed/artifacts/demo_ood.joblib")
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL)
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    default_llm_model: str = os.getenv("SPBCE_LLM_MODEL", "google/flan-t5-small")
    default_llm_provider: str = os.getenv("SPBCE_LLM_PROVIDER", "auto")


settings = Settings()
