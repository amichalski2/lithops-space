from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    storage_backend: str = "memory"
    benchmark_backend: str = "fake"
    supabase_url: str | None = None
    supabase_secret_key: str | None = None
    ceobench_executable: str | None = None
    ceobench_python: str | None = None
    ceobench_seed: int = 42
    model_provider: str = "static"
    openrouter_api_key: str | None = None
    openrouter_model: str = "qwen/qwen3-32b"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.7-flash"
    executable_model_planning: bool = False
    executive_authority_v2: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(override=False)
        return cls(
            storage_backend=os.getenv("LITHOPS_STORAGE_BACKEND", "memory").lower(),
            benchmark_backend=os.getenv("LITHOPS_BENCHMARK_BACKEND", "fake").lower(),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_secret_key=os.getenv("SUPABASE_SECRET_KEY"),
            ceobench_executable=os.getenv("CEOBENCH_EXECUTABLE"),
            ceobench_python=os.getenv("CEOBENCH_PYTHON"),
            ceobench_seed=int(os.getenv("CEOBENCH_SEED", "42")),
            model_provider=os.getenv("LITHOPS_MODEL_PROVIDER", "static").lower(),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_MODEL") or "qwen/qwen3-32b",
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
            executable_model_planning=os.getenv(
                "LITHOPS_EXECUTABLE_MODEL_PLANNING", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            executive_authority_v2=os.getenv(
                "LITHOPS_EXECUTIVE_AUTHORITY_V2", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        if self.storage_backend not in {"memory", "supabase"}:
            raise ValueError("LITHOPS_STORAGE_BACKEND must be memory or supabase")
        if self.benchmark_backend not in {"fake", "ceobench"}:
            raise ValueError("LITHOPS_BENCHMARK_BACKEND must be fake or ceobench")
        if self.model_provider not in {"static", "openrouter", "gemini"}:
            raise ValueError(
                "LITHOPS_MODEL_PROVIDER must be static, openrouter, or gemini"
            )
        if self.storage_backend == "supabase" and not (
            self.supabase_url and self.supabase_secret_key
        ):
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY are required for Supabase storage"
            )
        if self.benchmark_backend == "ceobench" and not self.ceobench_executable:
            raise ValueError(
                "CEOBENCH_EXECUTABLE is required for the CEO-Bench backend"
            )
        if self.model_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter provider")
        if self.model_provider == "openrouter" and not self.openrouter_model.strip():
            raise ValueError("OPENROUTER_MODEL must not be empty")
        if self.model_provider == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini provider"
            )
        if self.executive_authority_v2 and not self.executable_model_planning:
            raise ValueError(
                "LITHOPS_EXECUTIVE_AUTHORITY_V2 requires "
                "LITHOPS_EXECUTABLE_MODEL_PLANNING; the legacy planner does not "
                "implement the two-stage executive flow"
            )
