"""Author and deterministically evaluate one live sandboxed model artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from lithops.agents.model_coding_agent import (
    ACQUISITION_MODEL_CODER,
    ModelCodingAgent,
)
from lithops.infrastructure.llm import OpenRouterProvider
from lithops.infrastructure.llm.gemini_adk_provider import GeminiAdkProvider
from lithops.model_runtime import (
    FixedBaselineModel,
    SandboxedCompanyModel,
    SandboxedPythonRunner,
    TemporalEvaluationPolicy,
    TemporalModelEvaluator,
    TemporalObservation,
)
from verify_model_builder import challenge_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, sandbox, and temporally evaluate one OpenRouter model artifact."
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--provider",
        choices=("openrouter", "gemini"),
        default="openrouter",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def evaluation_observations() -> tuple[TemporalObservation, ...]:
    return tuple(
        TemporalObservation(
            observation_id=f"coding-smoke:{day}",
            day=day,
            state={
                "cash": cash,
                "revenue_weekly": 40_000.0,
                "customers": customers,
                "churn_rate": 0.04,
                "weekly_acquisition": 25.0,
                "marketing_spend": 10_000.0,
            },
            action_from_previous={
                "marketing_spend": 10_000.0,
                "experiment_duration_weeks": 2,
            },
        )
        for day, cash, customers in (
            (0, 1_000_000.0, 500.0),
            (7, 975_000.0, 510.0),
            (14, 950_000.0, 520.0),
            (21, 925_000.0, 530.0),
        )
    )


async def verify(provider_name: str, model_name: str) -> dict[str, object]:
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Missing required environment variable: OPENROUTER_API_KEY")
        provider = OpenRouterProvider(
            api_key=api_key,
            model=model_name,
            timeout_seconds=180,
        )
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Missing required environment variable: GEMINI_API_KEY")
        provider = GeminiAdkProvider(
            api_key=api_key,
            model=model_name,
            agent_name="lithops_model_coder",
            agent_description="Authors one scoped executable company model.",
        )
    agent = ModelCodingAgent(
        spec=ACQUISITION_MODEL_CODER,
        provider=provider,
        provider_name=provider_name,
    )
    package = challenge_package()
    artifact = await agent.author(
        package=package,
        parent_artifact=FixedBaselineModel().artifact,
    )
    runner = SandboxedPythonRunner()
    sandboxed = SandboxedCompanyModel(artifact, runner)
    artifact_tests = runner.run_artifact_tests(artifact)
    prior = {
        name: value
        for name, value, _ in ACQUISITION_MODEL_CODER.available_priors
        if name in artifact.required_priors
    }
    evaluation = TemporalModelEvaluator(
        TemporalEvaluationPolicy(n_rollouts=5, runtime_budget_ms=10_000)
    ).evaluate(
        run_id=package.run_id,
        runtime=sandboxed,
        observations=evaluation_observations(),
        prior=prior,
        seed=44,
    )
    return {
        "provider": provider_name,
        "model": provider.model_id,
        "agent": ACQUISITION_MODEL_CODER.name,
        "prompt_version": ACQUISITION_MODEL_CODER.prompt_version,
        "artifact": artifact.model_dump(mode="json"),
        "artifact_tests": [
            {
                "name": item.name,
                "passed": item.passed,
                "failure_reason": item.failure_reason,
            }
            for item in artifact_tests
        ],
        "temporal_evaluation": {
            "passed": evaluation.passed,
            "failure_codes": list(evaluation.failure_codes),
            "fold_count": len(evaluation.folds),
            "mean_total_score": (
                evaluation.mean_total_score if evaluation.folds else None
            ),
            "fitted_model_ids": [str(item.id) for item in evaluation.fitted_models],
        },
        "ceobench_tools": [],
        "production_secrets_exposed": [],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    args = parse_args()
    if args.provider == "gemini":
        model_name = args.model or os.getenv("GEMINI_MODEL") or "gemini-3.7-flash"
    else:
        model_name = args.model or os.getenv("OPENROUTER_MODEL") or "qwen/qwen3-32b"
    result = asyncio.run(verify(args.provider, model_name))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else project_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "model": result["model"],
                "artifact_id": result["artifact"]["id"],
                "artifact_hash": result["artifact"]["content_hash"],
                "artifact_tests_passed": all(
                    item["passed"] for item in result["artifact_tests"]
                ),
                "temporal_evaluation": result["temporal_evaluation"],
                "output": str(args.output) if args.output is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
