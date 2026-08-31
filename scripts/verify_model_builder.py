"""Run one live, actionless candidate-builder structured proposal."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from lithops.agents.candidate_model_builder import (
    ACQUISITION_BUILDER,
    CandidateModelBuilder,
)
from lithops.domain.evaluation import (
    HorizonPerformance,
    ModelHealthSignal,
    ModelHealthStatus,
)
from lithops.domain.model_challenge import (
    ChallengeMetric,
    ChallengeObservation,
    ChallengeParameterSensitivity,
    ChallengeResidual,
    ModelChallengePackage,
)
from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import WorldModelParameterName
from lithops.infrastructure.llm import OpenRouterProvider
from lithops.infrastructure.llm.gemini_adk_provider import GeminiAdkProvider
from lithops.world_model import bootstrap_world_model


def challenge_package() -> ModelChallengePackage:
    """Small deterministic development package; never primary benchmark evidence."""

    created_at = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
    run_id = UUID(int=1)
    model = bootstrap_world_model(
        run_id,
        ObservationSnapshot(
            day=0,
            cash=1_000_000,
            metrics={"marketing_spend": 10_000, "acquisition": 80},
            observed_at=created_at,
        ),
    )
    outcome_ids = (UUID(int=101), UUID(int=102), UUID(int=103))
    health = ModelHealthSignal(
        id=UUID(int=10),
        run_id=run_id,
        model_version_id=model.id,
        evaluated_day=21,
        status=ModelHealthStatus.DEGRADED,
        outcome_ids=outcome_ids,
        horizon_performance=(
            HorizonPerformance(
                horizon_days=7,
                outcome_count=3,
                mean_normalized_absolute_error=0.6,
                interval_coverage=0.0,
                mean_weighted_interval_score=500_000,
                signed_bias=-0.5,
            ),
        ),
        interval_miss_count=3,
        directional_bias=-0.5,
        rebuild_recommended=True,
        trigger_codes=("persistent_directional_bias",),
        evaluated_at=created_at,
    )
    actual_cash = (800_000.0, 650_000.0, 500_000.0)
    residuals = tuple(
        ChallengeResidual(
            outcome_id=outcome_id,
            prediction_id=UUID(int=200 + index),
            target_id=UUID(int=300 + index),
            issued_day=7 * (index - 1),
            horizon_days=7,
            target_day=7 * index,
            observed_day=7 * index,
            predicted_cash=1_000_000,
            lower_cash=900_000,
            upper_cash=1_100_000,
            actual_cash=cash,
            signed_error=cash - 1_000_000,
            normalized_absolute_error=abs(cash - 1_000_000) / cash,
            interval_hit=False,
            parameter_sensitivities=(
                ChallengeParameterSensitivity(
                    parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                    cash_sensitivity_per_unit=4_000_000,
                    evidence_reference=f"finite-difference:{outcome_id}:marketing_saturation",
                ),
            ),
        )
        for index, (outcome_id, cash) in enumerate(
            zip(outcome_ids, actual_cash, strict=True),
            start=1,
        )
    )
    return ModelChallengePackage(
        challenge_id=UUID(int=2),
        run_id=run_id,
        health_signal=health,
        active_model=model,
        observations=(
            ChallengeObservation(
                reference=f"observation:{run_id}:21",
                day=21,
                cash=500_000,
                metrics=(ChallengeMetric(name="weekly_acquisition", value=20),),
                observed_at=created_at,
            ),
        ),
        residuals=residuals,
        created_at=created_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one bounded model hypothesis with no CEO-Bench tools."
    )
    parser.add_argument("provider", choices=("openrouter", "gemini"))
    parser.add_argument("--model", default=None)
    return parser.parse_args()


async def verify(provider_name: str, model: str) -> dict[str, object]:
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Missing required environment variable: OPENROUTER_API_KEY")
        provider = OpenRouterProvider(api_key=api_key, model=model, timeout_seconds=120)
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "Missing required environment variable: GEMINI_API_KEY or GOOGLE_API_KEY"
            )
        provider = GeminiAdkProvider(
            api_key=api_key,
            model=model,
            agent_name=ACQUISITION_BUILDER.name,
            agent_description="Challenges acquisition-efficiency assumptions only.",
        )
    builder = CandidateModelBuilder(
        spec=ACQUISITION_BUILDER,
        provider=provider,
        provider_name=provider_name,
    )
    proposal = await builder.propose(challenge_package())
    return {
        "provider": provider_name,
        "model": provider.model_id,
        "builder_name": proposal.builder_name,
        "builder_version": proposal.builder_version,
        "prompt_version": proposal.prompt_version,
        "family": proposal.family.value,
        "parameter_adjustments": [
            {
                "parameter": item.parameter_name.value,
                "direction": item.direction.value,
                "step_size": item.step_size.value,
            }
            for item in proposal.diff.parameter_adjustments
        ],
        "relationship_activations": [
            item.relationship_key.value for item in proposal.diff.relationship_activations
        ],
        "evidence_kinds": sorted(item.kind.value for item in proposal.evidence),
        "ceobench_tools": [],
    }


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    args = parse_args()
    default_model = (
        "openrouter/free"
        if args.provider == "openrouter"
        else os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    )
    print(json.dumps(asyncio.run(verify(args.provider, args.model or default_model)), indent=2))


if __name__ == "__main__":
    main()
