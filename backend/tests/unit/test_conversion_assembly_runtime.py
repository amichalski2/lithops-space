from datetime import UTC, datetime
from uuid import UUID

import pytest
from lithops.agents.component_architect import (
    THRESHOLD_CONVERSION_ARCHITECT,
    ConversionComponentAuthor,
)
from lithops.application.executable_model_challenge import ExecutableModelChallenge
from lithops.domain.component_program import (
    ConversionComponentProgram,
    ConversionFeature,
    ConversionLink,
)
from lithops.domain.executable_model import (
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    ModelRuntimeKind,
)
from lithops.domain.models import ObservationSnapshot
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository
from lithops.model_runtime import (
    ConversionAssemblyModel,
    FixedBaselineModel,
    TemporalEvaluationPolicy,
    TemporalModelEvaluator,
    TemporalObservation,
)
from lithops.world_model import bootstrap_world_model

from backend.tests.unit.test_hypothesis_backtest import challenge_package

RUN_ID = UUID("77777777-7777-7777-7777-777777777777")


class ComponentProvider:
    model_id = "gemini-test"

    def __init__(self) -> None:
        self.payload = None

    async def generate_structured(self, *, system_prompt, user_prompt, output_schema):
        assert "not company actions and not\nPython code" in system_prompt
        self.payload = user_prompt
        return output_schema.model_validate(
            {
                "name": "quality_threshold_conversion",
                "scope": "conversion",
                "link": "threshold_logistic",
                "features": ["product_quality", "net_entry_price_monthly"],
                "threshold_feature": "product_quality",
                "rationale": "Conversion may change after a quality regime transition.",
                "falsifiers": [
                    "Exposed high-quality cohorts retain the same conversion distribution."
                ],
            }
        )


def _history() -> tuple[dict, ...]:
    return (
        {
            "day": 7,
            "cash": 99_000,
            "revenue_weekly": 0,
            "customers": 0,
            "weekly_leads": 200,
            "weekly_conversions": 0,
            "product_quality": 0.2,
            "entry_price_monthly": 25,
            "lead_promotion_monthly": 0,
        },
        {
            "day": 14,
            "cash": 98_000,
            "revenue_weekly": 0,
            "customers": 0,
            "weekly_leads": 100,
            "weekly_conversions": 20,
            "product_quality": 0.7,
            "entry_price_monthly": 25,
            "lead_promotion_monthly": 0,
        },
    )


@pytest.mark.asyncio
async def test_gemini_component_author_produces_typed_artifact_not_python() -> None:
    package = challenge_package()
    package = package.model_copy(
        update={
            "health_signal": package.health_signal.model_copy(
                update={"trigger_codes": ("persistent_zero_conversion_funnel",)}
            )
        }
    )
    provider = ComponentProvider()
    author = ConversionComponentAuthor(
        spec=THRESHOLD_CONVERSION_ARCHITECT,
        provider=provider,
        provider_name="gemini",
    )

    artifact = await author.author(
        package=package,
        parent_artifact=FixedBaselineModel().artifact,
    )

    assert author.supports(package)
    assert artifact.runtime_kind is ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY
    assert artifact.source_code is None
    assert artifact.component_program is not None
    assert artifact.component_program.link is ConversionLink.THRESHOLD_LOGISTIC
    assert {item.name for item in artifact.required_features} >= {
        "history.weekly_leads",
        "history.weekly_conversions",
        "history.product_quality",
    }


def test_conversion_assembly_fits_component_and_emits_reconciled_distribution() -> None:
    program = ConversionComponentProgram(
        name="quality_threshold_conversion",
        link=ConversionLink.THRESHOLD_LOGISTIC,
        features=(
            ConversionFeature.PRODUCT_QUALITY,
            ConversionFeature.NET_ENTRY_PRICE_MONTHLY,
        ),
        threshold_feature=ConversionFeature.PRODUCT_QUALITY,
        rationale="Conversion may change after a quality regime transition.",
        falsifiers=("High quality remains non-converting.",),
    )
    artifact = FixedBaselineModel().artifact.create(
        name="quality-threshold-conversion",
        protocol_version="2.0",
        runtime_kind=ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY,
        scope="conversion",
        hypothesis=program.rationale,
        authoring_agent="typed-test",
        provider="deterministic-test",
        model_name="typed-compiler",
        prompt_version="v1",
        component_program=program,
        limitations=program.falsifiers,
        parent_artifact_id=FixedBaselineModel().artifact.id,
    )
    runtime = ConversionAssemblyModel(artifact)
    world_model = bootstrap_world_model(
        RUN_ID,
        ObservationSnapshot(
            day=0,
            cash=100_000,
            observed_at=datetime(2026, 8, 27, tzinfo=UTC),
        ),
    )
    history = _history()
    fitted = runtime.fit(
        CompanyModelFitRequest(
            observation_ids=("low", "high"),
            training_start_day=7,
            training_end_day=14,
            history=history,
            prior={"legacy_world_model": world_model.model_dump(mode="json")},
            seed=1,
        )
    )
    state = {
        "week": 2,
        "cash": 98_000,
        "revenue_weekly": 0,
        "customers": 0,
        "churn_rate": 0.04,
        "price_per_customer_weekly": 5.833,
        "catalog_price_per_customer_weekly": 21.233,
        "entry_price_monthly": 25,
        "weekly_acquisition": 0,
        "weekly_leads": 100,
        "weekly_conversions": 20,
        "weekly_lost_leads": 80,
        "total_leads": 300,
        "total_conversions": 20,
        "total_lost_leads": 280,
        "marketing_spend": 700,
        "development_spend": 0,
        "operations_spend": 100,
        "product_quality": 0.7,
        "capacity": 1_000,
        "reputation": 0.5,
        "operating_cost_per_customer_weekly": 2,
    }
    action = {
        "name": "hold",
        "price_per_customer_weekly": 21.233,
        "marketing_spend": 700,
        "development_spend": 0,
        "operations_spend": 100,
    }

    distribution = runtime.predict(
        CompanyModelPredictRequest(
            fitted_model=fitted,
            state=state,
            action=action,
            horizons_days=(7,),
            n_rollouts=3,
            seed=7,
        )
    )

    assert len(distribution.samples) == 3
    assert all(sample.customers > 0 for sample in distribution.samples)
    assert all(
        sample.cash == pytest.approx(sample.accounting.ending_cash)
        for sample in distribution.samples
    )
    assert runtime.diagnostics(fitted)["component_scope"] == "conversion"


@pytest.mark.asyncio
async def test_typed_component_runs_through_real_challenge_factory() -> None:
    package = challenge_package()
    package = package.model_copy(
        update={
            "health_signal": package.health_signal.model_copy(
                update={"trigger_codes": ("persistent_zero_conversion_funnel",)}
            )
        }
    )
    author = ConversionComponentAuthor(
        spec=THRESHOLD_CONVERSION_ARCHITECT,
        provider=ComponentProvider(),
        provider_name="gemini",
    )
    repository = InMemoryRunRepository()
    service = ExecutableModelChallenge(
        repository=repository,
        authors=(author,),
        evaluator=TemporalModelEvaluator(
            TemporalEvaluationPolicy(n_rollouts=5, runtime_budget_ms=10_000)
        ),
    )

    def state(day: int, cash: float, quality: float, conversions: float) -> dict:
        return {
            "week": day // 7,
            "cash": cash,
            "revenue_weekly": conversions * 5.833,
            "customers": conversions,
            "churn_rate": 0.04,
            "price_per_customer_weekly": 5.833,
            "catalog_price_per_customer_weekly": 21.233,
            "entry_price_monthly": 25,
            "weekly_acquisition": conversions,
            "weekly_leads": 100,
            "weekly_conversions": conversions,
            "weekly_lost_leads": 100 - conversions,
            "total_leads": 100 * (day // 7 + 1),
            "total_conversions": conversions,
            "total_lost_leads": 100 * (day // 7 + 1) - conversions,
            "marketing_spend": 700,
            "development_spend": 0,
            "operations_spend": 100,
            "product_quality": quality,
            "capacity": 1_000,
            "reputation": 0.5,
            "operating_cost_per_customer_weekly": 2,
        }

    action = {
        "name": "hold",
        "price_per_customer_weekly": 21.233,
        "marketing_spend": 700,
        "development_spend": 0,
        "operations_spend": 100,
    }
    temporal = tuple(
        TemporalObservation(
            observation_id=f"obs-{day}",
            day=day,
            state=state(day, cash, quality, conversions),
            action_from_previous=action,
        )
        for day, cash, quality, conversions in (
            (0, 100_000, 0.2, 0),
            (7, 99_000, 0.2, 0),
            (14, 98_000, 0.7, 20),
            (21, 97_000, 0.75, 25),
        )
    )

    result = await service.run(
        package=package,
        observations=temporal,
        world_model=package.active_model,
        seed=11,
    )

    assert result.promotion is not None
    assert result.candidate_evaluation is not None
    assert result.candidate_evaluation.artifact.runtime_kind is (
        ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY
    )
    assert len(result.candidate_evaluation.folds) == 1
    assert result.candidate_evaluation.failure_codes == ()
