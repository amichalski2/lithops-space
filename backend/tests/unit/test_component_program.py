import pytest
from lithops.domain.component_program import (
    ConversionComponentProgram,
    ConversionEvidence,
    ConversionFeature,
    ConversionLink,
)
from lithops.model_runtime.component_program import (
    CompiledConversionComponent,
    fit_conversion_program,
)
from lithops.simulator.components import (
    BASELINE_TRANSITION_ASSEMBLY,
    TransitionModelAssembly,
)
from lithops.simulator.models import SimulationAction, SimulationState
from lithops.simulator.state_transition import advance_simulation_week
from pydantic import ValidationError


def _program(link: ConversionLink) -> ConversionComponentProgram:
    return ConversionComponentProgram(
        name=f"quality_{link.value}",
        link=link,
        features=(ConversionFeature.PRODUCT_QUALITY,),
        threshold_feature=(
            ConversionFeature.PRODUCT_QUALITY
            if link is ConversionLink.THRESHOLD_LOGISTIC
            else None
        ),
        rationale="Compare smooth and gated quality-to-conversion structures.",
        falsifiers=("High-quality exposed cohorts continue to produce no conversions.",),
    )


def _evidence() -> tuple[ConversionEvidence, ...]:
    return (
        ConversionEvidence(
            observation_id="low-1",
            day=7,
            leads=300,
            conversions=0,
            features={ConversionFeature.PRODUCT_QUALITY: 0.2},
        ),
        ConversionEvidence(
            observation_id="low-2",
            day=14,
            leads=200,
            conversions=0,
            features={ConversionFeature.PRODUCT_QUALITY: 0.35},
        ),
        ConversionEvidence(
            observation_id="high-1",
            day=21,
            leads=100,
            conversions=20,
            features={ConversionFeature.PRODUCT_QUALITY: 0.65},
        ),
        ConversionEvidence(
            observation_id="high-2",
            day=28,
            leads=100,
            conversions=30,
            features={ConversionFeature.PRODUCT_QUALITY: 0.8},
        ),
    )


def test_program_structure_is_generic_and_fitted_threshold_comes_from_evidence() -> None:
    fitted = fit_conversion_program(
        _program(ConversionLink.THRESHOLD_LOGISTIC),
        _evidence(),
    )
    component = CompiledConversionComponent(fitted)

    assert 0.35 < fitted.threshold < 0.8
    low_probability = component.probability(
        {ConversionFeature.PRODUCT_QUALITY: 0.25}
    )
    high_probability = component.probability(
        {ConversionFeature.PRODUCT_QUALITY: 0.75}
    )
    assert high_probability > low_probability
    assert high_probability - low_probability > 0.10
    assert fitted.observation_ids == ("low-1", "low-2", "high-1", "high-2")


def test_compiled_conversion_changes_funnel_but_trusted_shell_reconciles_cash() -> None:
    component = CompiledConversionComponent(
        fit_conversion_program(_program(ConversionLink.LOGISTIC), _evidence())
    )
    assembly = TransitionModelAssembly(
        quality=BASELINE_TRANSITION_ASSEMBLY.quality,
        lead_arrival=BASELINE_TRANSITION_ASSEMBLY.lead_arrival,
        conversion=component,
    )
    current = SimulationState(
        week=10,
        cash=100_000,
        revenue_weekly=0,
        customers=0,
        churn_rate=0.04,
        price_per_customer_weekly=10,
        entry_price_monthly=25,
        weekly_acquisition=0,
        weekly_leads=100,
        total_leads=1_000,
        marketing_spend=700,
        development_spend=0,
        product_quality=0.75,
        capacity=1_000,
        reputation=0.5,
        operating_cost_per_customer_weekly=2,
    )
    action = SimulationAction(
        name="hold",
        price_per_customer_weekly=10,
        marketing_spend=700,
        development_spend=0,
    )
    parameters = {
        "price_elasticity": 0.2,
        "marketing_saturation": 0.8,
        "churn_sensitivity": 0.2,
        "quality_lag_weeks": 4,
        "segment_response": 0.1,
    }

    result = advance_simulation_week(
        current,
        action,
        parameters,
        assembly=assembly,
    )

    assert result.weekly_conversions > 0
    assert result.weekly_conversions <= result.weekly_leads
    expected_cash = (
        current.cash
        + result.revenue_weekly
        - result.customers * result.operating_cost_per_customer_weekly
        - result.operations_spend
        - result.capacity_spend_weekly
        - result.marketing_spend
        - result.development_spend
    )
    assert result.cash == pytest.approx(expected_cash)


def test_threshold_program_rejects_a_threshold_not_present_in_its_feature_set() -> None:
    with pytest.raises(ValidationError, match="threshold feature must be"):
        ConversionComponentProgram(
            name="bad_threshold",
            link=ConversionLink.THRESHOLD_LOGISTIC,
            features=(ConversionFeature.REPUTATION,),
            threshold_feature=ConversionFeature.PRODUCT_QUALITY,
            rationale="Invalid cross-edge declaration.",
            falsifiers=("Any evidence.",),
        )
