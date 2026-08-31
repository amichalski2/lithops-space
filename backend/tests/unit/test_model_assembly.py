from uuid import UUID

import pytest
from lithops.domain.model_assembly import (
    ComponentAssignment,
    ModelComponentScope,
    WorldModelAssembly,
)
from lithops.simulator.components import (
    BASELINE_TRANSITION_ASSEMBLY,
    FunnelTransition,
    TransitionModelAssembly,
)
from lithops.simulator.models import SimulationAction, SimulationState
from lithops.simulator.state_transition import advance_simulation_week


class SevenCustomerConversion:
    def transition(self, **_: object) -> FunnelTransition:
        return FunnelTransition(
            acquired_customers=7,
            predicted_leads=20,
            lost_leads=13,
        )


def _state() -> SimulationState:
    return SimulationState(
        cash=100_000,
        revenue_weekly=0,
        customers=0,
        churn_rate=0.04,
        price_per_customer_weekly=10,
        weekly_acquisition=0,
        marketing_spend=700,
        development_spend=350,
        operations_spend=100,
        product_quality=0.4,
        capacity=100,
        reputation=0.5,
        operating_cost_per_customer_weekly=2,
    )


def _action() -> SimulationAction:
    return SimulationAction(
        name="conversion_probe",
        price_per_customer_weekly=10,
        marketing_spend=700,
        development_spend=350,
        operations_spend=100,
    )


def _parameters() -> dict:
    return {
        "price_elasticity": 0.2,
        "marketing_saturation": 0.8,
        "churn_sensitivity": 0.2,
        "quality_lag_weeks": 4,
        "segment_response": 0.1,
    }


def test_conversion_component_cannot_write_cash_or_accounting_costs() -> None:
    assembly = TransitionModelAssembly(
        quality=BASELINE_TRANSITION_ASSEMBLY.quality,
        lead_arrival=BASELINE_TRANSITION_ASSEMBLY.lead_arrival,
        conversion=SevenCustomerConversion(),
    )

    result = advance_simulation_week(
        _state(),
        _action(),
        _parameters(),
        assembly=assembly,
    )

    assert result.weekly_conversions == 7
    assert result.weekly_leads == 20
    assert result.customers == 7
    # The trusted shell, not the component, owns revenue, costs, spend, and cash.
    assert result.revenue_weekly == 70
    assert result.cash == pytest.approx(100_000 + 70 - 14 - 100 - 700 - 350)


def test_world_model_assembly_identity_is_order_independent_and_tamper_evident() -> None:
    conversion = ComponentAssignment(
        scope=ModelComponentScope.CONVERSION,
        artifact_id=UUID(int=1),
        artifact_hash="1" * 64,
        fitted_model_id=UUID(int=2),
        fitted_state_hash="2" * 64,
    )
    quality = ComponentAssignment(
        scope=ModelComponentScope.QUALITY_DYNAMICS,
        artifact_id=UUID(int=3),
        artifact_hash="3" * 64,
        fitted_model_id=UUID(int=4),
        fitted_state_hash="4" * 64,
    )

    first = WorldModelAssembly.create(components=(quality, conversion))
    second = WorldModelAssembly.create(components=(conversion, quality))

    assert first == second
    assert first.components[0].scope is ModelComponentScope.CONVERSION
    with pytest.raises(ValueError, match="duplicate"):
        WorldModelAssembly.create(components=(conversion, conversion))
