from datetime import UTC, datetime
from uuid import UUID

from lithops.domain.models import ObservationSnapshot
from lithops.domain.world_model import EvidenceKind, WorldModelParameterName
from lithops.world_model.bootstrap import P0_PRIORS, bootstrap_world_model

RUN_ID = UUID("11111111-1111-1111-1111-111111111111")
OBSERVED_AT = datetime(2026, 8, 25, tzinfo=UTC)


def observation(**metrics: float) -> ObservationSnapshot:
    return ObservationSnapshot(
        day=0,
        cash=1_000_000,
        metrics=metrics,
        observed_at=OBSERVED_AT,
    )


def test_bootstrap_is_deterministic_and_contains_the_fixed_p0_framework() -> None:
    snapshot = observation(
        pricing=99,
        conversion=0.12,
        churn=0.04,
        marketing_spend=10_000,
        acquisition=150,
        development_spend=8_000,
        product_quality=0.7,
        segment_count=3,
        segment_conversion=0.15,
    )

    first = bootstrap_world_model(RUN_ID, snapshot)
    second = bootstrap_world_model(RUN_ID, snapshot)

    assert first == second
    assert {parameter.name for parameter in first.parameters} == {
        prior.name for prior in P0_PRIORS
    }
    assert {relationship.key for relationship in first.relationships} == {
        "price_to_conversion",
        "price_to_churn",
        "marketing_spend_to_acquisition",
        "development_spend_to_quality",
        "quality_to_churn",
        "segment_to_conversion",
    }


def test_sparse_bootstrap_keeps_unknown_dynamics_low_confidence() -> None:
    model = bootstrap_world_model(RUN_ID, observation())

    assert all(parameter.confidence <= 0.25 for parameter in model.parameters)
    assert all(
        {evidence.kind for evidence in parameter.evidence} == {EvidenceKind.GENERIC_PRIOR}
        for parameter in model.parameters
    )


def test_observed_signals_add_evidence_without_claiming_causal_certainty() -> None:
    model = bootstrap_world_model(
        RUN_ID,
        observation(pricing=99, conversion=0.12, churn=0.04),
    )
    price = next(
        parameter
        for parameter in model.parameters
        if parameter.name is WorldModelParameterName.PRICE_ELASTICITY
    )

    assert price.confidence == 0.35
    assert {evidence.kind for evidence in price.evidence} == {
        EvidenceKind.GENERIC_PRIOR,
        EvidenceKind.OBSERVATION,
    }
    assert price.upper_bound - price.lower_bound == 1.6
