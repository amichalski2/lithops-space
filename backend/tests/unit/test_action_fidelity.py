from lithops.domain.evidence import ConfigurationEvidence, WeeklyEvidencePacket
from lithops.domain.models import ActionCommand, ObservationSnapshot
from lithops.evaluation.action_fidelity import action_fidelity_violations


def snapshot(*, price_a: float = 20.0) -> ObservationSnapshot:
    return ObservationSnapshot(
        day=7,
        cash=990_000,
        evidence=WeeklyEvidencePacket(
            day=7,
            window_start_day_exclusive=0,
            window_end_day_inclusive=7,
            configuration=ConfigurationEvidence(
                prices={"A": price_a, "B": 69.0, "C": 179.0},
                model_tiers={"A": 5, "B": 2, "C": 3},
                daily_channel_spend={"search_ads": 100.0},
                daily_operations_spend=500.0,
                daily_development_spend=250.0,
                capacity_tier=0,
                targeted_ads_json='{"targeted_spend":{"search_ads":{"S1":100.0}}}',
                targeted_development_json='{"S1":200.0}',
                lead_promotion_json='{"global":5.0}',
            ),
        ),
    )


def commands() -> list[ActionCommand]:
    values = [
        ("set_prices", {"A": 20.0, "B": 69.0, "C": 179.0}),
        ("set_model_tiers", {"A": 5, "B": 2, "C": 3}),
        ("set_daily_spend", {"operations": 500.0, "development": 250.0}),
        (
            "set_targeted_ad_spend",
            {"targeted_spend": {"search_ads": {"S1": 100.0}}},
        ),
        ("set_targeted_dev_spend", {"targeted_spend": {"S1": 200.0}}),
        ("set_lead_promotion", {"global_promotion": 5.0}),
    ]
    return [
        ActionCommand(tool=tool, arguments=arguments, idempotency_key=f"test-{index}")
        for index, (tool, arguments) in enumerate(values)
    ]


def test_matching_public_configuration_proves_action_fidelity() -> None:
    assert action_fidelity_violations(commands(), snapshot()) == ()


def test_unchanged_price_is_reported_as_execution_mismatch() -> None:
    assert action_fidelity_violations(commands(), snapshot(price_a=25.0)) == (
        "set_prices.A",
    )

