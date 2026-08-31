"""Creation and idempotent maturation of cash prediction ledger entries."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from lithops.domain.errors import ConflictError
from lithops.domain.evaluation import CashSensitivityEstimate
from lithops.domain.models import CashForecasts, DecisionRecord, ObservationSnapshot
from lithops.domain.predictions import (
    CashPredictionTarget,
    PredictionActual,
    PredictionLedgerEntry,
    PredictionOutcome,
    PredictionOutcomeAttribution,
)
from lithops.evaluation.forecast_scoring import score_cash_prediction
from lithops.evaluation.maturity import is_exactly_maturing


def create_cash_prediction(
    *,
    run_id: UUID,
    decision_id: UUID,
    decision_week: int,
    issued_day: int,
    model_version_id: UUID,
    model_artifact_id: UUID | None = None,
    model_artifact_hash: str | None = None,
    fitted_model_id: UUID | None = None,
    fitted_state_hash: str | None = None,
    prompt_version: str,
    observation_reference: str,
    assumptions: tuple[str, ...],
    evidence_references: tuple[str, ...],
    uncertainty_source: str,
    confidence: float,
    forecasts: CashForecasts,
    committed_at: datetime,
    cash_sensitivities: tuple[CashSensitivityEstimate, ...] = (),
) -> PredictionLedgerEntry:
    entry_id = uuid5(NAMESPACE_URL, f"lithops:{run_id}:{decision_id}:cash-prediction")
    targets = tuple(
        CashPredictionTarget(
            id=uuid5(entry_id, str(forecast.horizon_days)),
            horizon_days=forecast.horizon_days,
            target_day=issued_day + forecast.horizon_days,
            point=forecast.point,
            lower=forecast.lower,
            upper=forecast.upper,
        )
        for forecast in forecasts.ordered()
    )
    return PredictionLedgerEntry(
        id=entry_id,
        run_id=run_id,
        decision_id=decision_id,
        decision_week=decision_week,
        issued_day=issued_day,
        model_version_id=model_version_id,
        model_artifact_id=model_artifact_id,
        model_artifact_hash=model_artifact_hash,
        fitted_model_id=fitted_model_id,
        fitted_state_hash=fitted_state_hash,
        prompt_version=prompt_version,
        observation_reference=observation_reference,
        assumptions=assumptions,
        evidence_references=evidence_references,
        uncertainty_source=uncertainty_source,
        confidence=confidence,
        cash_sensitivities=cash_sensitivities,
        targets=targets,
        committed_at=committed_at,
    )


def mature_cash_predictions(
    entries: Iterable[PredictionLedgerEntry],
    observation: ObservationSnapshot,
    *,
    observation_reference: str,
    existing_outcomes: Iterable[PredictionOutcome] = (),
) -> tuple[PredictionOutcome, ...]:
    """Return only new outcomes; exact replays return an empty tuple."""

    outcomes_by_target = {outcome.target_id: outcome for outcome in existing_outcomes}
    new_outcomes: list[PredictionOutcome] = []
    for entry in entries:
        for target in entry.targets:
            if not is_exactly_maturing(target, observation.day):
                continue
            actual = PredictionActual(
                target_id=target.id,
                observed_day=observation.day,
                cash=observation.cash,
                observation_reference=observation_reference,
                observed_at=observation.observed_at,
            )
            score = score_cash_prediction(target, actual)
            candidate = PredictionOutcome(
                id=uuid5(target.id, f"actual:{observation.day}"),
                run_id=entry.run_id,
                ledger_entry_id=entry.id,
                target_id=target.id,
                actual=actual,
                score=score,
                recorded_at=observation.observed_at,
            )
            existing = outcomes_by_target.get(target.id)
            if existing is not None:
                same_observed_fact = (
                    existing.actual.observed_day == actual.observed_day
                    and existing.actual.cash == actual.cash
                    and existing.actual.observation_reference
                    == actual.observation_reference
                )
                if not same_observed_fact:
                    raise ConflictError(
                        f"prediction target already has a different outcome: {target.id}"
                    )
                continue
            outcomes_by_target[target.id] = candidate
            new_outcomes.append(candidate)
    return tuple(new_outcomes)


def attribute_prediction_policy_path(
    outcome: PredictionOutcome,
    *,
    entry: PredictionLedgerEntry,
    decisions: Iterable[DecisionRecord],
) -> PredictionOutcome:
    """Separate model residuals from cash changes caused by later replanning.

    Forecasts are conditional on the action committed at issuance, including the
    declared stages and reversion of its experiment program. A later, unrelated
    action is valuable policy adaptation, but its cash effect is not evidence that
    the conditional world model was wrong.
    """

    if outcome.ledger_entry_id != entry.id:
        raise ValueError("prediction outcome does not belong to the supplied entry")
    origin = next(
        (decision for decision in decisions if decision.id == entry.decision_id),
        None,
    )
    if origin is None:
        raise ValueError("prediction entry cannot resolve its issuing decision")
    target = next(
        (target for target in entry.targets if target.id == outcome.target_id),
        None,
    )
    if target is None:
        raise ValueError("prediction outcome cannot resolve its target")
    origin_program = origin.action_plan.experiment_program
    origin_commitment = (
        origin_program.commitment_id if origin_program is not None else None
    )
    last_policy_week = target.target_day // 7 - 1
    subsequent = sorted(
        (
            decision
            for decision in decisions
            if entry.decision_week < decision.week <= last_policy_week
            and decision.actual_outcome is not None
        ),
        key=lambda decision: (decision.week, str(decision.id)),
    )
    compatible_hashes = {origin.action_plan.semantic_hash}
    for decision in subsequent:
        plan = decision.action_plan
        same_action = plan.semantic_hash in compatible_hashes
        same_program = (
            origin_commitment is not None
            and plan.experiment_program is not None
            and plan.experiment_program.commitment_id == origin_commitment
        )
        planned_reversion = (
            origin_commitment is not None
            and plan.strategy_family == f"experiment_revert_{origin_commitment}"
        )
        if same_action or same_program or planned_reversion:
            if planned_reversion:
                compatible_hashes.add(plan.semantic_hash)
            continue
        return outcome.model_copy(
            update={
                "attribution": PredictionOutcomeAttribution.POLICY_PATH_DIVERGED,
                "policy_divergence_week": decision.week,
            }
        )
    return outcome
