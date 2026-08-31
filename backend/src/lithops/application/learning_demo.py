"""Reproducible evidence builder for Lithops' closed learning loop."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from lithops.agents.candidate_model_builder import (
    ACQUISITION_BUILDER,
    PRICING_BUILDER,
    DeterministicCandidateModelBuilder,
)
from lithops.agents.common import (
    AgentPermissionDenied,
    AgentRole,
    AgentTool,
    RoleScopedToolRegistry,
)
from lithops.application.model_challenge import ModelChallengeOrchestrator
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.application.weekly_planning import prepare_weekly_plan
from lithops.benchmark.learning_scenario import SCENARIO_NAME
from lithops.domain.model_challenge import ParameterDirection, ParameterStepSize
from lithops.domain.ports import BenchmarkPort, LearningRepository, RunRepository
from lithops.domain.world_model import WorldModelParameterName
from lithops.world_model.bootstrap import P0_PRIORS

DEMO_WEEKS = 6
DEMO_HORIZON_DAYS = 42
DEMO_ROLLOUTS = 80
CAUSAL_DECISION_INDEX = 4
CAUSAL_PARAMETER = WorldModelParameterName.MARKETING_SATURATION


class LearningDemoRepository(RunRepository, LearningRepository, Protocol):
    """Persistence surface used by the deterministic demo runner."""


def _rounded(value: float) -> float:
    return round(value, 6)


async def run_learning_demo(
    *,
    repository: LearningDemoRepository,
    benchmark: BenchmarkPort,
) -> dict[str, object]:
    """Run the fixed scenario and return ID-free, byte-reproducible evidence."""

    executive = StaticDecisionEngine()
    challenge_orchestrator = ModelChallengeOrchestrator(
        repository=repository,
        builders=(
            DeterministicCandidateModelBuilder(
                spec=ACQUISITION_BUILDER,
                parameter_name=WorldModelParameterName.MARKETING_SATURATION,
                direction=ParameterDirection.DECREASE,
                step_size=ParameterStepSize.LARGE,
            ),
            DeterministicCandidateModelBuilder(
                spec=PRICING_BUILDER,
                parameter_name=WorldModelParameterName.PRICE_ELASTICITY,
                direction=ParameterDirection.INCREASE,
            ),
        ),
        complexity_penalty_per_unit=0.001,
        minimum_required_improvement=0.001,
    )
    # The demo's causal story is calibrated against a fixed parameter set: it
    # exercises the learning machinery, not the current prior catalog. Pinning
    # the priors keeps it byte-reproducible as the catalog grows — an extra
    # parameter would otherwise reshuffle every sampled rollout world.
    demo_priors = tuple(
        prior
        for prior in P0_PRIORS
        if prior.name
        not in {
            WorldModelParameterName.PARTICIPATION_CONVERSION_RATE,
            WorldModelParameterName.PARTICIPATION_SOFTNESS,
        }
    )
    manager = RunManager(
        repository=repository,
        benchmark=benchmark,
        decision_engine=executive,
        model_challenge_orchestrator=challenge_orchestrator,
        planning_rollouts=DEMO_ROLLOUTS,
        bootstrap_priors=demo_priors,
    )
    run = await manager.create_run(horizon_days=DEMO_HORIZON_DAYS)
    for week in range(DEMO_WEEKS):
        await manager.step_run(run.id, request_id=f"demo-week-{week}")
    replay_counts_before = (
        getattr(benchmark, "execute_action_calls", None),
        getattr(benchmark, "advance_week_calls", None),
    )
    replay = await manager.step_run(
        run.id,
        request_id=f"demo-week-{DEMO_WEEKS - 1}",
    )
    replay_counts_after = (
        getattr(benchmark, "execute_action_calls", None),
        getattr(benchmark, "advance_week_calls", None),
    )
    replay_without_reality_change = replay.replayed and replay_counts_before == replay_counts_after
    if not replay_without_reality_change:
        raise RuntimeError("fleet demo replay repeated an action or benchmark advance")

    decisions = await repository.list_decisions(run.id)
    models = await repository.list_world_models(run.id)
    predictions = await repository.list_predictions(run.id)
    outcomes = await repository.list_prediction_outcomes(run.id)
    health_signals = await repository.list_model_health_signals(run.id)
    events = await repository.list_events(run.id)
    started = next(event for event in events if event.type == "model_challenge.started")
    challenge_id = UUID(str(started.payload["challenge_id"]))
    benchmark_calls_before_denial = getattr(benchmark, "execute_action_calls", None)
    permission_registry = RoleScopedToolRegistry(repository=repository)
    try:
        await permission_registry.invoke(
            run_id=run.id,
            correlation_id=str(challenge_id),
            role=AgentRole.CANDIDATE_MODEL_BUILDER,
            agent_name="acquisition_builder",
            agent_version="1.0",
            tool=AgentTool.SET_PRICES,
            arguments={"A": "permission-canary-must-not-be-persisted"},
        )
    except AgentPermissionDenied:
        pass
    else:
        raise RuntimeError("candidate builder unexpectedly crossed the permission boundary")
    events = await repository.list_events(run.id)
    permission_event = next(
        event
        for event in events
        if event.type == "agent.permission_denied"
        and event.payload.get("correlation_id") == str(challenge_id)
    )
    if "permission-canary" in permission_event.model_dump_json():
        raise RuntimeError("permission audit event persisted raw tool input")

    challenge = await repository.get_model_challenge(challenge_id)
    proposals = await repository.list_model_builder_proposals(challenge_id)
    backtests = await repository.list_hypothesis_backtests(challenge_id)
    builder_calls = await repository.list_model_builder_calls(challenge_id)
    challenge_decision = await repository.get_model_challenge_decision(challenge_id)
    if challenge is None or challenge_decision is None:
        raise RuntimeError("dynamic fleet did not persist its terminal challenge decision")
    proposals_by_id = {proposal.id: proposal for proposal in proposals}
    selected_names = [
        proposals_by_id[proposal_id].builder_name
        for proposal_id in challenge_decision.selected_proposal_ids
    ]

    if len(decisions) != DEMO_WEEKS or len(models) != DEMO_WEEKS:
        raise RuntimeError("learning demo did not produce one decision and model per week")
    if decisions[-1].actual_outcome is None:
        raise RuntimeError("learning demo ended without a committed final observation")

    event_position = {event.id: index for index, event in enumerate(events)}
    prediction_before_action = True
    for decision in decisions:
        committed = next(
            event
            for event in events
            if event.type == "prediction.committed"
            and event.payload.get("decision_id") == str(decision.id)
        )
        action_events = [
            event
            for event in events
            if event.type == "action.executed"
            and event.payload.get("decision_id") == str(decision.id)
        ]
        checkpoint = next(
            event
            for event in events
            if event.type == "decision.committed"
            and event.payload.get("decision_id") == str(decision.id)
        )
        prediction_before_action &= all(
            event_position[committed.id] < event_position[action.id] < event_position[checkpoint.id]
            for action in action_events
        )
    if not prediction_before_action:
        raise RuntimeError("a prediction was not committed before action and observation")

    changed_decision = decisions[CAUSAL_DECISION_INDEX]
    old_model = models[CAUSAL_DECISION_INDEX - 1]
    new_model = models[CAUSAL_DECISION_INDEX]
    new_parameter = next(
        parameter for parameter in new_model.parameters if parameter.name is CAUSAL_PARAMETER
    )
    old_parameter = next(
        parameter for parameter in old_model.parameters if parameter.name is CAUSAL_PARAMETER
    )
    parameter_change = next(
        change for change in new_model.changes if change.parameter_name is CAUSAL_PARAMETER
    )
    simulation_seed = changed_decision.observation.day + new_model.version * 10_000
    decision_history = tuple(decisions[:CAUSAL_DECISION_INDEX])

    old_plan = await prepare_weekly_plan(
        run=run,
        observation=changed_decision.observation,
        world_model=old_model,
        executive=executive,
        n_rollouts=DEMO_ROLLOUTS,
        simulation_seed=simulation_seed,
        decision_history=decision_history,
    )
    isolated_model = old_model.model_copy(
        update={
            "parameters": tuple(
                new_parameter if parameter.name is CAUSAL_PARAMETER else parameter
                for parameter in old_model.parameters
            )
        }
    )
    isolated_plan = await prepare_weekly_plan(
        run=run,
        observation=changed_decision.observation,
        world_model=isolated_model,
        executive=executive,
        n_rollouts=DEMO_ROLLOUTS,
        simulation_seed=simulation_seed,
        decision_history=decision_history,
    )
    new_plan = await prepare_weekly_plan(
        run=run,
        observation=changed_decision.observation,
        world_model=new_model,
        executive=executive,
        n_rollouts=DEMO_ROLLOUTS,
        simulation_seed=simulation_seed,
        decision_history=decision_history,
    )

    old_strategy = old_plan.action_plan.strategy_family
    isolated_strategy = isolated_plan.action_plan.strategy_family
    new_strategy = new_plan.action_plan.strategy_family
    executed_strategy = changed_decision.action_plan.strategy_family
    old_utilities = {
        candidate.strategy: candidate.robust_utility for candidate in old_plan.candidate_records
    }
    isolated_utilities = {
        candidate.strategy: candidate.robust_utility
        for candidate in isolated_plan.candidate_records
    }
    common_strategies = old_utilities.keys() & isolated_utilities.keys()
    maximum_utility_delta = max(
        (
            abs(isolated_utilities[strategy] - old_utilities[strategy])
            for strategy in common_strategies
        ),
        default=0.0,
    )
    if maximum_utility_delta <= 1e-9:
        raise RuntimeError("marketing-saturation update had no counterfactual planning effect")
    if isolated_strategy != new_strategy:
        raise RuntimeError("the isolated causal update does not reproduce the new-model winner")
    if new_strategy != executed_strategy:
        raise RuntimeError("counterfactual replay does not match the persisted decision")

    evidence_outcome = max(
        (outcome for outcome in outcomes if outcome.actual.observed_day == 28),
        key=lambda outcome: outcome.score.normalized_absolute_error,
    )
    prediction = next(
        item for item in predictions if item.id == evidence_outcome.ledger_entry_id
    )
    target = next(
        item for item in prediction.targets if item.id == evidence_outcome.target_id
    )
    degraded_signal = next(
        signal for signal in health_signals if signal.evaluated_day == 28
    )

    event_trace: list[dict[str, object]] = []
    for event in events:
        if event.type == "prediction.matured":
            event_trace.append(
                {
                    "type": event.type,
                    "day": event.payload["observed_day"],
                    "interval_hit": event.payload["interval_hit"],
                }
            )
        elif event.type == "prediction.committed":
            event_trace.append(
                {
                    "type": event.type,
                    "target_days": event.payload["target_days"],
                }
            )
        elif event.type == "action.executed":
            event_trace.append(
                {
                    "type": event.type,
                    "tool": event.payload["tool"],
                }
            )
        elif event.type == "model_health.evaluated":
            event_trace.append(
                {
                    "type": event.type,
                    "status": str(event.payload["status"]),
                    "rebuild_recommended": event.payload["rebuild_recommended"],
                }
            )
        elif event.type == "world_model.updated":
            event_trace.append(
                {
                    "type": event.type,
                    "version": event.payload["version"],
                    "changed_parameters": [
                        str(item) for item in event.payload["changed_parameters"]
                    ],
                }
            )
        elif event.type == "decision.prepared":
            event_trace.append(
                {
                    "type": event.type,
                    "week": event.payload["week"],
                    "selected_strategy": event.payload["selected_strategy"],
                }
            )
        elif event.type == "decision.committed":
            event_trace.append(
                {
                    "type": event.type,
                    "week": event.payload["week"],
                    "observed_day": event.payload["day"],
                }
            )
        elif event.type in {
            "model_challenge.started",
            "model_builder.completed",
            "hypothesis.backtested",
            "model_challenge.completed",
            "agent.permission_denied",
        }:
            event_trace.append(
                {
                    "type": event.type,
                    "builder_name": event.payload.get("builder_name"),
                    "supported": event.payload.get("supported"),
                    "resolution": event.payload.get("resolution"),
                    "tool": event.payload.get("tool"),
                }
            )

    return {
        "scenario": SCENARIO_NAME,
        "evidence_boundary": (
            "Deterministic development harness; not primary CEO-Bench benchmark evidence."
        ),
        "run": {
            "horizon_days": DEMO_HORIZON_DAYS,
            "completed_steps": DEMO_WEEKS,
            "final_observed_day": decisions[-1].actual_outcome.day,
            "rollouts_per_candidate": DEMO_ROLLOUTS,
            "prediction_before_action_verified": prediction_before_action,
            "replay_without_duplicate_action_or_advance": replay_without_reality_change,
        },
        "prediction_miss": {
            "issued_day": prediction.issued_day,
            "horizon_days": target.horizon_days,
            "target_day": target.target_day,
            "predicted_cash": _rounded(target.point),
            "interval_95": [_rounded(target.lower), _rounded(target.upper)],
            "actual_cash": _rounded(evidence_outcome.actual.cash),
            "signed_residual_actual_minus_prediction": _rounded(
                evidence_outcome.score.signed_error
            ),
            "normalized_absolute_error": _rounded(
                evidence_outcome.score.normalized_absolute_error
            ),
            "interval_hit": evidence_outcome.score.interval_hit,
        },
        "model_health": {
            "evaluated_day": degraded_signal.evaluated_day,
            "status": degraded_signal.status.value,
            "rebuild_recommended": degraded_signal.rebuild_recommended,
            "trigger_codes": list(degraded_signal.trigger_codes),
        },
        "parameter_update": {
            "name": CAUSAL_PARAMETER.value,
            "old_model_version": old_model.version,
            "new_model_version": new_model.version,
            "old_estimate": _rounded(old_parameter.estimate),
            "new_estimate": _rounded(new_parameter.estimate),
            "old_confidence": _rounded(old_parameter.confidence),
            "new_confidence": _rounded(new_parameter.confidence),
            "method": parameter_change.update_method,
            "evidence_days": sorted(
                {
                    item.observed_day
                    for item in parameter_change.evidence
                    if item.observed_day is not None
                }
            ),
        },
        "causal_strategy_replay": {
            "observation_day": changed_decision.observation.day,
            "simulation_seed": simulation_seed,
            "old_model_strategy": old_strategy,
            "only_marketing_saturation_updated_strategy": isolated_strategy,
            "full_new_model_strategy": new_strategy,
            "persisted_strategy": executed_strategy,
            "strategy_changed": old_strategy != isolated_strategy,
            "maximum_robust_utility_delta": _rounded(maximum_utility_delta),
            "explanation": (
                "With observation and rollout worlds held fixed, replacing only "
                "marketing_saturation with its learned value changes simulated utility; "
                "the robust winner changes only when the evidence crosses its decision boundary."
            ),
        },
        "dynamic_fleet": {
            "trigger_status": challenge.status.value,
            "requested_builders": list(challenge.requested_builders),
            "builder_calls": [
                {
                    "builder_name": call.builder_name,
                    "builder_version": call.builder_version,
                    "prompt_version": call.prompt_version,
                    "provider": call.provider,
                    "model_name": call.model_name,
                    "attempt": call.attempt,
                    "status": call.status.value,
                    "input_hash_present": len(call.input_hash) == 64,
                }
                for call in builder_calls
            ],
            "hypotheses": [
                {
                    "builder_name": proposal.builder_name,
                    "family": proposal.family.value,
                    "parameter_changes": [
                        {
                            "parameter": adjustment.parameter_name.value,
                            "direction": adjustment.direction.value,
                            "step_size": adjustment.step_size.value,
                        }
                        for adjustment in proposal.diff.parameter_adjustments
                    ],
                    "supported": next(
                        result.supported
                        for result in backtests
                        if result.proposal_id == proposal.id
                    ),
                    "penalized_improvement": _rounded(
                        next(
                            result.penalized_improvement
                            for result in backtests
                            if result.proposal_id == proposal.id
                        )
                    ),
                }
                for proposal in proposals
                if proposal.builder_name != "executive_merge"
            ],
            "resolution": challenge_decision.resolution.value,
            "selected_builders": selected_names,
            "rejected_builders": sorted(
                proposal.builder_name
                for proposal in proposals
                if proposal.builder_name not in selected_names
                and proposal.builder_name != "executive_merge"
            ),
            "base_model_version": old_model.version,
            "active_model_version": new_model.version,
            "permission_boundary": {
                "denied_tool": permission_event.payload["tool"],
                "reason_code": permission_event.payload["reason_code"],
                "input_hash_present": len(permission_event.payload["input_hash"]) == 64,
                "raw_input_absent": True,
                "benchmark_calls_unchanged": (
                    benchmark_calls_before_denial
                    == getattr(benchmark, "execute_action_calls", None)
                ),
            },
        },
        "event_trace": event_trace,
    }
