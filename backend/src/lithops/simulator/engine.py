"""Monte Carlo rollout engine; no LLM calls belong in this module."""

from __future__ import annotations

from math import exp
from random import Random

from lithops.domain.world_model import WorldModelVersion
from lithops.simulator.components import (
    BASELINE_TRANSITION_ASSEMBLY,
    TransitionModelAssembly,
)
from lithops.simulator.models import (
    ProcessNoise,
    RolloutOutcome,
    SimulationAction,
    SimulationState,
    WeeklyShock,
)
from lithops.simulator.parameter_sampling import sample_parameters
from lithops.simulator.state_transition import advance_simulation_week


def _weekly_shock(
    process_noise: ProcessNoise | None,
    random: Random,
    cash_random: Random,
) -> WeeklyShock | None:
    """Draw one week of innovations from the rollout's own seeded generator."""

    if process_noise is None or not process_noise.active:
        return None
    return WeeklyShock(
        acquisition_multiplier=exp(random.gauss(0.0, process_noise.acquisition_sigma)),
        churn_delta=random.gauss(0.0, process_noise.churn_sigma),
        revenue_multiplier=exp(random.gauss(0.0, process_noise.revenue_sigma)),
        cash_flow_delta=cash_random.gauss(0.0, process_noise.cash_flow_sigma),
    )


def simulate(
    *,
    state: SimulationState,
    action: SimulationAction,
    world_model: WorldModelVersion,
    horizon_weeks: int = 12,
    n_rollouts: int = 1_000,
    seed: int = 0,
    process_noise: ProcessNoise | None = None,
    assembly: TransitionModelAssembly = BASELINE_TRANSITION_ASSEMBLY,
) -> tuple[RolloutOutcome, ...]:
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be at least 1")
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be at least 1")

    random = Random(seed)
    # Keep the new cash-flow innovation on its own deterministic stream so adding it
    # does not silently reshuffle the established acquisition/churn/revenue draws.
    cash_random = Random(seed ^ 0xD1B54A32D192ED03)
    outcomes: list[RolloutOutcome] = []
    for rollout_index in range(n_rollouts):
        parameters = sample_parameters(world_model, random)
        states = [state]
        current = state
        for _ in range(horizon_weeks):
            shock = _weekly_shock(process_noise, random, cash_random)
            current = advance_simulation_week(
                current,
                action,
                parameters,
                shock,
                assembly,
            )
            states.append(current)
        outcomes.append(
            RolloutOutcome(
                rollout_index=rollout_index,
                states=tuple(states),
                ending_cash=current.cash,
                ending_customers=current.customers,
                bankrupt=any(item.cash < 0 for item in states),
            )
        )
    return tuple(outcomes)


def simulate_action_path(
    *,
    state: SimulationState,
    actions: tuple[SimulationAction, ...],
    world_model: WorldModelVersion,
    n_rollouts: int = 1_000,
    seed: int = 0,
    process_noise: ProcessNoise | None = None,
    assembly: TransitionModelAssembly = BASELINE_TRANSITION_ASSEMBLY,
) -> tuple[RolloutOutcome, ...]:
    """Roll forward the actions actually committed on a historical policy path."""

    if not actions:
        raise ValueError("action path must contain at least one weekly action")
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be at least 1")

    random = Random(seed)
    cash_random = Random(seed ^ 0xD1B54A32D192ED03)
    outcomes: list[RolloutOutcome] = []
    for rollout_index in range(n_rollouts):
        parameters = sample_parameters(world_model, random)
        states = [state]
        current = state
        for action in actions:
            shock = _weekly_shock(process_noise, random, cash_random)
            current = advance_simulation_week(
                current,
                action,
                parameters,
                shock,
                assembly,
            )
            states.append(current)
        outcomes.append(
            RolloutOutcome(
                rollout_index=rollout_index,
                states=tuple(states),
                ending_cash=current.cash,
                ending_customers=current.customers,
                bankrupt=any(item.cash < 0 for item in states),
            )
        )
    return tuple(outcomes)
