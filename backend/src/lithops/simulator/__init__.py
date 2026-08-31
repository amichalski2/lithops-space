"""Cheap deterministic-in-code company rollouts under uncertain model parameters."""

from lithops.simulator.engine import simulate, simulate_action_path
from lithops.simulator.models import (
    PendingResearch,
    ResearchTierFacts,
    RolloutOutcome,
    SimulationAction,
    SimulationState,
    TargetedAdAllocation,
    TargetedDevelopmentAllocation,
)
from lithops.simulator.strategy_search import (
    CandidateSimulation,
    StrategySearchResult,
    generate_candidate_actions,
    search_strategies,
)

__all__ = [
    "CandidateSimulation",
    "PendingResearch",
    "ResearchTierFacts",
    "RolloutOutcome",
    "SimulationAction",
    "SimulationState",
    "TargetedAdAllocation",
    "TargetedDevelopmentAllocation",
    "StrategySearchResult",
    "generate_candidate_actions",
    "search_strategies",
    "simulate",
    "simulate_action_path",
]
