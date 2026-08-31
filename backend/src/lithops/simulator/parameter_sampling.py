"""Seeded sampling of uncertain world-model parameters."""

from __future__ import annotations

from random import Random

from lithops.domain.world_model import WorldModelParameterName, WorldModelVersion


def sample_parameters(
    world_model: WorldModelVersion,
    random: Random,
) -> dict[WorldModelParameterName, float]:
    """Sample once per plausible world using each estimate as the triangular mode."""

    return {
        parameter.name: random.triangular(
            parameter.lower_bound,
            parameter.upper_bound,
            parameter.estimate,
        )
        for parameter in world_model.parameters
    }
