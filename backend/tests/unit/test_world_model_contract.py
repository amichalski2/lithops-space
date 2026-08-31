from uuid import uuid4

import pytest
from lithops.domain.world_model import (
    EvidenceKind,
    EvidenceReference,
    RelationshipShape,
    WorldModelParameter,
    WorldModelParameterChange,
    WorldModelParameterName,
    WorldModelRelationship,
    WorldModelVersion,
)
from pydantic import ValidationError


def prior() -> EvidenceReference:
    return EvidenceReference(
        kind=EvidenceKind.GENERIC_PRIOR,
        reference="generic-saas-prior-v1",
    )


def parameter(
    name: WorldModelParameterName = WorldModelParameterName.PRICE_ELASTICITY,
) -> WorldModelParameter:
    return WorldModelParameter(
        name=name,
        estimate=0.5,
        lower_bound=0.1,
        upper_bound=0.9,
        confidence=0.35,
        unit="ratio",
        evidence=(prior(),),
    )


def relationship(
    parameter_name: WorldModelParameterName = WorldModelParameterName.PRICE_ELASTICITY,
) -> WorldModelRelationship:
    return WorldModelRelationship(
        key="price_to_conversion",
        cause="pricing",
        effect="conversion",
        shape=RelationshipShape.LINEAR,
        parameter_names=(parameter_name,),
        confidence=0.35,
        evidence=(prior(),),
    )


def world_model() -> WorldModelVersion:
    return WorldModelVersion(
        run_id=uuid4(),
        version=1,
        source_observation_day=0,
        parameters=(parameter(),),
        relationships=(relationship(),),
    )


def test_world_model_parameter_requires_explicit_valid_uncertainty() -> None:
    with pytest.raises(ValidationError, match="lower_bound <= estimate <= upper_bound"):
        WorldModelParameter(
            name=WorldModelParameterName.PRICE_ELASTICITY,
            estimate=1.2,
            lower_bound=0.1,
            upper_bound=0.9,
            confidence=0.5,
            unit="ratio",
            evidence=(prior(),),
        )

    with pytest.raises(ValidationError):
        parameter().model_copy(update={"confidence": 1.1}).model_validate(
            parameter().model_copy(update={"confidence": 1.1}).model_dump()
        )


def test_world_model_version_is_deeply_immutable() -> None:
    model = world_model()

    with pytest.raises(ValidationError, match="frozen"):
        model.version = 2  # type: ignore[misc]

    with pytest.raises(ValidationError, match="frozen"):
        model.parameters[0].estimate = 0.8  # type: ignore[misc]


def test_world_model_rejects_duplicate_nodes_and_missing_parameter_references() -> None:
    model = world_model()

    with pytest.raises(ValidationError, match="parameter names must be unique"):
        WorldModelVersion(
            **model.model_dump(exclude={"id", "parameters"}),
            parameters=(parameter(), parameter()),
        )

    with pytest.raises(ValidationError, match="relationships reference missing parameters"):
        WorldModelVersion(
            **model.model_dump(exclude={"id", "relationships"}),
            relationships=(relationship(WorldModelParameterName.CHURN_SENSITIVITY),),
        )


def test_world_model_lineage_is_explicit() -> None:
    first = world_model()

    with pytest.raises(ValidationError, match="version 1 cannot reference"):
        first.model_copy(update={"based_on_version_id": uuid4()}).model_validate(
            first.model_copy(update={"based_on_version_id": uuid4()}).model_dump()
        )

    changed_parameter = parameter().model_copy(update={"estimate": 0.6})
    change = WorldModelParameterChange(
        parameter_name=changed_parameter.name,
        previous_estimate=0.5,
        new_estimate=0.6,
        previous_confidence=0.35,
        new_confidence=0.35,
        update_method="bounded_test_update",
        evidence=(prior(),),
    )
    second = WorldModelVersion(
        **first.model_dump(
            exclude={
                "id",
                "version",
                "based_on_version_id",
                "parameters",
                "changes",
                "update_method",
            }
        ),
        version=2,
        based_on_version_id=first.id,
        parameters=(changed_parameter,),
        changes=(change,),
        update_method="bounded_test_update",
    )
    assert second.based_on_version_id == first.id
    assert second.changes == (change,)
