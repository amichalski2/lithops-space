"""Resolve immutable artifacts to their governed executable runtimes."""

from lithops.domain.executable_model import ModelArtifact, ModelRuntimeKind
from lithops.domain.ports.executable_model import ExecutableCompanyModel
from lithops.model_runtime.conversion_assembly import ConversionAssemblyModel
from lithops.model_runtime.sandbox import SandboxedCompanyModel


def runtime_for_artifact(artifact: ModelArtifact) -> ExecutableCompanyModel:
    if artifact.runtime_kind is ModelRuntimeKind.TYPED_COMPONENT_ASSEMBLY:
        return ConversionAssemblyModel(artifact)
    if artifact.runtime_kind is ModelRuntimeKind.SANDBOXED_PYTHON:
        return SandboxedCompanyModel(artifact)
    raise ValueError("trusted baseline artifacts must be resolved by their owner")
