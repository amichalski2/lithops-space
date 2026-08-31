"""Constrained subprocess runtime for untrusted agent-authored model code."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from lithops.domain.executable_model import (
    AssertionOperator,
    CompanyModelFitRequest,
    CompanyModelPredictRequest,
    FittedModel,
    ModelArtifact,
    ModelEntrypoint,
    ModelOutcomeDistribution,
    ModelRuntimeKind,
)
from lithops.model_runtime.invariants import evaluate_model_outcomes


class SandboxPolicyError(ValueError):
    """Candidate source violates the static sandbox policy."""


class SandboxExecutionError(RuntimeError):
    """Candidate source failed inside the isolated process."""


class SandboxTimeoutError(SandboxExecutionError):
    """Candidate source exceeded its wall-clock budget."""


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    timeout_seconds: float = 2.0
    max_source_bytes: int = 100_000
    max_ast_nodes: int = 10_000
    max_input_bytes: int = 2_000_000
    max_output_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("sandbox timeout must be positive")
        for value in (
            self.max_source_bytes,
            self.max_ast_nodes,
            self.max_input_bytes,
            self.max_output_bytes,
        ):
            if value < 1:
                raise ValueError("sandbox limits must be positive")


class SandboxTestResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    failure_reason: str | None = Field(default=None, max_length=1_000)


_FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_FORBIDDEN_NAMES = {
    "builtins",
    "ctypes",
    "importlib",
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}
_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
)

_CHILD_HARNESS = r'''
import json
import sys

_UINT64 = 0xFFFFFFFFFFFFFFFF


def uniform01(seed, index):
    """Deterministic [0, 1) draw from a (seed, index) pair.

    Generated models get no imports and therefore no RNG of their own. This
    splitmix64-style integer hash gives them reproducible per-rollout variation
    using only integer arithmetic, so the same seed always replays exactly.
    """

    value = (int(seed) * 0x9E3779B97F4A7C15 + int(index) * 0xBF58476D1CE4E5B9) & _UINT64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64
    value = value ^ (value >> 31)
    return (value >> 11) / 9007199254740992.0


def normal01(seed, index):
    """Approximately standard-normal draw built from twelve uniforms.

    The sum-of-uniforms construction keeps this pure integer/float arithmetic, so
    it reproduces bit-for-bit across platforms without depending on libm.
    """

    total = 0.0
    for offset in range(12):
        total += uniform01(seed, int(index) * 12 + offset)
    return total - 6.0


SAFE_BUILTINS = {
    "uniform01": uniform01,
    "normal01": normal01,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}

try:
    payload = json.loads(sys.stdin.read())
    scope = {"__builtins__": SAFE_BUILTINS}
    exec(compile(payload["source_code"], "<model-artifact>", "exec"), scope, scope)
    result = scope[payload["entrypoint"]](**payload["arguments"])
    sys.stdout.write(json.dumps({"ok": True, "result": result}, separators=(",", ":")))
except BaseException as exc:
    sys.stdout.write(json.dumps({
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
    }, separators=(",", ":")))
'''


class SandboxedPythonRunner:
    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def validate(self, artifact: ModelArtifact) -> None:
        if artifact.runtime_kind != ModelRuntimeKind.SANDBOXED_PYTHON:
            raise SandboxPolicyError("sandbox runner accepts only sandboxed Python artifacts")
        if artifact.dependencies:
            raise SandboxPolicyError("sandbox protocol v1 does not allow third-party dependencies")
        source = artifact.source_code or ""
        if len(source.encode("utf-8")) > self.policy.max_source_bytes:
            raise SandboxPolicyError("candidate source exceeds the configured byte limit")
        try:
            tree = ast.parse(source, filename="<model-artifact>")
        except SyntaxError as exc:
            raise SandboxPolicyError(f"candidate source is invalid Python: {exc.msg}") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > self.policy.max_ast_nodes:
            raise SandboxPolicyError("candidate source exceeds the configured AST limit")
        for statement in tree.body:
            if not isinstance(statement, ast.FunctionDef):
                raise SandboxPolicyError("candidate module may contain only function definitions")
            if statement.decorator_list:
                raise SandboxPolicyError("candidate functions cannot use decorators")
        for node in nodes:
            if isinstance(node, _FORBIDDEN_NODES):
                raise SandboxPolicyError(f"forbidden syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                raise SandboxPolicyError(f"forbidden name: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise SandboxPolicyError(
                    f"private or dunder attribute access is forbidden: {node.attr}"
                )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                    raise SandboxPolicyError(f"forbidden call: {node.func.id}")
                if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS:
                    raise SandboxPolicyError(f"forbidden call: {node.func.attr}")
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        required = {entrypoint.value for entrypoint in ModelEntrypoint}
        missing = sorted(required - function_names)
        if missing:
            raise SandboxPolicyError(
                f"candidate source is missing required entrypoints: {', '.join(missing)}"
            )

    def execute(
        self,
        artifact: ModelArtifact,
        entrypoint: ModelEntrypoint,
        arguments: dict[str, JsonValue],
    ) -> JsonValue:
        self.validate(artifact)
        payload = json.dumps(
            {
                "source_code": artifact.source_code,
                "entrypoint": entrypoint.value,
                "arguments": arguments,
            },
            separators=(",", ":"),
        )
        if len(payload.encode("utf-8")) > self.policy.max_input_bytes:
            raise SandboxPolicyError("sandbox input exceeds the configured byte limit")
        with tempfile.TemporaryDirectory(prefix="lithops-model-") as directory:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", _CHILD_HARNESS],
                    input=payload,
                    capture_output=True,
                    text=True,
                    cwd=Path(directory),
                    env={"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"},
                    timeout=self.policy.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeoutError(
                    f"model artifact exceeded {self.policy.timeout_seconds:g}s timeout"
                ) from exc
        if completed.returncode != 0:
            raise SandboxExecutionError(
                f"sandbox process failed with exit code {completed.returncode}"
            )
        if len(completed.stdout.encode("utf-8")) > self.policy.max_output_bytes:
            raise SandboxExecutionError("sandbox output exceeds the configured byte limit")
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise SandboxExecutionError("sandbox returned invalid JSON") from exc
        if not envelope.get("ok"):
            error_type = envelope.get("error_type", "ArtifactError")
            error = envelope.get("error", "candidate execution failed")
            raise SandboxExecutionError(f"{error_type}: {error}")
        return envelope.get("result")

    def run_artifact_tests(self, artifact: ModelArtifact) -> tuple[SandboxTestResult, ...]:
        results: list[SandboxTestResult] = []
        for test in artifact.tests:
            try:
                output = self.execute(artifact, test.entrypoint, test.arguments)
                for assertion in test.assertions:
                    actual = self._resolve_path(output, assertion.path)
                    self._assert_value(
                        actual,
                        assertion.operator,
                        assertion.expected,
                        assertion.tolerance,
                    )
            except (SandboxExecutionError, SandboxPolicyError, AssertionError) as exc:
                results.append(
                    SandboxTestResult(
                        name=test.name,
                        passed=False,
                        failure_reason=str(exc),
                    )
                )
            else:
                results.append(SandboxTestResult(name=test.name, passed=True))
        return tuple(results)

    @staticmethod
    def _resolve_path(value: JsonValue, path: str) -> JsonValue:
        current: Any = value
        for part in path.split("."):
            if isinstance(current, dict):
                if part not in current:
                    raise AssertionError(f"assertion path does not exist: {path}")
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                try:
                    current = current[int(part)]
                except IndexError as exc:
                    raise AssertionError(f"assertion path does not exist: {path}") from exc
            else:
                raise AssertionError(f"assertion path does not exist: {path}")
        return current

    @staticmethod
    def _assert_value(
        actual: JsonValue,
        operator: AssertionOperator,
        expected: JsonValue,
        tolerance: float,
    ) -> None:
        if operator == AssertionOperator.EQUALS:
            passed = actual == expected
        elif operator == AssertionOperator.APPROX:
            passed = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
            passed = passed and abs(float(actual) - float(expected)) <= tolerance
        elif operator == AssertionOperator.GREATER_THAN_OR_EQUAL:
            passed = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
            passed = passed and float(actual) >= float(expected)
        else:
            passed = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
            passed = passed and float(actual) <= float(expected)
        if not passed:
            raise AssertionError(
                f"artifact assertion failed: actual={actual!r} "
                f"operator={operator.value} expected={expected!r}"
            )


class SandboxedCompanyModel:
    """ExecutableCompanyModel implementation backed by the constrained subprocess."""

    def __init__(
        self,
        artifact: ModelArtifact,
        runner: SandboxedPythonRunner | None = None,
    ) -> None:
        self._artifact = artifact
        self.runner = runner or SandboxedPythonRunner()
        self.runner.validate(artifact)

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    def fit(self, request: CompanyModelFitRequest) -> FittedModel:
        result = self.runner.execute(
            self.artifact,
            ModelEntrypoint.FIT,
            {
                "history": list(request.history),
                "prior": request.prior,
                "seed": request.seed,
            },
        )
        if not isinstance(result, dict):
            raise SandboxExecutionError("fit must return a JSON object")
        return FittedModel.create(
            artifact=self.artifact,
            request=request,
            fitted_state=result,
        )

    def predict(self, request: CompanyModelPredictRequest) -> ModelOutcomeDistribution:
        if request.fitted_model.artifact_id != self.artifact.id:
            raise ValueError("fitted model does not belong to this artifact")
        action = dict(request.action)
        if request.policy_action_path:
            # Protocol 1.1 keeps the six-argument sandbox entrypoint stable.  The
            # reserved semantic field carries the observed weekly policy path.
            action["policy_action_path"] = list(request.policy_action_path)
        result = self.runner.execute(
            self.artifact,
            ModelEntrypoint.PREDICT,
            {
                "fitted": request.fitted_model.fitted_state,
                "state": request.state,
                "action": action,
                "horizons_days": list(request.horizons_days),
                "n_samples": request.n_rollouts,
                "seed": request.seed,
            },
        )
        if not isinstance(result, dict) or not isinstance(result.get("samples"), list):
            raise SandboxExecutionError("predict must return an object containing samples")
        distribution = ModelOutcomeDistribution.model_validate(
            {
                "artifact_id": self.artifact.id,
                "artifact_hash": self.artifact.content_hash,
                "fitted_model_id": request.fitted_model.id,
                "horizons_days": request.horizons_days,
                "n_rollouts": request.n_rollouts,
                "samples": result["samples"],
            }
        )
        invariant_report = evaluate_model_outcomes(distribution)
        if not invariant_report.valid:
            codes = ", ".join(
                sorted({violation.code.value for violation in invariant_report.violations})
            )
            raise SandboxExecutionError(f"model outcome failed economic invariants: {codes}")
        return distribution

    def diagnostics(self, fitted_model: FittedModel) -> dict[str, JsonValue]:
        if fitted_model.artifact_id != self.artifact.id:
            raise ValueError("fitted model does not belong to this artifact")
        result = self.runner.execute(
            self.artifact,
            ModelEntrypoint.DIAGNOSTICS,
            {"fitted": fitted_model.fitted_state},
        )
        if not isinstance(result, dict):
            raise SandboxExecutionError("diagnostics must return a JSON object")
        return result
