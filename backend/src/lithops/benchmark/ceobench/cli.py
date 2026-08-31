from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lithops.domain.errors import BenchmarkError
from lithops.domain.models import CashForecasts


class CeobenchCommandError(BenchmarkError):
    """The public CEO-Bench CLI failed or returned an invalid response."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult: ...


class AsyncSubprocessRunner:
    """Runs a fixed argv vector without invoking a shell."""

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise CeobenchCommandError(
                f"CEO-Bench command timed out after {timeout_seconds:g} seconds"
            ) from exc

        return CommandResult(
            command=tuple(command),
            returncode=process.returncode or 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )


def parse_json_output(output: str) -> Any:
    """Extract one JSON document while tolerating harmless CLI log prefixes."""

    stripped = output.strip()
    if not stripped:
        raise CeobenchCommandError("CEO-Bench returned an empty response")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise CeobenchCommandError("CEO-Bench response did not contain valid JSON")


class CeobenchCli:
    def __init__(
        self,
        *,
        command: Sequence[str],
        working_directory: Path,
        runner: CommandRunner | None = None,
        default_timeout_seconds: float = 90,
        advance_timeout_seconds: float = 900,
    ) -> None:
        if not command or any(not part for part in command):
            raise ValueError("CEO-Bench command must contain non-empty argv parts")
        self.command = tuple(command)
        self.working_directory = working_directory
        self.runner = runner or AsyncSubprocessRunner()
        self.default_timeout_seconds = default_timeout_seconds
        self.advance_timeout_seconds = advance_timeout_seconds

    async def new_session(self, *, days: int, seed: int) -> Mapping[str, Any]:
        payload = await self._run_json(
            "new-session",
            "--days",
            str(days),
            "--seed",
            str(seed),
        )
        if not isinstance(payload, Mapping) or not payload.get("session_id"):
            raise CeobenchCommandError("new-session response is missing session_id")
        return payload

    async def query(self, session_id: str, sql: str) -> Any:
        return await self._run_json("query", sql, "--session", session_id)

    async def python_c(self, session_id: str, code: str) -> str:
        return await self._run("python-c", code, "--session", session_id)

    async def next_week(
        self,
        session_id: str,
        *,
        rationale: str,
        forecasts: CashForecasts,
    ) -> str:
        if not rationale.strip():
            raise CeobenchCommandError("weekly rationale must not be empty")

        arguments = ["next-week", rationale.strip()]
        for forecast in forecasts.ordered():
            arguments.extend(
                [
                    self._format_number(forecast.point),
                    self._format_number(forecast.lower),
                    self._format_number(forecast.upper),
                ]
            )
        arguments.extend(["--session", session_id])
        return await self._run(*arguments, timeout_seconds=self.advance_timeout_seconds)

    async def _run_json(self, *arguments: str) -> Any:
        return parse_json_output(await self._run(*arguments))

    async def _run(
        self,
        *arguments: str,
        timeout_seconds: float | None = None,
    ) -> str:
        command = (*self.command, *arguments)
        result = await self.runner.run(
            command,
            cwd=self.working_directory,
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-4_000:]
            raise CeobenchCommandError(
                f"CEO-Bench command failed with exit code {result.returncode}: {detail}"
            )
        return result.stdout

    @staticmethod
    def _format_number(value: float) -> str:
        return format(value, ".17g")
