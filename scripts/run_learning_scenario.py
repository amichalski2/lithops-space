"""Generate reproducible evidence for the closed-loop learning demo."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from lithops.application.learning_demo import run_learning_demo
from lithops.benchmark.learning_scenario import LearningScenarioBenchmarkAdapter
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


async def _run_once() -> dict[str, object]:
    return await run_learning_demo(
        repository=InMemoryRunRepository(),
        benchmark=LearningScenarioBenchmarkAdapter(),
    )


async def _main(output: Path, repeat: int) -> None:
    results = [await _run_once() for _ in range(repeat)]
    canonical = json.dumps(results[0], indent=2, sort_keys=True) + "\n"
    canonical_compact = json.dumps(results[0], sort_keys=True)
    if any(
        json.dumps(result, sort_keys=True) != canonical_compact
        for result in results[1:]
    ):
        raise RuntimeError("learning scenario evidence differed between repeated runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical, encoding="utf-8")
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    print(f"reproducible_runs={repeat}")
    print(f"sha256={digest}")
    print(f"evidence={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contracts/examples/learning-wow-scenario.json"),
    )
    parser.add_argument("--repeat", type=int, default=2, choices=range(2, 11))
    arguments = parser.parse_args()
    asyncio.run(_main(arguments.output, arguments.repeat))
