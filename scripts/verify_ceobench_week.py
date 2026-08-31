from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from lithops.application.step_run import RunManager, StaticDecisionEngine
from lithops.benchmark.ceobench import CeobenchAdapter, CeobenchCli
from lithops.infrastructure.persistence.repositories import InMemoryRunRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one disposable Lithops week through the official CEO-Bench CLI."
    )
    parser.add_argument("--python", required=True, help="Python 3.13+ executable")
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def verify(args: argparse.Namespace) -> dict[str, object]:
    executable = args.executable.resolve()
    adapter = CeobenchAdapter(
        cli=CeobenchCli(
            command=(args.python, str(executable)),
            working_directory=executable.parent,
        ),
        seed=args.seed,
    )
    manager = RunManager(
        repository=InMemoryRunRepository(),
        benchmark=adapter,
        decision_engine=StaticDecisionEngine(),
    )
    run = await manager.create_run(horizon_days=500)
    result = await manager.step_run(run.id, request_id="ceobench-dev-week-0")
    return {
        "benchmark_session_id": result.run.benchmark_session_id,
        "current_day": result.run.current_day,
        "cash": result.decision.actual_outcome.cash
        if result.decision.actual_outcome
        else None,
        "decision_id": str(result.decision.id),
        "forecast_horizons": [
            forecast.horizon_days for forecast in result.decision.forecasts.ordered()
        ],
        "receipts": [receipt.status for receipt in result.receipts],
    }


def main() -> None:
    project_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=project_env, override=False)
    missing = [name for name in ("ANTHROPIC_API_KEY",) if not os.getenv(name)]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    args = parse_args()
    print(json.dumps(asyncio.run(verify(args)), indent=2, default=str))


if __name__ == "__main__":
    main()
