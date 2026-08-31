from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from lithops.agents.executive import ExecutiveDecisionEngine
from lithops.domain.models import ObservationSnapshot, RunRecord
from lithops.infrastructure.llm.gemini_adk_provider import GeminiAdkProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one bounded Executive decision through Google ADK and Gemini."
    )
    parser.add_argument("--model", default=None)
    return parser.parse_args()


async def verify(model: str) -> dict[str, object]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing required environment variable: GEMINI_API_KEY or GOOGLE_API_KEY"
        )
    provider = GeminiAdkProvider(api_key=api_key, model=model)
    engine = ExecutiveDecisionEngine(provider)
    plan, forecasts = await engine.decide(
        run=RunRecord(),
        observation=ObservationSnapshot(
            day=0,
            cash=1_000_000,
            metrics={
                "revenue": 0,
                "customers": 0,
                "churn": 0,
            },
        ),
    )
    return {
        "provider": "google-adk",
        "model": provider.model_id,
        "strategy_family": plan.strategy_family,
        "tool": plan.commands[0].tool,
        "daily_spend": plan.commands[0].arguments,
        "forecast_horizons": [item.horizon_days for item in forecasts.ordered()],
        "forecast_intervals_valid": all(
            item.lower <= item.point <= item.upper for item in forecasts.items
        ),
    }


def main() -> None:
    project_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=project_env, override=False)
    args = parse_args()
    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    print(json.dumps(asyncio.run(verify(model)), indent=2))


if __name__ == "__main__":
    main()
