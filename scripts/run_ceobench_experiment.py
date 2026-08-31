from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from lithops.agents.component_architect import (
    SMOOTH_CONVERSION_ARCHITECT,
    THRESHOLD_CONVERSION_ARCHITECT,
    ConversionComponentAuthor,
)
from lithops.agents.executive import ExecutiveDecisionEngine
from lithops.agents.model_coding_agent import (
    ACQUISITION_MODEL_CODER,
    CAPACITY_MODEL_CODER,
    PRICING_MODEL_CODER,
    RETENTION_MODEL_CODER,
    ModelCodingAgent,
)
from lithops.application.executable_model_challenge import ExecutableModelChallenge
from lithops.application.executable_model_planning import ExecutableModelPlanner
from lithops.application.experiment_checkpoint import ExperimentCheckpoint
from lithops.application.step_run import RunManager
from lithops.benchmark.ceobench import CeobenchAdapter, CeobenchCli
from lithops.domain.experiment_contracts import (
    CAPABILITY_CATALOG_VERSION,
    EXPERIMENT_PROTOCOL_VERSION,
    OBSERVATION_CONTRACT_VERSION,
)
from lithops.domain.models import RunRecord, RunStatus
from lithops.infrastructure.llm import FalRouterProvider, OpenRouterProvider
from lithops.infrastructure.llm.gemini_adk_provider import GeminiAdkProvider
from lithops.infrastructure.persistence.repositories import SupabaseRunRepository
from lithops.worker import AutonomousRunWorker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen/qwen3-32b"
DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
MODEL_AUTHOR_REQUEST_TIMEOUT_SECONDS = 75.0
MODEL_AUTHOR_DEADLINE_SECONDS = 240.0
DEFAULT_MILESTONE_WEEKS = frozenset({4, 8, 12, 16, 24})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume a checkpointed CEO-Bench experiment."
    )
    parser.add_argument(
        "--provider",
        choices=("openrouter", "gemini", "fal"),
        default="openrouter",
        help="Model provider for the Executive and the scoped coding agents.",
    )
    parser.add_argument("--python", default="python3.13", help="CEO-Bench Python executable")
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--weeks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=("low", "medium", "high"),
        help=(
            "Reasoning budget for models that think before answering. Their chain "
            "of thought is billed and timed as output: one weekly choice took 103s "
            "at the provider default and 10s at 'low'. Left unset for models where "
            "the parameter does not apply."
        ),
    )
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument(
        "--executive-authority-v2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pin the two-stage Gemini strategic-authority protocol for this run. "
            "If omitted, LITHOPS_EXECUTIVE_AUTHORITY_V2 is used."
        ),
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Stable lease owner for explicitly reclaiming an interrupted experiment.",
    )
    parser.add_argument(
        "--max-weeks-this-process",
        type=int,
        default=None,
        help="Stop cleanly after N newly committed weeks; resume with the same command.",
    )
    parser.add_argument(
        "--milestone-every",
        type=int,
        default=None,
        help=(
            "Freeze a milestone snapshot every N committed weeks instead of the "
            "default 4/8/12/16/24 set."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-open a failed run and reconcile its persisted/external week.",
    )
    return parser.parse_args()


def _week_log_entry(
    committed: RunRecord,
    checkpoint: Any,
    report: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """One structured line per committed week.

    Written as a JSON object on stdout so Cloud Logging parses it into
    queryable fields; `severity` and `message` are the fields it reserves. The
    payload is the decision itself — what the Executive chose and why it says
    that is the constraint — so the reasoning chain is readable from the log
    stream alone, without opening the report.
    """

    weeks = report.get("weeks") or []
    latest = max(weeks, key=lambda week: week.get("week", -1), default={})
    metrics = (latest.get("observed") or {}).get("metrics", {})
    portfolio_revisions = report.get("strategic_portfolio_revisions") or []
    portfolio = (
        portfolio_revisions[-1].get("portfolio", {}) if portfolio_revisions else {}
    )
    entry = {
        "severity": "INFO",
        "component": "lithops.week",
        "message": (
            f"week {committed.current_day // 7} committed: "
            f"{latest.get('strategy', 'n/a')}"
        ),
        "checkpoint": str(checkpoint_path),
        "run_id": str(committed.id),
        "day": committed.current_day,
        "status": committed.status,
        "world_model_version": checkpoint.world_model_version,
        "week": committed.current_day // 7,
        "strategy": latest.get("strategy"),
        "selection_reason_code": latest.get("selection_reason_code"),
        "binding_constraint": portfolio.get("binding_constraint"),
        "cash": (latest.get("observed") or {}).get("cash"),
    }
    for name in (
        "active_customers",
        "weekly_revenue",
        "weekly_conversions",
        "churn_rate",
        "competitor_quality_bar_shift",
    ):
        value = metrics.get(name)
        if isinstance(value, int | float):
            entry[name] = value
    return entry


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _usage_by_role(
    executive_provider: object, author_provider: object
) -> dict[str, dict[str, float] | None]:
    """Token and cost accounting for the two provider instances this run uses.

    Providers without the accessor (test doubles, older builds) report nothing
    rather than failing the report.
    """

    def snapshot(provider: object) -> dict[str, float] | None:
        reader = getattr(provider, "usage_snapshot", None)
        return reader() if callable(reader) else None

    return {
        "executive": snapshot(executive_provider),
        "model_coding_agents": snapshot(author_provider),
    }


def _milestone_weeks(every: int | None, weeks: int) -> frozenset[int]:
    """Which committed weeks get a frozen snapshot beside the rolling artifacts.

    ``checkpoint.json`` and ``report.json`` are rewritten every week, so resume
    never depends on this set; milestones only preserve the intermediate states
    a longer run would otherwise overwrite.
    """

    if every is None:
        return DEFAULT_MILESTONE_WEEKS
    if every < 1:
        raise ValueError("--milestone-every must be at least 1")
    return frozenset(range(every, weeks + 1, every))


def _milestone_path(path: Path, week: int) -> Path:
    return path.with_name(f"{path.stem}-week-{week:02d}{path.suffix}")


def _load_checkpoint(path: Path) -> ExperimentCheckpoint | None:
    if not path.exists():
        return None
    return ExperimentCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_python_command(value: str) -> str:
    """Anchor interpreter paths without dereferencing virtualenv shims.

    ``Path.resolve()`` follows the ``bin/python`` symlink on POSIX. Passing
    that base interpreter to a subprocess drops the virtualenv site-packages.
    ``absolute()`` makes the path independent of the CLI working directory
    while preserving the final symlink and its virtualenv identity.
    """

    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        absolute = candidate.absolute()
        if not absolute.is_file():
            raise FileNotFoundError(f"CEO-Bench Python executable does not exist: {absolute}")
        return str(absolute)
    return value


async def _preflight_ceobench_runtime(python_command: str, *, cwd: Path) -> None:
    """Validate server-side CEO-Bench imports before creating a persisted run."""

    process = await asyncio.create_subprocess_exec(
        python_command,
        "-c",
        "import numpy, pandas, sklearn, sqlcipher3",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[-2_000:]
        raise RuntimeError(
            "CEO-Bench Python preflight failed; install external/ceobench-dev/"
            f"requirements.txt into that interpreter. Detail: {detail}"
        )


async def _latest_model_version(
    repository: SupabaseRunRepository,
    run_id,
) -> int | None:
    model = await repository.get_latest_world_model(run_id)
    return model.version if model is not None else None


async def _checkpoint_for(
    repository: SupabaseRunRepository,
    run: RunRecord,
    *,
    provider: str,
    model: str,
    seed: int,
    weeks: int,
    executive_authority_v2: bool,
) -> ExperimentCheckpoint:
    return ExperimentCheckpoint.from_run(
        run,
        provider=provider,
        model=model,
        benchmark_seed=seed,
        target_weeks=weeks,
        executive_authority_v2=executive_authority_v2,
        world_model_version=await _latest_model_version(repository, run.id),
    )


async def _build_report(
    repository: SupabaseRunRepository,
    run: RunRecord,
    *,
    provider: str,
    model: str,
    seed: int,
    weeks: int,
    executive_authority_v2: bool,
    usage_by_role: dict[str, dict[str, float] | None] | None = None,
) -> dict[str, Any]:
    decisions = await repository.list_decisions(run.id)
    receipts_by_decision = {
        decision.id: await repository.list_receipts(decision.id) for decision in decisions
    }
    predictions = await repository.list_predictions(run.id)
    outcomes = await repository.list_prediction_outcomes(run.id)
    experiment_outcomes = await repository.list_experiment_outcomes(run.id)
    insight_records = await repository.list_insight_records(run.id)
    portfolio_revisions = await repository.list_portfolio_revisions(run.id)
    commitment_reviews = await repository.list_commitment_reviews(run.id)
    candidate_evaluation_sets = []
    executive_choices = []
    for decision in decisions:
        evaluation_set = await repository.get_candidate_evaluation_set(
            run.id, decision.week
        )
        if evaluation_set is not None:
            candidate_evaluation_sets.append(evaluation_set)
        executive_choice = await repository.get_executive_choice(run.id, decision.week)
        if executive_choice is not None:
            executive_choices.append(executive_choice)
    models = await repository.list_world_models(run.id)
    health = await repository.list_model_health_signals(run.id)
    events = await repository.list_events(run.id)
    challenge_ids: list[UUID] = []
    for event in events:
        if event.type != "executable_model_challenge.started":
            continue
        try:
            challenge_id = UUID(str(event.payload.get("challenge_id")))
        except ValueError:
            continue
        if challenge_id not in challenge_ids:
            challenge_ids.append(challenge_id)
    challenges = []
    activations = await repository.list_model_activations(run.id)
    all_evaluation_folds = await repository.list_temporal_evaluation_folds(run.id)
    for challenge_id in challenge_ids:
        authoring_receipts = await repository.list_model_artifact_authoring_receipts(
            run.id,
            challenge_id,
        )
        artifacts = [
            await repository.get_model_artifact(receipt.artifact_id)
            for receipt in authoring_receipts
        ]
        folds = [fold for fold in all_evaluation_folds if fold.challenge_id == challenge_id]
        promotion = await repository.get_model_promotion_decision_for_challenge(
            run.id,
            challenge_id,
        )
        activation = next(
            (
                item
                for item in activations
                if promotion is not None and item.promotion_decision_id == promotion.id
            ),
            None,
        )
        outcome_event = next(
            (
                event
                for event in reversed(events)
                if event.payload.get("challenge_id") == str(challenge_id)
                and event.type
                in {
                    "executable_model_challenge.completed",
                    "executable_model_challenge.failed",
                }
            ),
            None,
        )
        challenges.append(
            {
                "challenge_id": challenge_id,
                "authoring_receipts": [item.model_dump(mode="json") for item in authoring_receipts],
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "temporal_folds": [item.model_dump(mode="json") for item in folds],
                "promotion": (promotion.model_dump(mode="json") if promotion is not None else None),
                "activation": (
                    activation.model_dump(mode="json") if activation is not None else None
                ),
                "outcome": (
                    outcome_event.model_dump(mode="json") if outcome_event is not None else None
                ),
            }
        )
    relevant_events = [
        event
        for event in events
        if event.type.startswith(
            ("worker.", "model_", "executable_model_", "agent.", "experiment.")
        )
        or event.type
        in {
            "run.failed",
            "run.retry_started",
            "operation.failed",
            "operation.reclaimed",
            "benchmark.advance_reconciled",
            "decision.committed",
            "decision.candidate_pool_degraded",
            "decision.exploration_assessed",
            "decision.going_concern_degraded",
            "decision.no_viable_plan",
            "decision.strategy_exhausted",
            "decision.candidate_diversity_low",
            "executive.unavailable",
            "executive_candidate_selected",
            "strategy_portfolio_updated",
        }
    ]
    final_observation = (
        decisions[-1].actual_outcome
        if decisions and decisions[-1].actual_outcome is not None
        else decisions[-1].observation
        if decisions
        else None
    )
    enterprise_response_count = sum(
        len(command.arguments.get("deals", []))
        for decision in decisions
        for command in decision.action_plan.commands
        if command.tool == "send_enterprise_deal"
    )
    authoring_receipts = [
        receipt for challenge in challenges for receipt in challenge["authoring_receipts"]
    ]
    return {
        "experiment": {
            "provider": provider,
            "model": model,
            "benchmark": "ceobench",
            "benchmark_seed": seed,
            "target_weeks": weeks,
            "executive_authority_v2": executive_authority_v2,
            "observation_contract": OBSERVATION_CONTRACT_VERSION,
            "experiment_protocol": EXPERIMENT_PROTOCOL_VERSION,
            "capability_catalog": CAPABILITY_CATALOG_VERSION,
            "canonical_checkpoint": "supabase-weekly-commit",
        },
        "run": run.model_dump(mode="json"),
        "summary": {
            "committed_weeks": len(
                [decision for decision in decisions if decision.actual_outcome is not None]
            ),
            "decision_count": len(decisions),
            "prediction_count": len(predictions),
            "matured_outcome_count": len(outcomes),
            "world_model_versions": len(models),
            "health_signal_count": len(health),
            "rebuild_recommendations": sum(item.rebuild_recommended for item in health),
            "provider_validation_retries": sum(
            int((role or {}).get("validation_retries", 0))
            for role in (usage_by_role or {}).values()
        ),
        "retry_event_count": sum(
                event.type in {"worker.step_retrying", "run.retry_started"} for event in events
            ),
            "model_challenge_count": len(challenges),
            "model_authoring_call_count": sum(
                len(challenge["authoring_receipts"]) for challenge in challenges
            ),
            "model_artifact_count": sum(len(challenge["artifacts"]) for challenge in challenges),
            "model_promotion_count": sum(
                challenge["promotion"] is not None
                and challenge["promotion"]["disposition"] == "promoted"
                for challenge in challenges
            ),
            "experiment_outcome_count": len(experiment_outcomes),
            "experiment_outcome_statuses": sorted(
                {item.outcome_status.value for item in experiment_outcomes}
            ),
            "insight_record_count": len(insight_records),
            "insight_parse_statuses": sorted(
                {item.parse_status.value for item in insight_records}
            ),
            "enterprise_offers_sent": sum(
                event.type == "enterprise.turn_sent" for event in events
            ),
            "configuration_incomplete_weeks": sum(
                event.type == "decision.prepared"
                and "configuration_incomplete" in str(event.payload.get("diagnostics", ""))
                for event in events
            ),
            "executed_tools": sorted(
                {
                    command.tool
                    for decision in decisions
                    for command in decision.action_plan.commands
                }
            ),
            "portfolio_revision_count": len(portfolio_revisions),
            "candidate_evaluation_set_count": len(candidate_evaluation_sets),
            "executive_choice_count": len(executive_choices),
            "commitment_review_count": len(commitment_reviews),
            "active_model_sequence": activations[-1].sequence if activations else None,
            "enterprise_responses_attempted": enterprise_response_count,
            "final_funnel": (
                {
                    key: final_observation.metrics.get(key)
                    for key in (
                        "weekly_leads",
                        "weekly_conversions",
                        "weekly_lost_leads",
                        "total_leads",
                        "total_conversions",
                        "total_lost_leads",
                        "pending_leads",
                        "active_customers",
                        "active_seats",
                        "weekly_revenue",
                        "lead_conversion_rate",
                        "entry_price_monthly",
                        "lead_promotion_monthly",
                        "open_enterprise_threads",
                    )
                }
                if final_observation is not None
                else None
            ),
        },
        "experiment_outcomes": [
            item.model_dump(mode="json") for item in experiment_outcomes
        ],
        "insight_records": [item.model_dump(mode="json") for item in insight_records],
        "strategic_portfolio_revisions": [
            item.model_dump(mode="json") for item in portfolio_revisions
        ],
        "commitment_reviews": [
            item.model_dump(mode="json") for item in commitment_reviews
        ],
        "candidate_evaluation_sets": [
            item.model_dump(mode="json") for item in candidate_evaluation_sets
        ],
        "executive_choices": [
            item.model_dump(mode="json") for item in executive_choices
        ],
        "provider_trace": {
            "executive": {
                "provider": provider,
                "model": model,
                "prompt_versions": sorted({decision.prompt_version for decision in decisions}),
                "calls_committed": len(decisions),
                "usage": (usage_by_role or {}).get("executive"),
            },
            "model_coding_agents": {
                "provider": provider,
                "model": model,
                "durable_call_count": len(authoring_receipts),
                "author_keys": sorted({receipt["author_key"] for receipt in authoring_receipts}),
                "usage": (usage_by_role or {}).get("model_coding_agents"),
            },
        },
        "weeks": [
            {
                "week": decision.week,
                "strategy": decision.action_plan.strategy_family,
                "observed": decision.observation.model_dump(mode="json"),
                "actual": (
                    decision.actual_outcome.model_dump(mode="json")
                    if decision.actual_outcome is not None
                    else None
                ),
                "world_model_version_id": decision.world_model_version_id,
                "model_artifact_id": decision.model_artifact_id,
                "model_artifact_hash": decision.model_artifact_hash,
                "fitted_model_id": decision.fitted_model_id,
                "fitted_state_hash": decision.fitted_state_hash,
                "prediction_id": decision.prediction_id,
                "prompt_version": decision.prompt_version,
                "selection_reason_code": decision.selection_reason_code,
                "selection_reason": decision.selection_reason,
                "selected_bankruptcy_probability": next(
                    (
                        candidate.bankruptcy_probability
                        for candidate in decision.candidate_evaluations
                        if candidate.strategy == decision.action_plan.strategy_family
                    ),
                    None,
                ),
                "actions": [
                    command.model_dump(mode="json") for command in decision.action_plan.commands
                ],
                "forecasts": decision.forecasts.model_dump(mode="json"),
                "candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in decision.candidate_evaluations
                ],
                "receipts": [
                    receipt.model_dump(mode="json") for receipt in receipts_by_decision[decision.id]
                ],
            }
            for decision in decisions
        ],
        "prediction_outcomes": [item.model_dump(mode="json") for item in outcomes],
        "world_models": [item.model_dump(mode="json") for item in models],
        "model_health": [item.model_dump(mode="json") for item in health],
        "executable_model_challenges": challenges,
        "active_model_assignments": [item.model_dump(mode="json") for item in activations],
        "audit_events": [item.model_dump(mode="json") for item in relevant_events],
    }


async def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.weeks < 1 or args.rollouts < 1:
        raise ValueError("weeks and rollouts must be positive")
    if args.provider == "gemini":
        model = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        provider_key_name = "GEMINI_API_KEY"
    elif args.provider == "fal":
        # fal's router speaks OpenRouter model ids, so the same default and
        # override variable apply; only the credentials and transport differ.
        model = args.model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        provider_key_name = "FAL_KEY"
    else:
        model = args.model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        provider_key_name = "OPENROUTER_API_KEY"
    executive_authority_v2 = (
        args.executive_authority_v2
        if args.executive_authority_v2 is not None
        else os.getenv("LITHOPS_EXECUTIVE_AUTHORITY_V2", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    required = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SECRET_KEY": os.getenv("SUPABASE_SECRET_KEY"),
        provider_key_name: os.getenv(provider_key_name),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"missing required environment variables: {', '.join(missing)}")

    executable = args.executable.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"CEO-Bench executable does not exist: {executable}")
    python_command = _resolve_python_command(args.python)
    await _preflight_ceobench_runtime(python_command, cwd=executable.parent)
    repository = SupabaseRunRepository(
        url=required["SUPABASE_URL"] or "",
        secret_key=required["SUPABASE_SECRET_KEY"] or "",
    )
    provider_key = required[provider_key_name] or ""
    if args.provider == "gemini":
        executive_provider = GeminiAdkProvider(
            api_key=provider_key,
            model=model,
            agent_name="lithops_executive",
            agent_description="Selects one bounded weekly company action plan.",
        )
        author_provider = GeminiAdkProvider(
            api_key=provider_key,
            model=model,
            agent_name="lithops_model_coder",
            agent_description="Authors one scoped executable company model.",
        )
    elif args.provider == "fal":
        executive_provider = FalRouterProvider(
            api_key=provider_key,
            model=model,
            timeout_seconds=240.0,
        )
        author_provider = FalRouterProvider(
            api_key=provider_key,
            model=model,
            timeout_seconds=MODEL_AUTHOR_REQUEST_TIMEOUT_SECONDS,
        )
    else:
        executive_provider = OpenRouterProvider(
            api_key=provider_key,
            model=model,
            timeout_seconds=240.0,
            reasoning_effort=args.reasoning_effort,
        )
        author_provider = OpenRouterProvider(
            api_key=provider_key,
            model=model,
            timeout_seconds=MODEL_AUTHOR_REQUEST_TIMEOUT_SECONDS,
            provider_sort="throughput",
        )
    adapter = CeobenchAdapter(
        cli=CeobenchCli(
            command=(python_command, str(executable)),
            working_directory=executable.parent,
            default_timeout_seconds=240.0,
            advance_timeout_seconds=900.0,
        ),
        seed=args.seed,
    )
    executive = ExecutiveDecisionEngine(executive_provider)
    coding_agents = (
        ConversionComponentAuthor(
            spec=SMOOTH_CONVERSION_ARCHITECT,
            provider=author_provider,
            provider_name=args.provider,
        ),
        ConversionComponentAuthor(
            spec=THRESHOLD_CONVERSION_ARCHITECT,
            provider=author_provider,
            provider_name=args.provider,
        ),
        *tuple(
            ModelCodingAgent(
                spec=spec,
                provider=author_provider,
                provider_name=args.provider,
                # The v5 contract includes executable counterfactual checks. Give the
                # author one feedback-guided repair attempt instead of discarding an
                # otherwise useful family after the first invalid draft.
                max_attempts=2,
            )
            for spec in (
                PRICING_MODEL_CODER,
                ACQUISITION_MODEL_CODER,
                RETENTION_MODEL_CODER,
                CAPACITY_MODEL_CODER,
            )
        ),
    )
    manager = RunManager(
        repository=repository,
        benchmark=adapter,
        decision_engine=executive,
        executable_model_planner=ExecutableModelPlanner(
            repository=repository,
            executive=executive,
            n_rollouts=args.rollouts,
        ),
        executable_model_challenge=ExecutableModelChallenge(
            repository=repository,
            authors=coding_agents,
            author_timeout_seconds=MODEL_AUTHOR_DEADLINE_SECONDS,
        ),
        model_challenge_cooldown_days=28,
        planning_rollouts=args.rollouts,
        executive_authority_v2=executive_authority_v2,
    )

    stored_checkpoint = _load_checkpoint(args.checkpoint)
    if stored_checkpoint is None:
        run = await manager.create_run(horizon_days=args.weeks * 7)
        run = await manager.start_run(run.id)
    else:
        run = await manager.get_run(stored_checkpoint.run_id)
        stored_checkpoint.assert_compatible(
            run,
            provider=args.provider,
            model=model,
            benchmark_seed=args.seed,
            target_weeks=args.weeks,
            executive_authority_v2=executive_authority_v2,
        )
        if run.status == RunStatus.PAUSED:
            run = await manager.resume_run(run.id)
        elif run.status == RunStatus.FAILED:
            if not args.retry_failed:
                raise RuntimeError("run is failed; pass --retry-failed to reconcile and resume")
            run = await manager.retry_failed_run(run.id)

    milestone_weeks = _milestone_weeks(args.milestone_every, args.weeks)

    async def persist_artifacts(committed: RunRecord, _: int) -> None:
        checkpoint = await _checkpoint_for(
            repository,
            committed,
            provider=args.provider,
            model=model,
            seed=args.seed,
            weeks=args.weeks,
            executive_authority_v2=executive_authority_v2,
        )
        checkpoint_payload = checkpoint.model_dump(mode="json")
        report_payload = await _build_report(
            repository,
            committed,
            provider=args.provider,
            model=model,
            seed=args.seed,
            weeks=args.weeks,
            executive_authority_v2=executive_authority_v2,
            usage_by_role=_usage_by_role(executive_provider, author_provider),
        )
        _atomic_json(args.checkpoint, checkpoint_payload)
        _atomic_json(args.report, report_payload)
        committed_week = committed.current_day // 7
        if committed.current_day % 7 == 0 and committed_week in milestone_weeks:
            _atomic_json(_milestone_path(args.checkpoint, committed_week), checkpoint_payload)
            _atomic_json(_milestone_path(args.report, committed_week), report_payload)
        print(
            json.dumps(
                _week_log_entry(committed, checkpoint, report_payload, args.checkpoint),
                default=str,
            ),
            flush=True,
        )

    await persist_artifacts(run, 0)
    if run.status in {RunStatus.COMPLETED, RunStatus.BANKRUPT}:
        return await _build_report(
            repository,
            run,
            provider=args.provider,
            model=model,
            seed=args.seed,
            weeks=args.weeks,
            executive_authority_v2=executive_authority_v2,
            usage_by_role=_usage_by_role(executive_provider, author_provider),
        )

    worker = AutonomousRunWorker(
        manager=manager,
        repository=repository,
        owner_id=args.worker_id or f"experiment-{run.id}",
        # A single simulated week can run for the better part of an hour when
        # the benchmark's customer models hit provider rate limits (hundreds of
        # per-customer calls after a marketing push). The lease renews only
        # between weeks, so the TTL must outlast the slowest realistic week —
        # a 25-minute TTL killed a run at a 55-minute week.
        lease_ttl_seconds=5_400,
        step_timeout_seconds=4_800.0,
        retry_backoff_seconds=5.0,
    )
    result = await worker.run(
        run.id,
        max_weeks=args.max_weeks_this_process,
        on_checkpoint=persist_artifacts,
    )
    await persist_artifacts(result.run, result.weeks_completed)
    return await _build_report(
        repository,
        result.run,
        provider=args.provider,
        model=model,
        seed=args.seed,
        weeks=args.weeks,
        executive_authority_v2=executive_authority_v2,
        usage_by_role=_usage_by_role(executive_provider, author_provider),
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    report = asyncio.run(run_experiment(args))
    print(json.dumps(report["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
