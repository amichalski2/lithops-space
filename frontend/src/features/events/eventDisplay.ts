import type { EventRecord } from "../../api/client";
import { compactMoney, percent, sentenceCase, titleCase } from "../../lib/format";

/** The institutional actor a judge should attribute each event to. */
export function eventActor(type: string): string {
  if (type.startsWith("world_model") || type.startsWith("model_health")) return "World model";
  if (type.startsWith("prediction")) return "Evaluator";
  if (type.startsWith("action")) return "Action gateway";
  if (type.startsWith("experiment")) return "Causal controller";
  if (type.startsWith("decision")) return "Executive";
  if (type.startsWith("run")) return "Run manager";
  return "CEO-Bench";
}

export function eventTitle(type: string): string {
  return titleCase(type.replace(".", " "));
}

export function eventDetail(event: EventRecord): string {
  const payload = (event.payload ?? {}) as Record<string, unknown>;
  switch (event.type) {
    case "run.created":
      return `Fresh session · ${String(payload.horizon_days ?? "—")} day horizon`;
    case "run.started":
    case "run.resumed":
      return "Autonomous loop operating";
    case "run.paused":
      return `Paused at safe checkpoint · day ${String(payload.day ?? "—")}`;
    case "run.pause_requested":
      return "Pause requested · stopping at the next checkpoint";
    case "run.failed":
      return String(payload.reason ?? "Run failed");
    case "world_model.created":
      return `World Model v${String(payload.version ?? 0)} bootstrapped from observable state`;
    case "world_model.updated": {
      const changed = Array.isArray(payload.changed_parameters) ? payload.changed_parameters : [];
      return changed.length
        ? `Model v${String(payload.version ?? "—")} · ${changed.map((name) => titleCase(String(name))).join(", ")} recalibrated`
        : `Model v${String(payload.version ?? "—")} · no parameter change`;
    }
    case "model_health.evaluated": {
      const status = sentenceCase(String(payload.status ?? "evaluated"));
      return payload.rebuild_recommended ? `${status} · rebuild recommended` : `${status} · calibration retained`;
    }
    case "decision.prepared":
      return `${titleCase(String(payload.selected_strategy ?? "Strategy selected"))} · ${String(payload.candidate_count ?? 0)} candidates simulated`;
    case "prediction.committed": {
      const targets = Array.isArray(payload.target_days) ? payload.target_days.join(", ") : "future";
      return `Forecast locked before advance · target days ${targets}`;
    }
    case "prediction.matured":
      return `${payload.interval_hit ? "Interval hit" : "Interval miss"} · error ${percent(Number(payload.normalized_absolute_error ?? 0), 1)}`;
    case "action.executed":
      return `${String(payload.tool ?? "action")} · receipt ${String(payload.receipt_id ?? "—").slice(0, 8)}`;
    case "experiment.program_started":
      return `${String(payload.control ?? "control")} commitment; matures week ${String(payload.minimum_maturity_week ?? "-")}`;
    case "experiment.program_continued":
      return `${String(payload.control ?? "control")} commitment retained; ends week ${String(payload.maximum_end_week ?? "-")}`;
    case "experiment.program_matured":
      return "Outcome window matured; explicit rollback is next";
    case "experiment.program_reversion_planned":
      return "Pre-committed operating level restored";
    case "experiment.program_reverted":
      return "Rollback committed with an auditable receipt";
    case "decision.committed":
      return `Week ${String(payload.week ?? "—")} committed · cash ${compactMoney(Number(payload.cash ?? 0))}`;
    default:
      return eventTitle(event.type);
  }
}
