import type {
  DecisionExplanation,
  DecisionRecord,
  EventRecord,
  ModelHealthSignal,
  PredictionView,
  RunRecord,
  RunReport,
  WorldModelVersion,
} from "../api/client";
import { METRIC_NAMES, metric, type Observation } from "./format";

export const TERMINAL_STATUSES = ["completed", "bankrupt", "failed"] as const;
export const HORIZONS = [7, 28, 84, 182] as const;

export type AnnotatedEvent = EventRecord & { effectiveDay: number };

export type ReplayData = {
  run: RunRecord;
  events: AnnotatedEvent[];
  decisions: DecisionRecord[];
  predictions: PredictionView[];
  latestModel: WorldModelVersion | null;
  report: RunReport | null;
  explanations: Map<string, DecisionExplanation>;
};

export type MarkerState = "pending" | "hit" | "miss" | "awaiting";

export type PredictionMarker = {
  key: string;
  targetDay: number;
  horizonDays: number;
  issuedDay: number;
  decisionWeek: number;
  point: number;
  lower: number;
  upper: number;
  actualCash: number | null;
  state: MarkerState;
};

export type CompanyState = {
  observation: Observation;
  cash: number;
  revenue: number | null;
  customers: number | null;
  churn: number | null;
  /** Weeks of runway implied by the latest weekly cash delta; null while cash is not falling. */
  runwayWeeks: number | null;
  /** Week-over-week active customer growth; null without a comparable earlier snapshot. */
  growth: number | null;
};

export type ModelAtDay = {
  model: WorldModelVersion | null;
  /** False when the panel is showing the newest known model instead of the one used that week. */
  exact: boolean;
};

export type ReplayInput = {
  run: RunRecord;
  events: EventRecord[];
  decisions: DecisionRecord[];
  predictions: PredictionView[];
  latestModel: WorldModelVersion | null;
  report: RunReport | null;
  explanations?: Map<string, DecisionExplanation>;
};

/** Normalizes one batch of API responses into the day-indexed model every selector reads. */
export function buildReplayData(input: ReplayInput): ReplayData {
  return {
    run: input.run,
    events: annotateEvents(input.events, input.run.current_day),
    decisions: [...input.decisions].sort((left, right) => left.week - right.week),
    predictions: input.predictions,
    latestModel: input.latestModel,
    report: input.report,
    explanations: input.explanations ?? new Map(),
  };
}

export function isTerminal(status: string): boolean {
  return (TERMINAL_STATUSES as readonly string[]).includes(status);
}

function payloadDay(event: EventRecord): number | null {
  const day = (event.payload as Record<string, unknown> | undefined)?.day;
  return typeof day === "number" ? day : null;
}

/**
 * Events carry no day of their own, so each one inherits the day of the nearest
 * `decision.committed` at or after it — the week whose work it belongs to.
 */
export function annotateEvents(events: EventRecord[], currentDay: number): AnnotatedEvent[] {
  const ordered = [...events].sort((left, right) => (left.sequence ?? 0) - (right.sequence ?? 0));
  const annotated: AnnotatedEvent[] = new Array(ordered.length);
  let carried = currentDay;
  for (let index = ordered.length - 1; index >= 0; index -= 1) {
    const event = ordered[index];
    const own = payloadDay(event);
    if (event.type === "decision.committed" && own !== null) carried = own;
    const effectiveDay = event.type === "run.created" ? 0 : (own ?? carried);
    annotated[index] = { ...event, effectiveDay };
  }
  return annotated;
}

/** The day a decision's week closed, preferring the committed event over the observation clock. */
export function commitDay(data: ReplayData, decision: DecisionRecord): number {
  const committed = data.events.find(
    (event) =>
      event.type === "decision.committed" &&
      (event.payload as Record<string, unknown> | undefined)?.decision_id === decision.id,
  );
  const own = committed ? payloadDay(committed) : null;
  if (own !== null) return own;
  if (decision.actual_outcome) return decision.actual_outcome.day;
  return decision.observation.day + 7;
}

export function eventsUpToDay(data: ReplayData, day: number): AnnotatedEvent[] {
  return data.events.filter((event) => event.effectiveDay <= day);
}

export function decisionsUpToDay(data: ReplayData, day: number): DecisionRecord[] {
  return data.decisions
    .filter((decision) => commitDay(data, decision) <= day)
    .sort((left, right) => left.week - right.week);
}

export function activeDecision(data: ReplayData, day: number): DecisionRecord | null {
  return decisionsUpToDay(data, day).at(-1) ?? null;
}

/** Every observation the run has published, in day order, deduplicated by day. */
export function observationTimeline(data: ReplayData): Observation[] {
  const byDay = new Map<number, Observation>();
  for (const decision of [...data.decisions].sort((left, right) => left.week - right.week)) {
    byDay.set(decision.observation.day, decision.observation);
    if (decision.actual_outcome) byDay.set(decision.actual_outcome.day, decision.actual_outcome);
  }
  return [...byDay.values()].sort((left, right) => left.day - right.day);
}

export function companyStateAtDay(data: ReplayData, day: number): CompanyState | null {
  const timeline = observationTimeline(data);
  const index = timeline.findLastIndex((observation) => observation.day <= day);
  if (index < 0) return null;
  const observation = timeline[index];
  const previous = index > 0 ? timeline[index - 1] : null;

  const customers = metric(observation, METRIC_NAMES.customers);
  const previousCustomers = previous ? metric(previous, METRIC_NAMES.customers) : null;
  const cashDelta = previous ? observation.cash - previous.cash : null;

  return {
    observation,
    cash: observation.cash,
    revenue: metric(observation, METRIC_NAMES.revenue),
    customers,
    churn: metric(observation, METRIC_NAMES.churn),
    runwayWeeks: cashDelta !== null && cashDelta < 0 ? observation.cash / -cashDelta : null,
    growth:
      customers !== null && previousCustomers ? (customers - previousCustomers) / previousCustomers : null,
  };
}

export function modelVersionAtDay(data: ReplayData, day: number): ModelAtDay {
  const committed = data.decisions.filter(
    (decision) => decision.status !== "prepared" && commitDay(data, decision) <= day,
  );
  // Before the first week closes there is genuinely no model — never borrow a later one.
  if (committed.length === 0) return { model: null, exact: true };

  let best: { day: number; model: WorldModelVersion } | null = null;
  for (const decision of committed) {
    const explanation = decision.id ? data.explanations.get(decision.id) : undefined;
    if (!explanation) continue;
    const closed = commitDay(data, decision);
    if (!best || closed >= best.day) best = { day: closed, model: explanation.world_model };
  }
  if (best) return { model: best.model, exact: true };
  return { model: data.latestModel, exact: false };
}

/** Health signals from every source the API exposes, deduplicated and ordered by evaluation day. */
export function healthTimeline(data: ReplayData): ModelHealthSignal[] {
  const byId = new Map<string, ModelHealthSignal>();
  for (const explanation of data.explanations.values()) {
    for (const signal of explanation.model_health_signals) byId.set(signal.id ?? "", signal);
  }
  if (data.report?.latest_model_health) {
    const signal = data.report.latest_model_health;
    byId.set(signal.id ?? "", signal);
  }
  return [...byId.values()].sort((left, right) => left.evaluated_day - right.evaluated_day);
}

export function healthAtDay(data: ReplayData, day: number): ModelHealthSignal | null {
  return healthTimeline(data).findLast((signal) => signal.evaluated_day <= day) ?? null;
}

export function predictionMarkersAtDay(data: ReplayData, day: number): PredictionMarker[] {
  const markers: PredictionMarker[] = [];
  for (const view of data.predictions) {
    const entry = view.prediction;
    if (entry.issued_day > day) continue;
    const outcomes = new Map(view.outcomes.map((outcome) => [outcome.target_id, outcome]));
    for (const target of entry.targets) {
      const outcome = outcomes.get(target.id ?? "");
      const matured = day >= target.target_day;
      const state: MarkerState = !matured
        ? "pending"
        : outcome
          ? outcome.score.interval_hit
            ? "hit"
            : "miss"
          : "awaiting";
      markers.push({
        key: `${entry.id}-${target.id}`,
        targetDay: target.target_day,
        horizonDays: target.horizon_days,
        issuedDay: entry.issued_day,
        decisionWeek: entry.decision_week,
        point: target.point,
        lower: target.lower,
        upper: target.upper,
        actualCash: matured && outcome ? outcome.actual.cash : null,
        state,
      });
    }
  }
  return markers.sort((left, right) => left.targetDay - right.targetDay);
}

/** Ledger verdict for one decision as of a day: every matured forecast, collapsed to one badge. */
export function decisionVerdict(
  data: ReplayData,
  decision: DecisionRecord,
  day: number,
): { state: MarkerState; matured: number; total: number } {
  const view = data.predictions.find((item) => item.prediction.decision_id === decision.id);
  if (!view) return { state: "pending", matured: 0, total: 0 };
  const markers = predictionMarkersAtDay(data, day).filter(
    (marker) => marker.decisionWeek === view.prediction.decision_week,
  );
  const matured = markers.filter((marker) => marker.state === "hit" || marker.state === "miss");
  const state: MarkerState = matured.some((marker) => marker.state === "miss")
    ? "miss"
    : matured.length > 0
      ? "hit"
      : markers.some((marker) => marker.state === "awaiting")
        ? "awaiting"
        : "pending";
  return { state, matured: matured.length, total: view.prediction.targets.length };
}

export function averageConfidence(model: WorldModelVersion | null): number | null {
  if (!model?.parameters.length) return null;
  return model.parameters.reduce((sum, parameter) => sum + parameter.confidence, 0) / model.parameters.length;
}
