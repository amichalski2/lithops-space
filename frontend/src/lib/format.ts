import type { DecisionRecord } from "../api/client";

export type Observation = DecisionRecord["observation"];

/** Metric names the CEO-Bench observation dictionary is known to use, by concept. */
export const METRIC_NAMES = {
  revenue: ["weekly_revenue", "revenue", "mrr"],
  customers: ["active_customers", "customers"],
  churn: ["weekly_churn", "churn", "churn_rate"],
  spend: ["weekly_spend", "operating_spend", "total_spend"],
} as const;

export function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

/** Maps a run's backend status onto the four tones the status mark is drawn in. */
export function statusTone(status: string): "positive" | "danger" | "warning" | "neutral" {
  if (status === "failed" || status === "bankrupt") return "danger";
  if (status === "paused" || status === "pausing") return "warning";
  if (status === "running" || status === "completed") return "positive";
  return "neutral";
}

/**
 * First letter up, the rest left alone. Backend states arrive lowercase and snake_cased
 * (`model_health`, `bankrupt`), and the interface shows them as prose rather than as shouting.
 */
export function sentenceCase(value: string): string {
  const spaced = value.replaceAll("_", " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function compactMoney(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: Math.abs(value) >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: Math.abs(value) >= 10_000 ? 1 : 0,
  }).format(value);
}

export function preciseMoney(value: number): string {
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export function percent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function signed(value: number, format: (input: number) => string): string {
  return `${value >= 0 ? "+" : ""}${format(value)}`;
}

/** Reads the first numeric match from the open-ended observation metric dictionary. */
export function metric(observation: Observation | null, names: readonly string[]): number | null {
  if (!observation) return null;
  for (const name of names) {
    const value = observation.metrics?.[name];
    if (typeof value === "number") return value;
  }
  return null;
}

export function formatOptionalMetric(
  value: number | null,
  kind: "number" | "money" | "percent" = "number",
): string {
  if (value === null) return "—";
  if (kind === "money") return compactMoney(value);
  if (kind === "percent") return percent(value, 1);
  return Math.round(value).toLocaleString("en-US");
}

export function padDay(day: number): string {
  return String(Math.max(0, Math.round(day))).padStart(3, "0");
}
