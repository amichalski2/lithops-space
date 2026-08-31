import { HudIcon, type IconName } from "../../components/ui/HudIcon";

export type VerdictState = "hit" | "miss" | "open" | "pending" | "awaiting";

export const VERDICT_LABEL: Record<VerdictState, string> = {
  hit: "Hit",
  miss: "Miss",
  open: "Open",
  awaiting: "Open",
  pending: "Open",
};

/** Only a settled verdict earns an arrow — an open one would be claiming a direction it lacks. */
const VERDICT_ICON: Partial<Record<VerdictState, IconName>> = { hit: "trendUp", miss: "trendDown" };

/**
 * One pill for every graded thing, so a forecast reads the same on the cockpit ledger, on the
 * decision page and in a receipt list. `tone` exists because receipts are graded too, but on a
 * different axis: executed or unconfirmed rather than hit or missed.
 */
export function Verdict({ state, label, tone }: { state: VerdictState; label?: string; tone?: string }) {
  // The arrow means "the forecast moved this way". A caller supplying its own label is grading
  // something else, so it gets the pill without a direction it never had.
  const icon = label ? undefined : VERDICT_ICON[state];
  return (
    <i className={`verdict verdict-${tone ?? state}`}>
      {label ?? VERDICT_LABEL[state]}
      {icon && <HudIcon name={icon} />}
    </i>
  );
}
