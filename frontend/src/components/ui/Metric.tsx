import type { ReactNode } from "react";

import { HudBadge, type IconName } from "./HudIcon";

export function Metric({
  label,
  value,
  note,
  tone,
  icon,
  aside,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: "danger" | "warning" | "acid";
  icon?: IconName;
  /** Trailing slot for a trend shape; the note falls back into it when no chart is supplied. */
  aside?: ReactNode;
}) {
  return (
    <article className={tone ? `metric metric-${tone}` : "metric"}>
      {icon && <HudBadge name={icon} />}
      <span>{label}</span>
      <strong>{value}</strong>
      {aside ? <div className="metric-aside">{aside}</div> : <small>{note ?? "Current observation"}</small>}
    </article>
  );
}
