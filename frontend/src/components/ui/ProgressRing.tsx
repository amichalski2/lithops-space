import type { ReactNode } from "react";

const RADIUS = 96;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

/**
 * The run's position in its horizon, drawn as an arc around the mark. The dashed track doubles as
 * the tick scale, so progress needs no separate legend; `progress` is clamped because a resumed
 * run can report a day past its own horizon.
 */
export function ProgressRing({
  progress,
  label,
  children,
}: {
  progress: number;
  label?: string;
  children: ReactNode;
}) {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(progress) ? progress : 0));

  return (
    <div className="stage-orb">
      <svg
        className="orb-ring"
        viewBox="0 0 220 220"
        {...(label ? { role: "img", "aria-label": label } : { "aria-hidden": true })}
      >
        <circle className="orb-track" cx="110" cy="110" r={RADIUS} />
        <circle
          className="orb-progress"
          cx="110"
          cy="110"
          r={RADIUS}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={CIRCUMFERENCE * (1 - clamped)}
        />
      </svg>
      {children}
    </div>
  );
}
