import type { ReactNode } from "react";

/**
 * The cockpit's line-icon set. These are drawn rather than imported because every one of them
 * renders at 18-20px inside a glass badge: at that size a raster asset smears, while a stroked
 * path stays crisp and inherits `currentColor`, so a badge can tint its icon with one CSS rule.
 * One shared 24x24 grid and a 1.5 stroke keep the whole set optically consistent.
 */
const ICONS = {
  cash: (
    <>
      <circle cx="12" cy="12" r="8.6" />
      <path d="M12 6.9v10.2" />
      <path d="M14.5 9.5a2.4 2.4 0 0 0-2.3-1.6h-.6a2.1 2.1 0 0 0 0 4.2h1.2a2.1 2.1 0 0 1 0 4.2h-.6a2.4 2.4 0 0 1-2.3-1.6" />
    </>
  ),
  revenue: (
    <>
      <path d="M4 19.6h16" />
      <path d="M7.6 19.6v-3.4M12 19.6v-6.8M16.4 19.6v-4.8" />
      <path d="m6.2 10.6 3.9-3.9 3.1 3.1L19 4" />
      <path d="M15.4 4H19v3.6" />
    </>
  ),
  customers: (
    <>
      <circle cx="9.4" cy="8.4" r="3.3" />
      <path d="M3.7 19.3a5.7 5.7 0 0 1 11.4 0" />
      <path d="M16.1 5.5a3.3 3.3 0 0 1 0 5.8" />
      <path d="M17.7 13.7a5.7 5.7 0 0 1 3.6 5.6" />
    </>
  ),
  churn: (
    <>
      <path d="M20.2 11.2A8.2 8.2 0 0 0 5.8 6.8" />
      <path d="M3.8 12.8a8.2 8.2 0 0 0 14.4 4.4" />
      <path d="M9.2 6.6H5.4V2.8" />
      <path d="M14.8 17.4h3.8v3.8" />
    </>
  ),
  burn: <path d="M13.4 2.4 5.6 13.5h5.3l-.9 8.1 8.4-11.4h-5.5z" />,
  growth: (
    <>
      <path d="m3.6 17.4 5.6-5.6 3.4 3.4 7.8-8.4" />
      <path d="M15.6 6.8h4.8v4.8" />
    </>
  ),
  /* World-model parameters ------------------------------------------------ */
  elasticity: (
    <>
      <path d="M5 4.6v14.8" />
      <path d="M20.2 12H9" />
      <path d="m12.8 7.8-4.2 4.2 4.2 4.2" />
    </>
  ),
  marketing: (
    <>
      <path d="M4.6 9.6v4.8a1.6 1.6 0 0 0 1.6 1.6h1.5l1.4 4.6h2.3l-1.3-4.6h1l7.3 3.6V4.4L11.1 8H6.2a1.6 1.6 0 0 0-1.6 1.6z" />
      <path d="M21 10.2v3.6" />
    </>
  ),
  sensitivity: (
    <>
      <circle cx="12" cy="8" r="3.6" />
      <path d="M5.2 20.4a6.8 6.8 0 0 1 13.6 0" />
    </>
  ),
  lag: (
    <>
      <path d="M7 3.2h10M7 20.8h10" />
      <path d="M8.4 3.2v3c0 2 3.6 3.7 3.6 5.8 0-2.1 3.6-3.8 3.6-5.8v-3" />
      <path d="M8.4 20.8v-3c0-2 3.6-3.7 3.6-5.8 0 2.1 3.6 3.8 3.6 5.8v3" />
    </>
  ),
  segment: (
    <>
      <circle cx="12" cy="7.4" r="2.8" />
      <path d="M7.6 14.8a5 5 0 0 1 8.8 0" />
      <circle cx="4.9" cy="11" r="2.1" />
      <path d="M2 18.6a4 4 0 0 1 3.5-3.4" />
      <circle cx="19.1" cy="11" r="2.1" />
      <path d="M22 18.6a4 4 0 0 0-3.5-3.4" />
    </>
  ),
  parameter: (
    <>
      <path d="M4 7.4h8.6M17.4 7.4H20" />
      <circle cx="15" cy="7.4" r="2.1" />
      <path d="M4 16.6h2.6M11.4 16.6H20" />
      <circle cx="9" cy="16.6" r="2.1" />
    </>
  ),
  /* Transport and verdicts ------------------------------------------------ */
  play: <path d="M9.4 6.6 17.4 12l-8 5.4z" fill="currentColor" />,
  pause: <path d="M9.6 6.4v11.2M14.4 6.4v11.2" />,
  replay: (
    <>
      <path d="M20.2 12a8.2 8.2 0 1 1-2.6-6" />
      <path d="M20.6 3.4v4.4h-4.4" />
    </>
  ),
  trendUp: (
    <>
      <path d="m4 16.6 5.4-5.4 3.2 3.2L20 7.4" />
      <path d="M15.6 7.4H20v4.4" />
    </>
  ),
  trendDown: (
    <>
      <path d="m4 7.4 5.4 5.4 3.2-3.2 7.4 7" />
      <path d="M15.6 16.6H20v-4.4" />
    </>
  ),
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof ICONS;

export function HudIcon({ name, className }: { name: IconName; className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      focusable="false"
    >
      {ICONS[name]}
    </svg>
  );
}

/** Wraps an icon in the glass badge the panels use so callers never repeat the markup. */
export function HudBadge({ name, className }: { name: IconName; className?: string }) {
  return (
    <span className={className ? `hud-badge ${className}` : "hud-badge"}>
      <HudIcon name={name} />
    </span>
  );
}

const PARAMETER_ICONS: [RegExp, IconName][] = [
  [/elastic|price/, "elasticity"],
  [/market|saturation|acquisition/, "marketing"],
  [/churn|retention/, "sensitivity"],
  [/lag|quality|delay/, "lag"],
  [/segment|response|cohort/, "segment"],
];

/**
 * World-model parameters are open-ended — the backend can introduce one we have never seen — so
 * the icon is matched on the name and always falls back rather than throwing on an unknown key.
 */
export function parameterIcon(name: string): IconName {
  const normalized = name.toLowerCase();
  return PARAMETER_ICONS.find(([pattern]) => pattern.test(normalized))?.[1] ?? "parameter";
}
