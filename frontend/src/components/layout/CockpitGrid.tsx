import type { ReactNode } from "react";

export function CockpitGrid({
  company,
  stage,
  model,
  decisions,
  logs,
}: {
  company: ReactNode;
  stage: ReactNode;
  model: ReactNode;
  decisions: ReactNode;
  logs: ReactNode;
}) {
  return (
    <div className="cockpit-grid">
      <div className="area-company">{company}</div>
      <div className="area-stage">{stage}</div>
      <div className="area-model">{model}</div>
      <div className="area-decisions">{decisions}</div>
      <div className="area-logs">{logs}</div>
    </div>
  );
}
