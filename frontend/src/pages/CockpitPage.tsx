import { useState } from "react";
import { useLocation } from "react-router-dom";

import { CockpitGrid } from "../components/layout/CockpitGrid";
import { CompanyStatusPanel } from "../features/cockpit/CompanyStatusPanel";
import { DayAxisStage } from "../features/cockpit/DayAxisStage";
import { DecisionsListPanel } from "../features/decisions/DecisionsListPanel";
import { LogsPanel } from "../features/events/LogsPanel";
import { RunStatusStrip } from "../features/runs/RunStatusStrip";
import { CockpitTour } from "../features/tour/CockpitTour";
import { WorldModelPanel } from "../features/world-model/WorldModelPanel";

export function CockpitPage() {
  // Arriving from the landing CTA means "explain this to me". It is an explicit click rather than
  // a first-visit guess, so the tour opens every time that path is taken and never otherwise.
  const requested = (useLocation().state as { tour?: boolean } | null)?.tour === true;
  const [touring, setTouring] = useState(requested);

  return (
    // `hud` opts this view into the shared instrument layer (tokens, glass, pills); `cockpit`
    // adds only what this screen alone has. The landing page carries neither and keeps the gold.
    <div className="hud cockpit">
      <RunStatusStrip />
      <CockpitGrid
        company={<CompanyStatusPanel />}
        stage={<DayAxisStage />}
        model={<WorldModelPanel />}
        decisions={<DecisionsListPanel />}
        logs={<LogsPanel />}
      />
      <button
        type="button"
        className="tour-launch"
        onClick={() => setTouring(true)}
        aria-label="Replay the guided tour"
      >
        ?
      </button>
      {touring && <CockpitTour onClose={() => setTouring(false)} />}
    </div>
  );
}
