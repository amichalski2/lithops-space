import { Outlet, useLocation, useParams } from "react-router-dom";

import { ReplayProvider } from "../features/cockpit/ReplayProvider";

/** Owns the replay cache so the cockpit and every decision page share one load. */
export function RunLayout() {
  const { runId = "" } = useParams();
  // Arriving from the landing means "show me the whole story", not "show me now".
  const autoplay = (useLocation().state as { autoplay?: boolean } | null)?.autoplay === true;

  return (
    <ReplayProvider runId={runId} autoplay={autoplay}>
      <Outlet />
    </ReplayProvider>
  );
}
