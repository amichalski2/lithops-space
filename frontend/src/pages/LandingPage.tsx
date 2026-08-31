import { useNavigate } from "react-router-dom";

import { LandingHero } from "../features/landing/LandingHero";
import { resolveReplayRunId } from "../features/runs/demoRun";

export function LandingPage() {
  const navigate = useNavigate();
  const busy = false;
  const error: string | null = null;
  const replayRunId = resolveReplayRunId();

  /**
   * One CTA, two paths: replay the run we already have, or mint one when there is none. The
   * replay opens paused with the guided tour up — dropping a first-time visitor into a moving
   * clock spends the one moment they are willing to be taught.
   */
  function runSimulation() {
    if (replayRunId) navigate(`/runs/${replayRunId}`, { state: { tour: true } });
    else navigate("/launch");
  }

  return (
    <LandingHero
      hasDemoRun={replayRunId !== null}
      busy={busy}
      error={error}
      onRun={runSimulation}
      onCreate={() => navigate("/launch")}
    />
  );
}
