import { useNavigate } from "react-router-dom";

import { LandingHero } from "../features/landing/LandingHero";
import { resolveReplayRunId } from "../features/runs/demoRun";

export function LandingPage() {
  const navigate = useNavigate();
  const replayRunId = resolveReplayRunId();

  /**
   * One CTA, one path: replay the finished run. The replay opens paused with the guided tour
   * up — dropping a first-time visitor into a moving clock spends the one moment they are
   * willing to be taught.
   */
  function runSimulation() {
    if (replayRunId) navigate(`/runs/${replayRunId}`, { state: { tour: true } });
    else navigate("/launch");
  }

  return <LandingHero busy={false} error={null} onRun={runSimulation} />;
}
