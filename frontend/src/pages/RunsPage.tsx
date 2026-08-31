import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { resolveReplayRunId } from "../features/runs/demoRun";
import { LaunchPanel } from "../features/runs/LaunchPanel";

export function RunsPage() {
  const navigate = useNavigate();
  const [replayRunId] = useState(resolveReplayRunId);

  function openReplay() {
    if (replayRunId) navigate(`/runs/${replayRunId}`, { state: { autoplay: true } });
  }

  return <LaunchPanel replayRunId={replayRunId} onReplay={openReplay} />;
}
