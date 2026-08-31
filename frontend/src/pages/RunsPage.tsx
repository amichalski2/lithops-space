import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { lithopsApi } from "../api/client";
import { resolveReplayRunId } from "../features/runs/demoRun";
import { setGeminiApiKey } from "../features/runs/byok";
import { LaunchPanel } from "../features/runs/LaunchPanel";

export function RunsPage() {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [replayRunId] = useState(resolveReplayRunId);

  function openReplay() {
    if (replayRunId) navigate(`/runs/${replayRunId}`, { state: { autoplay: true } });
  }

  async function createRun(geminiApiKey: string) {
    setBusy(true);
    setError(null);
    try {
      setGeminiApiKey(geminiApiKey);
      const created = await lithopsApi.createRun();
      if (!created.id) throw new Error("Lithops API returned a run without an id");
      navigate(`/runs/${created.id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create a fresh run");
    } finally {
      setBusy(false);
    }
  }

  return (
    <LaunchPanel
      replayRunId={replayRunId}
      busy={busy}
      error={error}
      onReplay={openReplay}
      onCreate={createRun}
    />
  );
}
