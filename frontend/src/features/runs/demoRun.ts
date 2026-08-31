import { STORED_RUN_KEY } from "../cockpit/ReplayProvider";

/** The configured demo run wins, so a first-time visitor always lands on a run worth replaying. */
export function resolveReplayRunId(): string | null {
  const runtime = window.__LITHOPS_CONFIG__?.demoRunId?.trim();
  if (runtime) return runtime;
  const configured = import.meta.env.VITE_DEMO_RUN_ID?.trim();
  if (configured) return configured;
  try {
    return localStorage.getItem(STORED_RUN_KEY);
  } catch {
    return null;
  }
}
