import { useReplay } from "./ReplayProvider";
import { StatusIcon } from "../../components/ui/StatusIcon";
import { isTerminal } from "../../lib/replay";

/** Judges must never mistake a stored replay for a live run; this badge is always visible. */
export function LiveBadge() {
  const { data, live, goLive } = useReplay();
  const terminal = isTerminal(data.run.status);

  if (live && !terminal) {
    return (
      <span className="mode-badge mode-live">
        <StatusIcon />
        Live
      </span>
    );
  }

  return (
    <span className="mode-badge mode-replay">
      Replay · {data.run.id?.slice(0, 8) ?? "—"}
      {!terminal && (
        <button onClick={goLive} className="go-live">
          Go live →
        </button>
      )}
    </span>
  );
}
