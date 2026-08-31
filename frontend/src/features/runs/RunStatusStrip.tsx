import { useReplay } from "../cockpit/ReplayProvider";
import { ErrorNotice } from "../../components/ui/ErrorNotice";
import { compactMoney } from "../../lib/format";

/** Empty, paused, failed and terminal states the PRD requires the cockpit to make explicit. */
export function RunStatusStrip() {
  const { data, notice, dismissNotice } = useReplay();
  const { run, report } = data;

  return (
    <>
      {notice && <ErrorNotice message={notice} onRetry={dismissNotice} />}

      {run.status === "failed" && (
        <div className="failure-state">
          <strong>Run failed at safe checkpoint</strong>
          <span>{run.failure_reason ?? "Unknown failure"}</span>
        </div>
      )}

      {(run.status === "paused" || run.status === "pausing") && (
        <div className="pause-state">
          Paused safely · checkpoint day {run.current_day} · no partial action batch
        </div>
      )}

      {(run.status === "completed" || run.status === "bankrupt") && (
        <div className={run.status === "bankrupt" ? "failure-state" : "terminal-state"}>
          <strong>{run.status === "bankrupt" ? "Bankrupt" : "Run complete"}</strong>
          <span>
            Day {run.current_day} · {report?.decision_count ?? 0} decisions ·{" "}
            {report?.matured_outcome_count ?? 0} matured forecasts
            {data.decisions.at(-1)?.actual_outcome
              ? ` · final cash ${compactMoney(data.decisions.at(-1)!.actual_outcome!.cash)}`
              : ""}
          </span>
        </div>
      )}
    </>
  );
}
