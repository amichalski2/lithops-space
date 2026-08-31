import type { ModelHealthSignal } from "../../api/client";
import { StatusIcon } from "../../components/ui/StatusIcon";
import { percent, titleCase } from "../../lib/format";

/** The degraded-model state of the world-model panel: what tripped, and on what evidence. */
export function ChallengePanel({ health }: { health: ModelHealthSignal }) {
  return (
    <div className="challenge" aria-label="Model challenge">
      <div className="challenge-head">
        <StatusIcon tone="danger" className="challenge-pulse" />
        <div>
          <p>Model error high</p>
          <strong>
            {health.interval_miss_count} interval miss{health.interval_miss_count === 1 ? "" : "es"} ·
            evaluated day {health.evaluated_day}
          </strong>
        </div>
        <b>{health.rebuild_recommended ? "Rebuild recommended" : "Confidence reduced"}</b>
      </div>

      <div className="challenge-triggers">
        {health.trigger_codes.length ? (
          health.trigger_codes.map((code) => <span key={code}>{titleCase(code)}</span>)
        ) : (
          <span>Monitoring residuals</span>
        )}
      </div>

      <table className="horizon-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th>Error</th>
            <th>Coverage</th>
            <th>WIS</th>
          </tr>
        </thead>
        <tbody>
          {health.horizon_performance.map((row) => (
            <tr key={row.horizon_days}>
              <td>D+{row.horizon_days}</td>
              <td>{percent(row.mean_normalized_absolute_error, 1)}</td>
              <td className={row.interval_coverage < 0.9 ? "coverage-low" : undefined}>
                {percent(row.interval_coverage)}
              </td>
              <td>{row.mean_weighted_interval_score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
