import { ChallengePanel } from "./ChallengePanel";
import { useReplay } from "../cockpit/ReplayProvider";
import { HudBadge, parameterIcon } from "../../components/ui/HudIcon";
import { percent, titleCase } from "../../lib/format";
import { healthAtDay, modelVersionAtDay } from "../../lib/replay";

export function WorldModelPanel() {
  const { data, clock, syncing } = useReplay();
  const { model, exact } = modelVersionAtDay(data, clock.currentDay);
  const health = healthAtDay(data, clock.currentDay);
  const challenged = health?.status === "degraded";

  return (
    <section
      className={challenged ? "side-panel model-panel is-challenged" : "side-panel model-panel"}
      aria-label="World model parameters"
      data-tour="model"
    >
      <header>
        <p className="eyebrow">{challenged ? "World model / challenge" : "World model"}</p>
        <span className="chip">{model ? `v${model.version}${exact ? "" : " · syncing"}` : "Bootstrapping"}</span>
      </header>

      {!model ? (
        <p className="empty">World Model v0 is waiting for its first observation.</p>
      ) : (
        <>
          {challenged && health && <ChallengePanel health={health} />}

          <div className="model-changes">
            <span>Recalibration in this version</span>
            {model.changes.length === 0 ? (
              <p className="empty">No parameter changed — the model still explains reality.</p>
            ) : (
              model.changes.map((change) => (
                <p key={change.parameter_name}>
                  <strong>{titleCase(change.parameter_name)}</strong>
                  <b>
                    {change.previous_estimate.toFixed(2)} → {change.new_estimate.toFixed(2)}
                  </b>
                  <small>
                    {percent(change.previous_confidence)} → {percent(change.new_confidence)} confidence ·{" "}
                    {titleCase(change.update_method)}
                  </small>
                </p>
              ))
            )}
          </div>

          <div className="parameter-list">
            {model.parameters.map((parameter) => (
              <article key={parameter.name}>
                <HudBadge name={parameterIcon(parameter.name)} />
                <div>
                  <strong>{titleCase(parameter.name)}</strong>
                  <small>
                    {percent(parameter.confidence)} confidence · {parameter.lower_bound.toFixed(2)}—
                    {parameter.upper_bound.toFixed(2)} {parameter.unit}
                    {parameter.lag_weeks > 0 ? ` · lag ${parameter.lag_weeks}w` : ""}
                  </small>
                </div>
                <span>{parameter.estimate.toFixed(2)}</span>
                {/* Confidence is stated in the sub-line; the track repeats it as the row's own
                    edge so a low-confidence parameter is spottable without reading the numbers. */}
                <div className="confidence-track">
                  <i style={{ width: `${parameter.confidence * 100}%` }} />
                </div>
              </article>
            ))}
          </div>

          {syncing && !exact && <p className="sync-note">Reconstructing per-week model versions…</p>}
        </>
      )}
    </section>
  );
}
