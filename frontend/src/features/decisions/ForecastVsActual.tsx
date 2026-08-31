import type { PredictionView } from "../../api/client";
import { Verdict } from "./verdict";
import { compactMoney, preciseMoney, percent, signed } from "../../lib/format";

export function ForecastVsActual({ view, currentDay }: { view: PredictionView; currentDay: number }) {
  const outcomes = new Map(view.outcomes.map((outcome) => [outcome.target_id, outcome]));

  return (
    <section className="panel forecast-panel" aria-label="Prediction versus actual">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">Ledger</p>
          <h2>Prediction vs reality</h2>
        </div>
        <span className="chip">{percent(view.prediction.confidence)} confidence</span>
      </header>
      <p className="panel-intro">
        Committed on day {view.prediction.issued_day}, before any action reached CEO-Bench.
        Source of uncertainty: {view.prediction.uncertainty_source}.
      </p>

      <div className="forecast-list">
        {view.prediction.targets.map((target) => {
          const outcome = outcomes.get(target.id ?? "");
          const pending = currentDay < target.target_day;
          const state = outcome ? (outcome.score.interval_hit ? "hit" : "miss") : "open";
          return (
            <article key={target.id} className={`forecast-${state}`}>
              <div className="forecast-horizon">
                <span>D+{target.horizon_days}</span>
                <small>Day {target.target_day}</small>
              </div>
              <div>
                <span>Predicted</span>
                <strong>{preciseMoney(target.point)}</strong>
                <small>
                  {compactMoney(target.lower)} — {compactMoney(target.upper)} · 95%
                </small>
              </div>
              <div>
                <span>Actual</span>
                <strong>
                  {outcome ? preciseMoney(outcome.actual.cash) : pending ? "Pending" : "Awaiting score"}
                </strong>
                <small>
                  {outcome
                    ? `${signed(outcome.score.signed_error, compactMoney)} residual · ${percent(outcome.score.normalized_absolute_error, 1)} error`
                    : "Outcome not observed"}
                </small>
              </div>
              <Verdict state={state} />
            </article>
          );
        })}
      </div>

      <details>
        <summary>Assumptions & evidence</summary>
        <ul>
          {view.prediction.assumptions.map((assumption) => (
            <li key={assumption}>{assumption}</li>
          ))}
          {view.prediction.evidence_references.map((reference) => (
            <li key={reference}>
              <code>{reference}</code>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
