import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { lithopsApi, type DecisionExplanation } from "../api/client";
import { LoadingState } from "../components/ui/LoadingState";
import { HudBadge, parameterIcon } from "../components/ui/HudIcon";
import { PanelHeading } from "../components/ui/PanelHeading";
import { StatusIcon } from "../components/ui/StatusIcon";
import { useReplay } from "../features/cockpit/ReplayProvider";
import { ActionReceipts } from "../features/decisions/ActionReceipts";
import { CandidateTable } from "../features/decisions/CandidateTable";
import { ForecastVsActual } from "../features/decisions/ForecastVsActual";
import { compactMoney, percent, sentenceCase, statusTone, titleCase } from "../lib/format";
import { commitDay } from "../lib/replay";

export function DecisionPage() {
  const { runId = "", decisionId = "" } = useParams();
  const { data, explanationFor } = useReplay();
  const cached = explanationFor(decisionId);
  const [fetched, setFetched] = useState<DecisionExplanation | null>(null);
  const [resolved, setResolved] = useState(false);

  const record = data.decisions.find((decision) => decision.id === decisionId) ?? null;
  const explanation = cached ?? fetched;

  useEffect(() => {
    if (cached || !decisionId) return;
    let cancelled = false;
    setResolved(false);
    void lithopsApi
      .getDecisionIfExplained(runId, decisionId)
      .then((result) => {
        if (cancelled) return;
        setFetched(result);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setResolved(true);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, decisionId, cached]);

  const back = (
    <Link className="back-link" to={`/runs/${runId}`}>
      ← Back to cockpit
    </Link>
  );

  if (!record) {
    return (
      <section className="hud decision-page">
        {back}
        <p className="empty">This decision is not part of run {runId.slice(0, 8)}.</p>
      </section>
    );
  }

  if (!explanation) {
    if (!resolved) return <LoadingState label="Loading decision" message="Reading decision evidence" />;
    return (
      <section className="hud decision-page">
        {back}
        <PanelHeading index="Decision" title={`Week ${record.week + 1} · artifacts pending`} />
        <p className="empty">
          This decision is still <strong>{record.status}</strong>. Lithops publishes the model
          version, forecasts and receipts only once the week commits, so there is nothing to audit
          yet.
        </p>
      </section>
    );
  }

  const { decision, world_model: model, prediction, model_health_signals: signals } = explanation;
  const health = signals.at(-1) ?? null;

  return (
    <section className="hud decision-page">
      {back}

      <header className="decision-head">
        <div>
          <p className="eyebrow">Week {decision.week + 1} · Day {commitDay(data, decision)}</p>
          <h1>{titleCase(decision.action_plan.name)}</h1>
          <p className="lede">{decision.selection_reason ?? decision.action_plan.rationale}</p>
          <code className="reason-code">{decision.selection_reason_code ?? "ROBUST_SELECTION"}</code>
        </div>
        <dl className="decision-facts">
          <div>
            <dt>Status</dt>
            <dd>{sentenceCase(decision.status)}</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>v{model.version}</dd>
          </div>
          <div>
            <dt>Prompt</dt>
            <dd>{decision.prompt_version ?? "—"}</dd>
          </div>
          <div>
            <dt>Rollouts</dt>
            <dd>{decision.candidate_evaluations?.[0]?.rollout_count.toLocaleString() ?? "—"}</dd>
          </div>
        </dl>
      </header>

      <CandidateTable
        candidates={decision.candidate_evaluations ?? []}
        selectedStrategy={decision.action_plan.strategy_family}
      />

      <div className="decision-columns">
        <ForecastVsActual view={prediction} currentDay={data.run.current_day} />
        <ActionReceipts decision={decision} events={data.events} />
      </div>

      <section className="panel model-diff" aria-label="Model version used">
        <PanelHeading index="World model" title={`Version ${model.version}`} aside={titleCase(model.update_method)} />
        <div className="parameter-list">
          {model.parameters.map((parameter) => (
            <article key={parameter.name}>
              <HudBadge name={parameterIcon(parameter.name)} />
              <div>
                <strong>{titleCase(parameter.name)}</strong>
                <small>
                  {percent(parameter.confidence)} confidence · evidence{" "}
                  {parameter.evidence.slice(0, 3).map((item) => item.reference).join(", ")}
                  {parameter.evidence.length > 3 ? ` +${parameter.evidence.length - 3} more` : ""}
                </small>
              </div>
              <span>{parameter.estimate.toFixed(2)}</span>
              <div className="confidence-track">
                <i style={{ width: `${parameter.confidence * 100}%` }} />
              </div>
            </article>
          ))}
        </div>
        {model.changes.length > 0 && (
          <div className="model-changes">
            <span>Changed in this version</span>
            {model.changes.map((change) => (
              <p key={change.parameter_name}>
                <strong>{titleCase(change.parameter_name)}</strong>
                <b>
                  {change.previous_estimate.toFixed(2)} → {change.new_estimate.toFixed(2)}
                </b>
                <small>
                  {percent(change.previous_confidence)} → {percent(change.new_confidence)} confidence ·{" "}
                  {change.evidence.map((item) => item.reference).join(", ")}
                </small>
              </p>
            ))}
          </div>
        )}
        {health && (
          <p className="health-line">
            <StatusIcon tone={statusTone(health.status)} />
            Model health at evaluation: <strong>{sentenceCase(health.status)}</strong> ·{" "}
            {health.interval_miss_count} interval misses · bias{" "}
            {compactMoney(health.directional_bias)}
            {health.rebuild_recommended ? " · Rebuild recommended" : ""}
          </p>
        )}
      </section>
    </section>
  );
}
