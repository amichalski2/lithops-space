import { useReplay } from "./ReplayProvider";
import { Metric } from "../../components/ui/Metric";
import { Sparkline } from "../../components/ui/Sparkline";
import {
  METRIC_NAMES,
  compactMoney,
  formatOptionalMetric,
  metric,
  percent,
  signed,
} from "../../lib/format";
import { companyStateAtDay, observationTimeline } from "../../lib/replay";

export function CompanyStatusPanel() {
  const { data, clock } = useReplay();
  const state = companyStateAtDay(data, clock.currentDay);

  // The growth tile shows the series it is derived from, so a single week's number is readable
  // against the run's own trend rather than in isolation.
  const customerTrend = observationTimeline(data)
    .filter((observation) => observation.day <= clock.currentDay)
    .map((observation) => metric(observation, METRIC_NAMES.customers))
    .filter((value): value is number => value !== null);

  return (
    <section className="side-panel company-panel" aria-label="Company state" data-tour="company">
      <header>
        <p className="eyebrow">Company overview</p>
        <span>{state ? `Observed D${state.observation.day}` : "Awaiting first observation"}</span>
      </header>
      <div className="metric-stack">
        <Metric
          icon="cash"
          label="Cash balance"
          value={state ? compactMoney(state.cash) : "—"}
          note={state ? `D${state.observation.day}` : "No obs"}
          tone={state && state.cash <= 0 ? "danger" : undefined}
        />
        <Metric
          icon="revenue"
          label="Weekly revenue"
          value={formatOptionalMetric(state?.revenue ?? null, "money")}
          note="Weekly"
        />
        <Metric
          icon="customers"
          label="Active customers"
          value={formatOptionalMetric(state?.customers ?? null)}
          note="Active"
        />
        <Metric
          icon="churn"
          label="Churn rate"
          value={formatOptionalMetric(state?.churn ?? null, "percent")}
          note="Weekly"
          tone={state?.churn != null && state.churn > 0.045 ? "warning" : undefined}
        />
        <Metric
          icon="burn"
          label="Runway"
          value={state?.runwayWeeks == null ? "Stable" : `${state.runwayWeeks.toFixed(1)} W`}
          note={state?.runwayWeeks == null ? "Not falling" : "At burn"}
          tone={state?.runwayWeeks != null && state.runwayWeeks < 12 ? "danger" : undefined}
        />
        <Metric
          icon="growth"
          label="Growth trend"
          value={state?.growth == null ? "—" : signed(state.growth, (value) => percent(value, 1))}
          note="W/W"
          aside={
            customerTrend.length > 1 ? (
              <Sparkline values={customerTrend} label="Active customers over the run so far" />
            ) : undefined
          }
          tone={state?.growth != null && state.growth < 0 ? "warning" : undefined}
        />
      </div>
    </section>
  );
}
