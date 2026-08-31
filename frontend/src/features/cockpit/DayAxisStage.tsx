import { TransportControls } from "./TransportControls";
import { useReplay } from "./ReplayProvider";
import { DayAxis } from "../predictions/DayAxis";
import { LithopsMark } from "../../components/ui/LithopsMark";
import { ProgressRing } from "../../components/ui/ProgressRing";
import { StatusIcon } from "../../components/ui/StatusIcon";
import { padDay, percent, sentenceCase, statusTone } from "../../lib/format";
import { averageConfidence, healthAtDay, modelVersionAtDay } from "../../lib/replay";

export function DayAxisStage() {
  const { data, clock } = useReplay();
  const { model } = modelVersionAtDay(data, clock.currentDay);
  const health = healthAtDay(data, clock.currentDay);
  const confidence = averageConfidence(model);
  const horizon = data.run.horizon_days;

  return (
    <section className="stage" aria-label="Run clock" data-tour="stage">
      <div className="stage-core">
        <p className="stage-status">
          <StatusIcon tone={statusTone(data.run.status)} />
          {sentenceCase(data.run.status)}
          {model ? ` · Model v${model.version}` : " · Bootstrapping"}
          {health ? ` · ${sentenceCase(health.status)}` : ""}
        </p>

        <h1 className="stage-day">
          <span key={clock.currentDay} className="day-value">
            Day <em>{padDay(clock.currentDay)}</em>
          </span>
          <small>/ {horizon}</small>
        </h1>

        {/* Driven by the fractional day, not the whole one: the clock already advances it every
            animation frame, so the arc moves at the real playback rate at every speed. */}
        <ProgressRing
          progress={horizon > 0 ? clock.dayFloat / horizon : 0}
          label={`Day ${clock.currentDay} of ${horizon}`}
        >
          <div className={clock.playing ? "stage-mark is-flying" : "stage-mark"}>
            <LithopsMark className="mark" />
          </div>
        </ProgressRing>

        <p className="stage-confidence">
          {confidence === null ? (
            "No model confidence yet"
          ) : (
            <>
              <em>{percent(confidence)}</em> mean model confidence
            </>
          )}
        </p>
      </div>

      <DayAxis />
      <TransportControls />
    </section>
  );
}
