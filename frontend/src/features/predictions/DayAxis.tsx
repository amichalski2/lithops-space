import { useCallback, useMemo, useRef, type KeyboardEvent, type PointerEvent } from "react";

import { useReplay } from "../cockpit/ReplayProvider";
import { compactMoney } from "../../lib/format";
import { HORIZONS, commitDay, decisionsUpToDay, predictionMarkersAtDay } from "../../lib/replay";

const WIDTH = 1000;
const HEIGHT = 132;
const BASELINE = 116;
const LANE_TOP = 22;
const LANE_GAP = 22;
/** Keeps the horizon labels clear of the day-0 markers. */
const GUTTER = 54;

const laneY = (horizon: number) => LANE_TOP + HORIZONS.indexOf(horizon as 7) * LANE_GAP;

/**
 * A 500-day track would spend most of a run almost empty, so the axis only spans as far as
 * the run has actually reached or predicted, growing towards the full horizon as it goes.
 */
function axisDomain(frontier: number, furthestTarget: number, horizon: number): number {
  const needed = Math.max(frontier, furthestTarget) * 1.06;
  if (needed >= horizon) return Math.max(horizon, furthestTarget);
  return Math.min(horizon, Math.max(91, Math.ceil(needed / 28) * 28));
}

function tickStep(domain: number): number {
  if (domain <= 126) return 7;
  if (domain <= 280) return 14;
  if (domain <= 500) return 28;
  return 56;
}

/**
 * The prediction ledger drawn on the run's own clock: every committed forecast sits at its
 * target day and changes state the moment the playhead reaches it.
 */
export function DayAxis() {
  const { data, clock } = useReplay();
  const horizon = data.run.horizon_days;
  const svg = useRef<SVGSVGElement>(null);

  const markers = predictionMarkersAtDay(data, clock.currentDay);
  const committed = decisionsUpToDay(data, clock.currentDay);
  const frontier = data.run.current_day;

  const domain = useMemo(
    () => axisDomain(frontier, Math.max(0, ...markers.map((marker) => marker.targetDay)), horizon),
    [frontier, markers, horizon],
  );
  const x = useCallback((day: number) => GUTTER + (day / domain) * (WIDTH - GUTTER), [domain]);
  const step = tickStep(domain);

  const seekFromPointer = useCallback(
    (event: PointerEvent<SVGRectElement>) => {
      const rect = svg.current?.getBoundingClientRect();
      if (!rect?.width) return;
      const unit = rect.width / WIDTH;
      const day = ((event.clientX - rect.left) / unit - GUTTER) / (WIDTH - GUTTER);
      clock.seek(Math.round(day * domain));
    },
    [clock, domain],
  );

  const onKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    const step = event.shiftKey ? 7 : 1;
    if (event.key === "ArrowRight") clock.nudge(step);
    else if (event.key === "ArrowLeft") clock.nudge(-step);
    else if (event.key === "Home") clock.seek(0);
    else if (event.key === "End") clock.seek(frontier);
    else if (event.key === " " || event.key === "Enter") clock.toggle();
    else return;
    event.preventDefault();
  };

  return (
    <svg
      ref={svg}
      className="day-axis"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="slider"
      tabIndex={0}
      aria-label="Day scrubber"
      aria-valuemin={0}
      aria-valuemax={frontier}
      aria-valuenow={clock.currentDay}
      aria-valuetext={`Day ${clock.currentDay} of ${horizon}`}
      onKeyDown={onKeyDown}
    >
      {HORIZONS.map((value) => (
        <g key={value}>
          <line className="lane-rule" x1={GUTTER} x2={WIDTH} y1={laneY(value)} y2={laneY(value)} />
          <text className="lane-label" x={0} y={laneY(value) + 3}>
            D+{value}
          </text>
        </g>
      ))}

      {Array.from({ length: Math.floor(domain / step) + 1 }, (_, index) => index * step).map((day) => (
        <g key={day}>
          <line className="tick" x1={x(day)} x2={x(day)} y1={BASELINE - 4} y2={BASELINE + 4} />
          {day % (step * 3) === 0 && (
            <text className="tick-label" x={x(day)} y={HEIGHT - 2}>
              {day}
            </text>
          )}
        </g>
      ))}

      <line className="baseline" x1={GUTTER} x2={WIDTH} y1={BASELINE} y2={BASELINE} />
      <line className="baseline-done" x1={GUTTER} x2={x(clock.dayFloat)} y1={BASELINE} y2={BASELINE} />

      {committed.map((decision) => (
        <line
          key={decision.id}
          className="week-notch"
          x1={x(commitDay(data, decision))}
          x2={x(commitDay(data, decision))}
          y1={BASELINE - 7}
          y2={BASELINE}
        >
          <title>
            Week {decision.week + 1} committed on day {commitDay(data, decision)}
          </title>
        </line>
      ))}

      {markers.map((marker) => {
        const cx = x(marker.targetDay);
        const cy = laneY(marker.horizonDays);
        const label = `D+${marker.horizonDays} from week ${marker.decisionWeek + 1} · predicted ${compactMoney(marker.point)} (${compactMoney(marker.lower)}—${compactMoney(marker.upper)}) for day ${marker.targetDay}${
          marker.actualCash !== null ? ` · actual ${compactMoney(marker.actualCash)}` : ""
        }`;
        return marker.state === "miss" ? (
          <rect
            key={marker.key}
            className="marker marker-miss"
            x={cx - 4}
            y={cy - 4}
            width={8}
            height={8}
            transform={`rotate(45 ${cx} ${cy})`}
          >
            <title>{label}</title>
          </rect>
        ) : (
          <circle key={marker.key} className={`marker marker-${marker.state}`} cx={cx} cy={cy} r={4}>
            <title>{label}</title>
          </circle>
        );
      })}

      {frontier < domain && (
        <line className="frontier" x1={x(frontier)} x2={x(frontier)} y1={8} y2={BASELINE} />
      )}

      <g className="playhead" transform={`translate(${x(clock.dayFloat)} 0)`}>
        <line x1={0} x2={0} y1={4} y2={BASELINE} />
        <polygon points="-5,0 5,0 0,7" />
      </g>

      <rect
        className="scrub-surface"
        x={0}
        y={0}
        width={WIDTH}
        height={HEIGHT}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          seekFromPointer(event);
        }}
        onPointerMove={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) seekFromPointer(event);
        }}
        onPointerUp={(event) => event.currentTarget.releasePointerCapture(event.pointerId)}
      />
    </svg>
  );
}
