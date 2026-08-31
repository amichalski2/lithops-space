import { LiveBadge } from "./LiveBadge";
import { useReplay } from "./ReplayProvider";
import { HudIcon } from "../../components/ui/HudIcon";
import { SPEEDS } from "../../hooks/useReplayClock";

export function TransportControls() {
  const { clock } = useReplay();
  const speedLabel = `${SPEEDS.indexOf(clock.speed) === 0 ? 1 : SPEEDS.indexOf(clock.speed) * 2}x`;

  return (
    <div className="transport" data-tour="transport">
      <button className="primary run-button" onClick={clock.toggle}>
        <span className="run-glyph">
          <HudIcon name={clock.playing ? "pause" : clock.atEnd ? "replay" : "play"} />
        </span>
        {clock.playing ? "Pause" : clock.atEnd ? "Replay from day 0" : "Run simulation"}
      </button>

      <button className="ghost" onClick={clock.cycleSpeed} aria-label={`Playback speed ${speedLabel}`}>
        {speedLabel}
      </button>

      <LiveBadge />
    </div>
  );
}
