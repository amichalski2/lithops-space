import { useCallback, useEffect, useRef, useState } from "react";

/** Playback rates in simulated days per real second, shown to the operator as 1x/2x/4x. */
export const SPEEDS = [7, 14, 28] as const;
export type Speed = (typeof SPEEDS)[number];

/** Guards against a single huge frame delta after the browser throttles a background tab. */
const MAX_FRAME_SECONDS = 0.1;

export type ReplayClock = {
  /** Fractional day, used for smooth motion of the playhead and logo. */
  dayFloat: number;
  /** Whole day every panel selector is keyed on. */
  currentDay: number;
  playing: boolean;
  speed: Speed;
  atEnd: boolean;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (day: number) => void;
  nudge: (days: number) => void;
  cycleSpeed: () => void;
};

export function useReplayClock(maxDay: number): ReplayClock {
  const [dayFloat, setDayFloat] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(SPEEDS[0]);
  const frame = useRef<number | null>(null);
  const previous = useRef<number | null>(null);
  const limit = Math.max(0, maxDay);

  const clamp = useCallback((day: number) => Math.min(limit, Math.max(0, day)), [limit]);

  useEffect(() => {
    if (!playing) {
      previous.current = null;
      return;
    }
    const step = (timestamp: number) => {
      const last = previous.current;
      previous.current = timestamp;
      if (last !== null) {
        const elapsed = Math.min(MAX_FRAME_SECONDS, (timestamp - last) / 1000);
        setDayFloat((current) => {
          const next = current + elapsed * speed;
          if (next >= limit) {
            setPlaying(false);
            return limit;
          }
          return next;
        });
      }
      frame.current = requestAnimationFrame(step);
    };
    frame.current = requestAnimationFrame(step);
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      frame.current = null;
      previous.current = null;
    };
  }, [playing, speed, limit]);

  const seek = useCallback((day: number) => setDayFloat(clamp(day)), [clamp]);

  const play = useCallback(() => {
    setDayFloat((current) => (current >= limit ? 0 : current));
    setPlaying(true);
  }, [limit]);

  return {
    dayFloat,
    currentDay: Math.floor(dayFloat),
    playing,
    speed,
    atEnd: dayFloat >= limit,
    play,
    pause: useCallback(() => setPlaying(false), []),
    toggle: useCallback(() => (playing ? setPlaying(false) : play()), [playing, play]),
    seek,
    nudge: useCallback((days: number) => setDayFloat((current) => clamp(Math.round(current) + days)), [clamp]),
    cycleSpeed: useCallback(
      () => setSpeed((current) => SPEEDS[(SPEEDS.indexOf(current) + 1) % SPEEDS.length]),
      [],
    ),
  };
}
