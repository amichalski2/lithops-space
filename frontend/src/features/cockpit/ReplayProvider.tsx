import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { Link } from "react-router-dom";

import { ApiError, lithopsApi, type DecisionExplanation } from "../../api/client";
import { ErrorNotice } from "../../components/ui/ErrorNotice";
import { LoadingState } from "../../components/ui/LoadingState";
import { useReplayClock, type ReplayClock } from "../../hooks/useReplayClock";
import { buildReplayData, isTerminal, type ReplayData } from "../../lib/replay";
import { getGeminiApiKey } from "../runs/byok";

const POLL_INTERVAL_MS = 5_000;
export const STORED_RUN_KEY = "lithops.active-run";

export type ReplayContextValue = {
  data: ReplayData;
  clock: ReplayClock;
  /** True while the playhead is pinned to the newest committed day. */
  live: boolean;
  goLive: () => void;
  /** True while per-week model versions are still being fetched in the background. */
  syncing: boolean;
  busy: boolean;
  notice: string | null;
  dismissNotice: () => void;
  advanceWeek: () => Promise<void>;
  explanationFor: (decisionId: string) => DecisionExplanation | null;
};

const ReplayContext = createContext<ReplayContextValue | null>(null);

export function useReplay(): ReplayContextValue {
  const value = useContext(ReplayContext);
  if (!value) throw new Error("useReplay must be used inside a ReplayProvider");
  return value;
}

async function loadRun(runId: string, explanations: Map<string, DecisionExplanation>) {
  const [run, events, decisions, predictions, latestModel, report] = await Promise.all([
    lithopsApi.getRun(runId),
    lithopsApi.listEvents(runId),
    lithopsApi.listDecisions(runId),
    lithopsApi.listPredictions(runId),
    lithopsApi.getWorldModel(runId),
    lithopsApi.getReport(runId),
  ]);
  return buildReplayData({ run, events, decisions, predictions, latestModel, report, explanations });
}

export function ReplayProvider({
  runId,
  children,
  autoplay = false,
}: {
  runId: string;
  children: ReactNode;
  autoplay?: boolean;
}) {
  const [data, setData] = useState<ReplayData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [live, setLive] = useState(false);

  const explanations = useRef(new Map<string, DecisionExplanation>());
  const requested = useRef(new Set<string>());
  const initialised = useRef(false);

  const clock = useReplayClock(data?.run.current_day ?? 0);
  const { seek, pause, play } = clock;

  const refresh = useCallback(async () => {
    const next = await loadRun(runId, explanations.current);
    setData(next);
    setError(null);
    // Only a run that actually loaded is worth remembering as this browser's last run.
    try {
      localStorage.setItem(STORED_RUN_KEY, runId);
    } catch {
      // A browser that refuses storage still replays fine; it just will not resume.
    }
    return next;
  }, [runId]);

  /** Only `DecisionExplanation` carries the model version a given week reasoned with. */
  const syncExplanations = useCallback(
    async (source: ReplayData) => {
      const pending = source.decisions.filter(
        (decision) =>
          decision.id &&
          decision.status !== "prepared" &&
          decision.prediction_id &&
          !requested.current.has(decision.id),
      );
      if (pending.length === 0) return;
      setSyncing(true);
      for (const decision of pending) requested.current.add(decision.id!);
      const results = await Promise.allSettled(
        pending.map((decision) => lithopsApi.getDecisionIfExplained(runId, decision.id!)),
      );
      let changed = false;
      results.forEach((result, index) => {
        if (result.status === "fulfilled" && result.value) {
          explanations.current.set(pending[index].id!, result.value);
          changed = true;
        }
      });
      setSyncing(false);
      if (changed) setData((current) => (current ? { ...current, explanations: explanations.current } : current));
    },
    [runId],
  );

  useEffect(() => {
    let cancelled = false;
    explanations.current = new Map();
    requested.current = new Set();
    initialised.current = false;
    setData(null);
    setError(null);
    void (async () => {
      try {
        const next = await loadRun(runId, explanations.current);
        if (cancelled) return;
        setData(next);
        await syncExplanations(next);
      } catch (reason) {
        if (!cancelled) {
          setError(
            reason instanceof ApiError && reason.status === 404
              ? "This run is not on the Lithops API."
              : reason instanceof Error
                ? reason.message
                : "Could not read the run ledger",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId, syncExplanations]);

  /**
   * Positioned in an effect rather than in the loader: the clock only learns its upper bound
   * from the render that publishes the run, so seeking any earlier is clamped away to day 0.
   */
  useEffect(() => {
    if (!data || initialised.current) return;
    initialised.current = true;
    if (autoplay) {
      seek(0);
      play();
      return;
    }
    seek(data.run.current_day);
    setLive(!isTerminal(data.run.status));
  }, [data, seek, play, autoplay]);

  const terminal = data ? isTerminal(data.run.status) : false;

  useEffect(() => {
    if (!data || terminal) return;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const next = await refresh();
          await syncExplanations(next);
        } catch {
          // A transient poll failure keeps the last good replay on screen.
        }
      })();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [data, terminal, refresh, syncExplanations]);

  const currentDay = data?.run.current_day ?? 0;
  useEffect(() => {
    if (live) seek(currentDay);
  }, [live, currentDay, seek]);

  const goLive = useCallback(() => {
    pause();
    setLive(true);
    seek(currentDay);
  }, [pause, seek, currentDay]);

  const controls = useMemo<ReplayClock>(
    () => ({
      ...clock,
      play: () => {
        setLive(false);
        clock.play();
      },
      pause: () => {
        setLive(false);
        clock.pause();
      },
      toggle: () => {
        setLive(false);
        clock.toggle();
      },
      seek: (day: number) => {
        setLive(false);
        clock.seek(day);
      },
      nudge: (days: number) => {
        setLive(false);
        clock.nudge(days);
      },
    }),
    [clock],
  );

  const advanceWeek = useCallback(async () => {
    setBusy(true);
    setNotice(null);
    try {
      await lithopsApi.stepRun(
        runId,
        `cockpit-${crypto.randomUUID()}`,
        getGeminiApiKey(),
      );
      const next = await refresh();
      setLive(true);
      seek(next.run.current_day);
      await syncExplanations(next);
    } catch (reason) {
      setNotice(
        reason instanceof ApiError && reason.status === 409
          ? "An operation is already in progress for this run."
          : reason instanceof Error
            ? reason.message
            : "Could not advance the run",
      );
    } finally {
      setBusy(false);
    }
  }, [runId, refresh, seek, syncExplanations]);

  const value = useMemo<ReplayContextValue | null>(
    () =>
      data
        ? {
            data,
            clock: controls,
            live,
            goLive,
            syncing,
            busy,
            notice,
            dismissNotice: () => setNotice(null),
            advanceWeek,
            explanationFor: (decisionId: string) => data.explanations.get(decisionId) ?? null,
          }
        : null,
    [data, controls, live, goLive, syncing, busy, notice, advanceWeek],
  );

  if (error) {
    return (
      <div className="run-error">
        <ErrorNotice message={error} onRetry={() => void refresh()} />
        <Link className="back-link" to="/">
          ← BACK TO LAUNCH
        </Link>
      </div>
    );
  }
  if (!value) return <LoadingState />;
  return <ReplayContext.Provider value={value}>{children}</ReplayContext.Provider>;
}
