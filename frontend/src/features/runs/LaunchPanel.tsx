import { ErrorNotice } from "../../components/ui/ErrorNotice";
import { ByokPlaceholder } from "./ByokPlaceholder";

const LOOP = ["Observe", "Model", "Simulate", "Act", "Evaluate", "Recalibrate"];

export function LaunchPanel({
  replayRunId,
  busy,
  error,
  onReplay,
  onCreate,
}: {
  replayRunId: string | null;
  busy: boolean;
  error: string | null;
  onReplay: () => void;
  onCreate: (geminiApiKey: string) => void;
}) {
  return (
    <section className="launch-panel">
      <div className="launch-index">01 — Operating loop</div>

      <div className="launch-body">
        <p className="eyebrow">Autonomous company operator</p>
        <h1>
          Learn the business.
          <br />
          <em>Act on reality.</em>
        </h1>
        <p className="lede">
          Lithops builds an uncertain model of a company, compares robust strategies, commits
          predictions before acting, and learns from every miss.
        </p>

        <div className="launch-actions">
          {replayRunId ? (
            <button className="primary" onClick={onReplay}>
              Replay best CEO-Bench run
            </button>
          ) : (
            <p className="lede">Enter your Gemini key below to create a fresh cloud simulation.</p>
          )}
        </div>

        {error && <ErrorNotice message={error} />}
        <ByokPlaceholder busy={busy} onCreate={onCreate} />
      </div>

      <div className="loop-rail" aria-label="Lithops operating loop">
        {LOOP.map((step, index) => (
          <span key={step}>
            <b>{String(index + 1).padStart(2, "0")}</b>
            {step}
          </span>
        ))}
      </div>
    </section>
  );
}
