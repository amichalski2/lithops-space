const LOOP = ["Observe", "Model", "Simulate", "Act", "Evaluate", "Recalibrate"];

export function LaunchPanel({
  replayRunId,
  onReplay,
}: {
  replayRunId: string | null;
  onReplay: () => void;
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
            <p className="lede">No finished run is configured for replay yet.</p>
          )}
        </div>
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
