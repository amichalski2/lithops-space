import { Link } from "react-router-dom";

import { Verdict } from "./verdict";
import { useReplay } from "../cockpit/ReplayProvider";
import { titleCase } from "../../lib/format";
import { activeDecision, commitDay, decisionVerdict, decisionsUpToDay } from "../../lib/replay";

export function DecisionsListPanel() {
  const { data, clock } = useReplay();
  const visible = decisionsUpToDay(data, clock.currentDay);
  const active = activeDecision(data, clock.currentDay);

  return (
    <section className="side-panel decisions-panel" aria-label="Decision list" data-tour="decisions">
      <header>
        <p className="eyebrow">Decisions</p>
        <span>{visible.length} committed</span>
      </header>

      {visible.length === 0 ? (
        <p className="empty">No decision has been committed yet.</p>
      ) : (
        <ol className="decision-list">
          {[...visible].reverse().map((decision) => {
            const verdict = decisionVerdict(data, decision, clock.currentDay);
            return (
              <li key={decision.id} className={decision.id === active?.id ? "is-active" : undefined}>
                <Link to={`/runs/${data.run.id}/decisions/${decision.id}`}>
                  <b>W{decision.week + 1}</b>
                  <div>
                    <strong>{titleCase(decision.action_plan.name)}</strong>
                    <small>
                      Day {commitDay(data, decision)} · {verdict.matured}/{verdict.total} matured
                    </small>
                  </div>
                  <Verdict state={verdict.state} />
                </Link>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
