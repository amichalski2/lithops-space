import type { DecisionRecord } from "../../api/client";
import { Verdict } from "./verdict";
import type { AnnotatedEvent } from "../../lib/replay";

/**
 * Receipts are only observable through the action gateway's events, so each planned command is
 * matched to its receipt by the idempotency key that protected it from double execution.
 */
export function ActionReceipts({
  decision,
  events,
}: {
  decision: DecisionRecord;
  events: AnnotatedEvent[];
}) {
  const receipts = new Map(
    events
      .filter((event) => event.type === "action.executed")
      .map((event) => event.payload as Record<string, unknown>)
      .filter((payload) => payload.decision_id === decision.id)
      .map((payload) => [String(payload.idempotency_key), payload]),
  );

  return (
    <section className="panel receipts-panel" aria-label="Action receipts">
      <header className="panel-heading">
        <div>
          <p className="eyebrow">Action gateway</p>
          <h2>What actually executed</h2>
        </div>
        <span className="chip">
          {receipts.size} / {decision.action_plan.commands.length} receipts
        </span>
      </header>

      <ol className="receipt-list">
        {decision.action_plan.commands.map((command) => {
          const receipt = receipts.get(command.idempotency_key);
          return (
            <li key={command.idempotency_key} className={receipt ? "is-executed" : "is-unconfirmed"}>
              <strong>{command.tool}</strong>
              <code>{JSON.stringify(command.arguments)}</code>
              <small>
                key {command.idempotency_key}
                {receipt ? ` · receipt ${String(receipt.receipt_id).slice(0, 12)}` : ""}
              </small>
              {/* A receipt is graded too, just on a different axis than a forecast: it either
                  reached the gateway or it did not. Missing is a warning, never a miss. */}
              <Verdict
                state={receipt ? "hit" : "open"}
                tone={receipt ? "hit" : "warning"}
                label={receipt ? "Executed" : "No receipt"}
              />
            </li>
          );
        })}
      </ol>
    </section>
  );
}
