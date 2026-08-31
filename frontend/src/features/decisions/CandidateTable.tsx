import type { CandidateEvaluationRecord } from "../../api/client";
import { compactMoney, percent, signed, titleCase } from "../../lib/format";

/**
 * Robust selection is only credible if the rejected plans stay visible, including the ones
 * with a higher expected value than the plan that won.
 */
export function CandidateTable({
  candidates,
  selectedStrategy,
}: {
  candidates: CandidateEvaluationRecord[];
  selectedStrategy: string;
}) {
  const ceiling = Math.max(1, ...candidates.map((candidate) => candidate.expected_ending_cash));

  return (
    <div className="candidate-table-wrap">
      <table className="strategy-table" aria-label="Simulated candidate strategies">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Outcome distribution</th>
            <th>Expected cash</th>
            <th>Downside p10</th>
            <th>Bankruptcy</th>
            <th>Going concern failure</th>
            <th>Customers &Delta;</th>
            <th>Robust utility</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((candidate) => {
            const selected = candidate.strategy === selectedStrategy;
            const expected = (candidate.expected_ending_cash / ceiling) * 100;
            const downside = (candidate.downside_ending_cash / ceiling) * 100;
            return (
              <tr key={candidate.strategy} className={selected ? "selected" : undefined}>
                <td>
                  {selected && <i>Selected</i>}
                  <strong>{titleCase(candidate.strategy)}</strong>
                </td>
                <td className="distribution-cell">
                  <svg viewBox="0 0 100 12" className="distribution" preserveAspectRatio="none" role="img"
                    aria-label={`Downside ${compactMoney(candidate.downside_ending_cash)} to expected ${compactMoney(candidate.expected_ending_cash)}`}>
                    <line className="dist-span" x1={downside} x2={expected} y1={6} y2={6} />
                    <rect className="dist-downside" x={downside - 0.5} y={1} width={1} height={10} />
                    <rect className="dist-expected" x={expected - 0.5} y={0} width={1} height={12} />
                  </svg>
                </td>
                <td>{compactMoney(candidate.expected_ending_cash)}</td>
                <td>{compactMoney(candidate.downside_ending_cash)}</td>
                <td className={candidate.bankruptcy_probability > 0.1 ? "cell-danger" : undefined}>
                  {percent(candidate.bankruptcy_probability, 1)}
                </td>
                <td className={(candidate.going_concern_failure_probability ?? 0) > 0.1 ? "cell-danger" : undefined}>
                  {percent(candidate.going_concern_failure_probability ?? 0, 1)}
                </td>
                <td>{signed(candidate.expected_customer_growth, (value) => Math.round(value).toLocaleString("en-US"))}</td>
                <td>
                  <span className={`robustness robustness-${candidate.robustness.toLowerCase()}`}>
                    {candidate.robust_utility.toFixed(2)} · {candidate.robustness}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
