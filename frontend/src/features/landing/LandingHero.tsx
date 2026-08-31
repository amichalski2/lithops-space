import { ErrorNotice } from "../../components/ui/ErrorNotice";
import { LithopsMark } from "../../components/ui/LithopsMark";

const STACK = [
  "Gemini 3.7",
  "Google ADK",
  "Cloud Run",
  "Cloud Logging",
  "Supabase",
  "BYOK",
];

/**
 * The one-pager that fronts the product: a single viewport, no scroll, no navigation. Its only
 * job is to get a visitor into a run — everything past this screen lives behind the primary CTA.
 */
export function LandingHero({
  hasDemoRun,
  busy,
  error,
  onRun,
  onCreate,
}: {
  hasDemoRun: boolean;
  busy: boolean;
  error: string | null;
  onRun: () => void;
  onCreate: () => void;
}) {
  return (
    <main className="landing">
      {/* The render is anchored to the backdrop, not to the copy grid, so it stays seated in the
          honeycomb valley instead of drifting with the text column. */}
      <div className="landing-backdrop" aria-hidden>
        <div className="landing-scene">
          <img className="landing-bg" src="/bg-hero.webp" alt="" decoding="async" />
          <img
            className="landing-render"
            src="/lithops-hero.webp"
            alt=""
            width={1400}
            height={1400}
            decoding="async"
            fetchPriority="high"
          />
        </div>
      </div>

      <header className="landing-brand">
        <LithopsMark className="landing-brand-mark" label="Lithops" />
        <span>Lithops</span>
      </header>

      <div className="landing-stage">
        <section className="landing-copy">
          <p className="landing-kicker">Autonomous company operating system</p>
          <h1>Writes the model that runs the company</h1>
          <p className="landing-lede">
            Coding agents write competing Python models of the business. A sandbox tests them on
            time-ordered evidence, the company runs on whichever predicts best, and every
            prediction is scored when reality arrives.
          </p>

          <div className="landing-actions">
            <button className="pill pill-solid" disabled={busy} onClick={onRun}>
              {busy && !hasDemoRun ? "Initializing" : "Run Simulation"}
              <i aria-hidden>↗</i>
            </button>
            {hasDemoRun && (
              <button className="pill pill-ghost" disabled={busy} onClick={onCreate}>
                {busy ? "Initializing" : "Create a Fresh Run"}
                <i aria-hidden>↗</i>
              </button>
            )}
          </div>

          {error && <ErrorNotice message={error} onRetry={onCreate} />}
        </section>
      </div>

      <footer className="landing-stack" role="group" aria-label="Hackathon project stack">
        <span>Hackathon project stack</span>
        <ul>
          {STACK.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </footer>
    </main>
  );
}
