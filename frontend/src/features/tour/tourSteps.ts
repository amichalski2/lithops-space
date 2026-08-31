export type TourStep = {
  /** Matches a `data-tour` attribute on the panel the step is about. */
  target: string;
  eyebrow: string;
  title: string;
  body: string;
};

/**
 * The guided read of the cockpit, in the order the panels answer each other: what day it is,
 * what the company did, what the system believed, what it decided, and how it was graded. The
 * copy states what a panel *is evidence of* rather than naming its widgets — a visitor arriving
 * from the landing page has thirty seconds of patience and no idea what a world model is.
 */
export const TOUR_STEPS: TourStep[] = [
  {
    target: "stage",
    eyebrow: "The clock",
    title: "You are looking at one day of a finished run",
    body: "Nothing here is a projection. The whole run already happened against the benchmark, and the cockpit replays it. The ring tracks how far into the horizon this day sits.",
  },
  {
    target: "company",
    eyebrow: "Step 1 · Evidence",
    title: "What the company actually did",
    body: "Six vitals read straight from the week's observation — cash, revenue, customers, churn, runway and the customer trend. This is the ground truth every other panel is judged against.",
  },
  {
    target: "model",
    eyebrow: "Step 2 · Belief",
    title: "What the system believes about the business",
    body: "The world model is the company's theory of itself. Each parameter carries an estimate, a confidence and bounds. When reality contradicts it, the model is rewritten — and the panel says which parameter moved and by how much.",
  },
  {
    target: "decisions",
    eyebrow: "Step 3 · Commitment",
    title: "One decision per week, locked before it acts",
    body: "Every week commits an action plan and a forecast at the same time. The forecast cannot be edited afterwards, which is what makes the next step mean anything.",
  },
  {
    target: "decisions",
    eyebrow: "Step 4 · The grade",
    title: "HIT or MISS, decided by the calendar",
    body: "When a forecast's target day arrives it is scored against what really happened. HIT and MISS are the ledger — the system is never graded on how convincing its reasoning sounded.",
  },
  {
    target: "logs",
    eyebrow: "Step 5 · Receipts",
    title: "Every step the agents took, in order",
    body: "The trace is the audit trail: which agent acted, on which day, and what it produced. Scrub the timeline and the log takes itself back with you.",
  },
  {
    target: "transport",
    eyebrow: "Ready",
    title: "Now play it",
    body: "Run Simulation walks the clock forward day by day and the panels update underneath it. You can scrub the timeline at any point, or step one real week further into the future.",
  },
];
