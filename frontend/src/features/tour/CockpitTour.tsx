import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { TOUR_STEPS } from "./tourSteps";

const CARD_WIDTH = 340;
const GAP = 16;
const PAD = 8;
/** Used to place the card before the first measurement lands; replaced on the same frame. */
const ASSUMED_CARD_HEIGHT = 210;

type Rect = { top: number; left: number; width: number; height: number };

function readTarget(name: string): Rect | null {
  const node = document.querySelector(`[data-tour="${name}"]`);
  if (!node) return null;
  const rect = node.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) return null;
  return { top: rect.top, left: rect.left, width: rect.width, height: rect.height };
}

/**
 * Places the card beside the spotlight, preferring the side with room. The cockpit does not
 * scroll, so a step never has to bring its target into view — only to avoid covering it.
 */
function placeCard(rect: Rect, cardHeight: number) {
  const { innerWidth, innerHeight } = window;
  const roomRight = innerWidth - (rect.left + rect.width);
  const roomLeft = rect.left;

  let left: number;
  if (roomRight >= CARD_WIDTH + GAP * 2) left = rect.left + rect.width + GAP;
  else if (roomLeft >= CARD_WIDTH + GAP * 2) left = rect.left - CARD_WIDTH - GAP;
  else left = rect.left + rect.width / 2 - CARD_WIDTH / 2;

  // Centre on the target vertically, then keep the whole card on screen.
  let top = rect.top + rect.height / 2 - cardHeight / 2;
  top = Math.min(innerHeight - cardHeight - GAP, Math.max(GAP, top));
  left = Math.min(innerWidth - CARD_WIDTH - GAP, Math.max(GAP, left));
  return { top, left };
}

export function CockpitTour({ onClose }: { onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);
  const [cardHeight, setCardHeight] = useState(ASSUMED_CARD_HEIGHT);
  const card = useRef<HTMLDivElement>(null);

  const step = TOUR_STEPS[index];
  const last = index === TOUR_STEPS.length - 1;

  const next = useCallback(() => {
    setIndex((current) => {
      if (current < TOUR_STEPS.length - 1) return current + 1;
      onClose();
      return current;
    });
  }, [onClose]);
  const back = useCallback(() => setIndex((current) => Math.max(0, current - 1)), []);

  // The panels are laid out by the grid, so the rect is only correct after layout; re-measure on
  // resize because the cockpit reflows into two and one column at its breakpoints.
  useLayoutEffect(() => {
    const measure = () => setRect(readTarget(step.target));
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [step.target]);

  useLayoutEffect(() => {
    if (card.current) setCardHeight(card.current.offsetHeight);
  }, [index]);

  useEffect(() => {
    card.current?.focus();
  }, [index]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      else if (event.key === "ArrowRight") next();
      else if (event.key === "ArrowLeft") back();
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, back, onClose]);

  const position = rect ? placeCard(rect, cardHeight) : null;

  return (
    <div className="tour" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      {/* Catches the clicks the spotlight's box-shadow cannot: a shadow is painted, not hit-tested. */}
      <div className="tour-block" />
      {rect && (
        <div
          className="tour-spotlight"
          style={{
            top: rect.top - PAD,
            left: rect.left - PAD,
            width: rect.width + PAD * 2,
            height: rect.height + PAD * 2,
          }}
        />
      )}

      <div
        ref={card}
        className="tour-card"
        tabIndex={-1}
        style={position ? { top: position.top, left: position.left } : { top: "50%", left: "50%" }}
      >
        <p className="tour-eyebrow">
          {step.eyebrow}
          <span>
            {index + 1} / {TOUR_STEPS.length}
          </span>
        </p>
        <h2 id="tour-title">{step.title}</h2>
        <p className="tour-body">{step.body}</p>

        <div className="tour-actions">
          <button type="button" className="tour-skip" onClick={onClose}>
            Skip
          </button>
          <button type="button" className="ghost" onClick={back} disabled={index === 0}>
            Back
          </button>
          <button type="button" className="tour-next" onClick={next}>
            {last ? "Got it" : "Next"}
          </button>
        </div>

        <ol className="tour-dots" aria-hidden>
          {TOUR_STEPS.map((item, dot) => (
            <li key={`${item.target}-${dot}`} className={dot === index ? "is-current" : undefined} />
          ))}
        </ol>
      </div>
    </div>
  );
}
