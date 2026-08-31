import { useEffect, useRef } from "react";

import { eventActor, eventDetail, eventTitle } from "./eventDisplay";
import { useReplay } from "../cockpit/ReplayProvider";
import { eventsUpToDay } from "../../lib/replay";
import { padDay } from "../../lib/format";

export function LogsPanel() {
  const { data, clock } = useReplay();
  const visible = eventsUpToDay(data, clock.currentDay);
  const list = useRef<HTMLOListElement>(null);

  useEffect(() => {
    // Assigning scrollTop rather than calling scrollTo keeps this working wherever the
    // smooth-scroll API is missing; the list already animates via CSS scroll-behavior.
    const node = list.current;
    if (clock.playing && node) node.scrollTop = node.scrollHeight;
  }, [visible.length, clock.playing]);

  return (
    <section className="logs-panel" aria-label="Run event timeline" data-tour="logs">
      <header>
        <p className="eyebrow">Live trace</p>
        <span>
          {visible.length} / {data.events.length} events
        </span>
      </header>
      {visible.length === 0 ? (
        <p className="empty">Waiting for the first observation.</p>
      ) : (
        <ol className="log-list" ref={list}>
          {visible.map((event) => (
            <li key={event.id ?? `${event.sequence}-${event.type}`}>
              <time>D{padDay(event.effectiveDay)}</time>
              <span className={`event-node event-${event.type.replaceAll(".", "-")}`} />
              <small>{eventActor(event.type)}</small>
              <strong>{eventTitle(event.type)}</strong>
              <p>{eventDetail(event)}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
