import { type FormEvent, useState } from "react";

export function ByokPlaceholder({
  busy,
  onCreate,
}: {
  busy: boolean;
  onCreate: (geminiApiKey: string) => void;
}) {
  const [key, setKey] = useState("");
  const valid = key.trim().length >= 20;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (valid && !busy) onCreate(key.trim());
  }

  return (
    <section className="byok" aria-label="Bring your own key">
      <p className="eyebrow">Bring your own key</p>
      <form className="byok-row" onSubmit={submit}>
        <input
          type="password"
          value={key}
          onChange={(event) => setKey(event.target.value)}
          placeholder="Gemini API key"
          aria-label="Gemini API key"
          autoComplete="off"
          spellCheck={false}
        />
        <button className="ghost" type="submit" disabled={!valid || busy}>
          {busy ? "Initializing…" : "Start fresh run"}
        </button>
      </form>
      <small>
        Used only for Gemini calls in this tab. The key is never stored in the browser, run ledger,
        receipts, or reports. Refreshing the page requires re-entry.
      </small>
    </section>
  );
}
