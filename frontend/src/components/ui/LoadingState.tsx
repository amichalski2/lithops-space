export function LoadingState({ label = "Loading cockpit", message = "Reading run ledger" }) {
  return (
    <section className="loading-state" aria-label={label}>
      <span className="spinner" />
      <p>{message}</p>
      <div className="loading-track">
        <i />
      </div>
    </section>
  );
}
