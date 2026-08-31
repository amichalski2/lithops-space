export function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-notice" role="alert">
      <div>
        <strong>Cockpit link interrupted</strong>
        <p>{message}</p>
      </div>
      {onRetry && <button onClick={onRetry}>Retry</button>}
    </div>
  );
}
