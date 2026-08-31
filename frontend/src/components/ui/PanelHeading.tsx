export function PanelHeading({ index, title, aside }: { index: string; title: string; aside?: string }) {
  return (
    <header className="panel-heading">
      <div>
        <p className="eyebrow">{index}</p>
        <h2>{title}</h2>
      </div>
      {aside && <span>{aside}</span>}
    </header>
  );
}
