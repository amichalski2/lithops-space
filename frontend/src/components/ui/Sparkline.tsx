const WIDTH = 96;
const HEIGHT = 30;
const PAD = 2;

/**
 * The trend shape behind a metric's number. It carries no axis on purpose: the tile already states
 * the current value, so the line only has to answer "which way, and how steadily".
 */
export function Sparkline({ values, label }: { values: number[]; label?: string }) {
  if (values.length < 2) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  // A flat series would divide by zero; drawing it down the middle is the honest rendering.
  const span = max - min || 1;
  const step = (WIDTH - PAD * 2) / (values.length - 1);
  const points = values.map((value, index) => {
    const x = PAD + index * step;
    const y = HEIGHT - PAD - ((value - min) / span) * (HEIGHT - PAD * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = points.at(-1)!.split(",");
  const rising = values.at(-1)! >= values[0];

  return (
    <svg
      className={rising ? "sparkline" : "sparkline is-falling"}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      {...(label ? { role: "img", "aria-label": label } : { "aria-hidden": true })}
    >
      <polygon className="sparkline-area" points={`${PAD},${HEIGHT} ${points.join(" ")} ${WIDTH - PAD},${HEIGHT}`} />
      <polyline className="sparkline-line" points={points.join(" ")} />
      <circle className="sparkline-head" cx={last[0]} cy={last[1]} r="1.8" />
    </svg>
  );
}
