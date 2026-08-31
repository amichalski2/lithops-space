/**
 * The Lithops mark. The artwork is a fixed render, so the masthead, the stage and the favicon all
 * pull the same file; callers only choose its height. WebP keeps the alpha channel the dark
 * backgrounds need without shipping the half-megabyte PNG on every page load.
 */
export function LithopsMark({ className, label }: { className?: string; label?: string }) {
  return (
    <img
      className={className}
      src="/logo.webp"
      width={512}
      height={512}
      decoding="async"
      {...(label ? { alt: label } : { alt: "", "aria-hidden": true })}
    />
  );
}
