import { Link, Outlet } from "react-router-dom";

import { LithopsMark } from "../components/ui/LithopsMark";
import { StatusIcon } from "../components/ui/StatusIcon";

export function App() {
  return (
    <main className="shell">
      <header className="masthead">
        <Link className="wordmark" to="/" aria-label="Lithops home">
          <LithopsMark className="mark" />
          <span>
            Lithops<small>World model OS</small>
          </span>
        </Link>
        <div className="environment">
          <StatusIcon /> CEO-Bench / Simulation cockpit
        </div>
      </header>
      <Outlet />
    </main>
  );
}
