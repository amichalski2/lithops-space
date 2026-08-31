import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider, createBrowserRouter } from "react-router-dom";

import { routes } from "./app/router";
import "./styles/global.css";
import "./styles/hud.css";
import "./styles/cockpit.css";
import "./styles/decision.css";
import "./styles/landing.css";
import "./styles/tour.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={createBrowserRouter(routes)} />
  </StrictMode>,
);
