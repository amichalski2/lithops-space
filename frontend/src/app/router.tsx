import { Navigate, type RouteObject } from "react-router-dom";

import { App } from "./App";
import { CockpitPage } from "../pages/CockpitPage";
import { DecisionPage } from "../pages/DecisionPage";
import { LandingPage } from "../pages/LandingPage";
import { RunLayout } from "../pages/RunLayout";
import { RunsPage } from "../pages/RunsPage";

export const routes: RouteObject[] = [
  // The landing owns the whole viewport, so it sits outside the App shell's fixed-width column.
  { path: "/", element: <LandingPage /> },
  {
    element: <App />,
    children: [
      { path: "launch", element: <RunsPage /> },
      {
        path: "runs/:runId",
        element: <RunLayout />,
        children: [
          { index: true, element: <CockpitPage /> },
          { path: "decisions/:decisionId", element: <DecisionPage /> },
        ],
      },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
];
