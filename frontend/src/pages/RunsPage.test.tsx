import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { STORED_RUN_KEY } from "../features/cockpit/ReplayProvider";
import { fixtures, installApi, runId } from "../tests/fixtures";
import { renderAt } from "../tests/render";

beforeEach(() => vi.stubEnv("VITE_DEMO_RUN_ID", ""));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

test("offers no launch action when there is no finished run to replay", () => {
  renderAt("/launch");

  expect(screen.getByText("No finished run is configured for replay yet.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Replay best CEO-Bench run/ })).not.toBeInTheDocument();
});

test("replays the configured evidence run from day zero", async () => {
  localStorage.setItem(STORED_RUN_KEY, runId);
  installApi(fixtures());
  const user = userEvent.setup();
  renderAt("/launch");

  await user.click(screen.getByRole("button", { name: /Replay best CEO-Bench run/ }));

  expect(await screen.findByRole("button", { name: /Pause/ })).toBeInTheDocument();
  expect(Number(screen.getByRole("slider", { name: "Day scrubber" }).getAttribute("aria-valuenow")))
    .toBeLessThan(14);
});
