import { cleanup, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { fixtures, installApi, runId } from "../tests/fixtures";
import { renderAt } from "../tests/render";

beforeEach(() => vi.stubEnv("VITE_DEMO_RUN_ID", ""));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

test("shows only services that exist in the submitted architecture", () => {
  renderAt("/");

  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
    "Writes the model that runs the company",
  );
  const stack = screen.getByRole("group", { name: "Hackathon project stack" });
  expect(stack).toHaveTextContent("Gemini 3.7");
  expect(stack).toHaveTextContent("Google ADK");
  expect(stack).toHaveTextContent("Cloud Run");
  expect(stack).toHaveTextContent("Cloud Logging");
  expect(stack).toHaveTextContent("Supabase");
  expect(stack).toHaveTextContent("BYOK");
  expect(stack).not.toHaveTextContent("Firestore");
});

test("routes a fresh participant to the BYOK launch instead of minting an inert run", async () => {
  const user = userEvent.setup();
  renderAt("/");

  await user.click(screen.getByRole("button", { name: /Run Simulation/ }));

  expect(await screen.findByRole("region", { name: "Bring your own key" })).toBeInTheDocument();
});

test("opens the configured demo run paused, behind a guided tour of the panels", async () => {
  vi.stubEnv("VITE_DEMO_RUN_ID", runId);
  installApi(fixtures());
  const user = userEvent.setup();
  renderAt("/");

  await user.click(screen.getByRole("button", { name: /Run Simulation/ }));

  const tour = await screen.findByRole("dialog");
  expect(tour).toHaveTextContent("You are looking at one day of a finished run");
  // Nothing may be moving while the visitor is still being told what they are looking at:
  // a running clock is the only thing that puts Pause on the transport.
  expect(screen.queryByRole("button", { name: /Pause/ })).not.toBeInTheDocument();

  await user.click(within(tour).getByRole("button", { name: "Next" }));
  expect(tour).toHaveTextContent("What the company actually did");

  await user.click(within(tour).getByRole("button", { name: "Back" }));
  expect(tour).toHaveTextContent("You are looking at one day of a finished run");

  await user.click(within(tour).getByRole("button", { name: "Skip" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

test("keeps the tour reachable after it is dismissed", async () => {
  installApi(fixtures());
  const user = userEvent.setup();
  renderAt(`/runs/${runId}`);

  // A run opened directly is not a first visit, so the tour stays out of the way until asked for.
  await screen.findByRole("heading", { level: 1 });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Replay the guided tour" }));
  expect(await screen.findByRole("dialog")).toHaveTextContent("finished run");
});
