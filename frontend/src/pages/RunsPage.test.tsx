import { cleanup, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { STORED_RUN_KEY } from "../features/cockpit/ReplayProvider";
import { clearGeminiApiKey, getGeminiApiKey } from "../features/runs/byok";
import { fixtures, installApi, runId } from "../tests/fixtures";
import { renderAt } from "../tests/render";

beforeEach(() => vi.stubEnv("VITE_DEMO_RUN_ID", ""));

afterEach(() => {
  cleanup();
  clearGeminiApiKey();
  localStorage.clear();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

test("requires a participant key before creating a fresh cloud simulation", async () => {
  installApi(fixtures());
  const user = userEvent.setup();
  renderAt("/launch");

  const input = screen.getByLabelText("Gemini API key");
  const submit = screen.getByRole("button", { name: "Start fresh run" });
  expect(input).toBeEnabled();
  expect(submit).toBeDisabled();

  const key = "participant-gemini-key-1234567890";
  await user.type(input, key);
  await user.click(submit);

  expect(getGeminiApiKey()).toBe(key);
  expect(await screen.findByRole("region", { name: "Company state" })).toBeInTheDocument();
});

test("surfaces API failure without persisting the participant key", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response("CEO-Bench unavailable", { status: 503 }),
  );
  const user = userEvent.setup();
  renderAt("/launch");

  await user.type(screen.getByLabelText("Gemini API key"), "participant-key-123456789012345");
  await user.click(screen.getByRole("button", { name: "Start fresh run" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("CEO-Bench unavailable");
  expect(localStorage.getItem("gemini-api-key")).toBeNull();
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
