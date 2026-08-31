import { cleanup, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";

import { fixtures, installApi, runId } from "../tests/fixtures";
import { renderAt } from "../tests/render";

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

const cockpit = () => renderAt(`/runs/${runId}`);

/** The clock renders "Day" and its number as separate nodes so the number can carry the accent. */
const expectDay = (day: string) =>
  waitFor(() => expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(`Day ${day}`));

test("opens on the newest committed day with company state, model and ledger in view", async () => {
  installApi(fixtures());
  cockpit();

  await expectDay("014");

  const company = screen.getByRole("region", { name: "Company state" });
  expect(within(company).getByText("$1.0M")).toBeInTheDocument();
  expect(within(company).getByText("1,290")).toBeInTheDocument();

  const model = await screen.findByRole("region", { name: "World model parameters" });
  await waitFor(() => expect(model).toHaveTextContent("Price Elasticity"));
  expect(model).toHaveTextContent("v2");

  const decisions = screen.getByRole("region", { name: "Decision list" });
  expect(within(decisions).getByText("2 committed")).toBeInTheDocument();
  expect(within(decisions).getAllByText("Balanced Growth")).toHaveLength(2);
  expect(within(decisions).getAllByText("Hit").length).toBeGreaterThan(0);

  expect(screen.getByRole("region", { name: "Run event timeline" })).toHaveTextContent(
    "Model Health Evaluated",
  );
});

test("places every committed forecast on the day axis and matures it on its target day", async () => {
  installApi(fixtures());
  cockpit();

  await expectDay("014");
  const slider = screen.getByRole("slider", { name: "Day scrubber" });
  expect(slider).toHaveAttribute("aria-valuenow", "14");
  expect(slider).toHaveAttribute("aria-valuemax", "14");

  // One lane per required horizon; two weeks of four forecasts sit on them.
  expect(slider.querySelectorAll(".lane-label")).toHaveLength(4);
  expect(slider.querySelectorAll(".marker")).toHaveLength(8);
  expect(slider.querySelectorAll(".marker-hit")).toHaveLength(2);
  expect(slider.querySelectorAll(".marker-miss")).toHaveLength(0);
});

test("rewinding the clock takes back the log, the decisions and the model version", async () => {
  installApi(fixtures());
  const user = userEvent.setup();
  cockpit();

  const model = await screen.findByRole("region", { name: "World model parameters" });
  await waitFor(() => expect(model).toHaveTextContent("v2"));

  const slider = screen.getByRole("slider", { name: "Day scrubber" });
  slider.focus();
  await user.keyboard("{Home}");

  expect(slider).toHaveAttribute("aria-valuenow", "0");
  await expectDay("000");

  // Day 0 knows only that the run was created — no week has committed yet.
  const logs = screen.getByRole("region", { name: "Run event timeline" });
  expect(logs).toHaveTextContent("Run Created");
  expect(logs).not.toHaveTextContent("Decision Committed");
  expect(screen.getByRole("region", { name: "Decision list" })).toHaveTextContent(
    "No decision has been committed yet.",
  );
  expect(model).toHaveTextContent("Bootstrapping");

  await user.keyboard("{End}");
  expect(slider).toHaveAttribute("aria-valuenow", "14");
  await waitFor(() => expect(model).toHaveTextContent("v2"));
});

test("shows the model-error path: missed interval, challenge state and recalibration", async () => {
  installApi(fixtures(true));
  cockpit();

  const model = await screen.findByRole("region", { name: "World model parameters" });
  await waitFor(() => expect(model).toHaveTextContent("Model error high"));

  const challenge = within(model).getByLabelText("Model challenge");
  expect(challenge).toHaveTextContent("Rebuild recommended");
  expect(challenge).toHaveTextContent("Persistent Interval Miss");
  expect(challenge).toHaveTextContent("31.0%");

  // The recalibration that the miss caused is stated as a before/after on the parameter.
  expect(model).toHaveTextContent("0.72 → 0.64");
  expect(model).toHaveTextContent("61% → 38% confidence");

  const slider = screen.getByRole("slider", { name: "Day scrubber" });
  expect(slider.querySelectorAll(".marker-miss")).toHaveLength(1);
  expect(within(screen.getByRole("region", { name: "Decision list" })).getByText("Miss")).toBeInTheDocument();
});

test("keeps the replay honest with a live badge and a link into each decision", async () => {
  installApi(fixtures());
  cockpit();

  expect(await screen.findByText("Live")).toBeInTheDocument();

  const decisions = screen.getByRole("region", { name: "Decision list" });
  expect(within(decisions).getAllByRole("link")[0]).toHaveAttribute(
    "href",
    expect.stringContaining(`/runs/${runId}/decisions/`),
  );
});
