import { cleanup, screen, within } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import { fixtures, installApi, preparedId, runId, week1Id } from "../tests/fixtures";
import { renderAt } from "../tests/render";

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

test("audits one decision from compared candidates to the model version it used", async () => {
  installApi(fixtures(true));
  renderAt(`/runs/${runId}/decisions/${week1Id}`);

  expect(await screen.findByRole("heading", { name: "Balanced Growth" })).toBeInTheDocument();
  expect(
    screen.getByText(
      "Balanced growth keeps the downside above the runway floor in most plausible worlds.",
    ),
  ).toBeInTheDocument();
  expect(screen.getByText("ROBUST_DOWNSIDE_PROTECTION")).toBeInTheDocument();

  // All three simulated candidates stay visible, including the higher expected-value one that lost.
  const candidates = screen.getByRole("table", { name: "Simulated candidate strategies" });
  expect(within(candidates).getAllByRole("row")).toHaveLength(4);
  expect(within(candidates).getByText("Aggressive Growth")).toBeInTheDocument();
  expect(within(candidates).getByText("Selected")).toBeInTheDocument();

  const forecasts = screen.getByRole("region", { name: "Prediction versus actual" });
  expect(within(forecasts).getAllByText(/D\+/)).toHaveLength(4);
  expect(forecasts).toHaveTextContent("Miss");
  expect(forecasts).toHaveTextContent("Pending");

  const receipts = screen.getByRole("region", { name: "Action receipts" });
  expect(receipts).toHaveTextContent("2 / 2 receipts");
  expect(receipts).toHaveTextContent("set_daily_spend");
  expect(within(receipts).getAllByText("Executed")).toHaveLength(2);

  const model = screen.getByRole("region", { name: "Model version used" });
  expect(model).toHaveTextContent("Version 2");
  expect(model).toHaveTextContent("0.72 → 0.64");
  expect(model).toHaveTextContent("Rebuild recommended");
});

test("explains that a decision still being prepared has nothing to audit yet", async () => {
  installApi(fixtures());
  renderAt(`/runs/${runId}/decisions/${preparedId}`);

  expect(await screen.findByText(/artifacts pending/)).toBeInTheDocument();
  expect(screen.getByText(/publishes the model version, forecasts and receipts/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "← Back to cockpit" })).toHaveAttribute(
    "href",
    `/runs/${runId}`,
  );
});
