import { expect, test } from "vitest";

import type { DecisionExplanation } from "../api/client";
import { fixtures, week0Id, week1Id } from "../tests/fixtures";
import {
  buildReplayData,
  commitDay,
  companyStateAtDay,
  decisionsUpToDay,
  eventsUpToDay,
  healthAtDay,
  modelVersionAtDay,
  predictionMarkersAtDay,
  type ReplayData,
} from "./replay";

function replay(modelError = false, withExplanations = true): ReplayData {
  const fixture = fixtures(modelError);
  const explanations = new Map<string, DecisionExplanation>();
  if (withExplanations) {
    for (const [id, explanation] of Object.entries(fixture.explanations)) {
      explanations.set(id, explanation as DecisionExplanation);
    }
  }
  return buildReplayData({
    run: fixture.run as never,
    events: fixture.events as never,
    decisions: fixture.decisions as never,
    predictions: fixture.predictions as never,
    latestModel: fixture.latestModel as never,
    report: fixture.report as never,
    explanations,
  });
}

test("annotates every event with the day of the week it belongs to", () => {
  const data = replay();
  const byType = (type: string) => data.events.filter((event) => event.type === type);

  expect(byType("run.created")[0].effectiveDay).toBe(0);
  // Week 0 bootstrap work is dated by the day that week committed.
  expect(byType("world_model.created")[0].effectiveDay).toBe(7);
  expect(byType("decision.committed")[0].effectiveDay).toBe(7);
  expect(byType("decision.committed")[1].effectiveDay).toBe(14);
  expect(byType("action.executed").at(-1)?.effectiveDay).toBe(14);
});

test("truncates the log and the decision list when the clock rewinds", () => {
  const data = replay();

  expect(eventsUpToDay(data, 14)).toHaveLength(data.events.length);
  expect(eventsUpToDay(data, 7).map((event) => event.type)).not.toContain("world_model.updated");
  expect(eventsUpToDay(data, 0).map((event) => event.type)).toEqual(["run.created"]);

  expect(decisionsUpToDay(data, 14).map((decision) => decision.week)).toEqual([0, 1]);
  expect(decisionsUpToDay(data, 7).map((decision) => decision.week)).toEqual([0]);
  expect(decisionsUpToDay(data, 6)).toHaveLength(0);
});

test("dates the still-prepared decision from its observation instead of a missing receipt", () => {
  const data = replay();
  const prepared = data.decisions.find((decision) => decision.status === "prepared");

  expect(commitDay(data, prepared!)).toBe(21);
});

test("reads company state as of the day, with growth from the previous observation", () => {
  const data = replay();

  expect(companyStateAtDay(data, 0)?.cash).toBe(1_000_000);
  expect(companyStateAtDay(data, 13)?.cash).toBe(1_010_000);

  const day14 = companyStateAtDay(data, 14);
  expect(day14?.customers).toBe(1_290);
  expect(day14?.growth).toBeCloseTo((1_290 - 1_240) / 1_240, 5);
  // Cash rose between day 7 and day 14, so no runway is implied.
  expect(day14?.runwayWeeks).toBeNull();
});

test("derives runway only while cash is falling", () => {
  const data = replay(true);
  const day14 = companyStateAtDay(data, 14);

  expect(day14?.cash).toBe(700_000);
  expect(day14?.runwayWeeks).toBeCloseTo(700_000 / 310_000, 5);
});

test("moves prediction markers from pending to matured as the clock passes the target day", () => {
  const data = replay();
  const week1Seven = (day: number) =>
    predictionMarkersAtDay(data, day).find(
      (marker) => marker.decisionWeek === 1 && marker.horizonDays === 7,
    );

  // The week 1 forecast is not issued until day 7 and matures on day 14.
  expect(week1Seven(6)).toBeUndefined();
  expect(week1Seven(7)?.state).toBe("pending");
  expect(week1Seven(13)?.state).toBe("pending");
  expect(week1Seven(14)?.state).toBe("hit");

  expect(predictionMarkersAtDay(data, 14)).toHaveLength(8);
  expect(predictionMarkersAtDay(data, 0)).toHaveLength(4);
});

test("marks a missed interval and surfaces the degraded signal only after it is evaluated", () => {
  const data = replay(true);

  expect(
    predictionMarkersAtDay(data, 14).find(
      (marker) => marker.decisionWeek === 1 && marker.horizonDays === 7,
    )?.state,
  ).toBe("miss");

  expect(healthAtDay(data, 7)?.status).toBe("healthy");
  expect(healthAtDay(data, 14)?.status).toBe("degraded");
  expect(healthAtDay(data, 14)?.rebuild_recommended).toBe(true);
  expect(healthAtDay(data, 6)).toBeNull();
});

test("rewinds the world model to the version each week actually used", () => {
  const data = replay(true);

  // Day 0 predates the first committed week, so no model may be shown at all.
  expect(modelVersionAtDay(data, 0)).toEqual({ model: null, exact: true });
  expect(modelVersionAtDay(data, 7)).toMatchObject({ exact: true });
  expect(modelVersionAtDay(data, 7).model?.version).toBe(1);
  expect(modelVersionAtDay(data, 14).model?.version).toBe(2);
  expect(modelVersionAtDay(data, 14).model?.changes[0]?.new_estimate).toBe(0.64);
});

test("falls back to the latest known model while explanations are still loading", () => {
  const data = replay(true, false);
  const shown = modelVersionAtDay(data, 7);

  expect(shown.exact).toBe(false);
  expect(shown.model?.version).toBe(2);
});
