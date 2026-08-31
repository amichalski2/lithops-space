import { vi } from "vitest";

export const runId = "11111111-1111-4111-8111-111111111111";
export const week0Id = "22222222-2222-4222-8222-222222222220";
export const week1Id = "22222222-2222-4222-8222-222222222221";
export const preparedId = "22222222-2222-4222-8222-222222222222";
export const model1Id = "33333333-3333-4333-8333-333333333331";
export const model2Id = "33333333-3333-4333-8333-333333333332";

const iso = (minute: number) => `2026-08-24T10:${String(minute).padStart(2, "0")}:00Z`;

function observation(day: number, cash: number, customers: number, revenue: number, churn: number) {
  return {
    day,
    cash,
    metrics: { weekly_revenue: revenue, active_customers: customers, churn },
    observed_at: iso(day),
  };
}

function target(id: string, horizon: number, issued: number, point: number, spread: number) {
  return {
    id,
    horizon_days: horizon,
    target_day: issued + horizon,
    point,
    lower: point - spread,
    upper: point + spread,
    interval_probability: 0.95,
  };
}

function score(targetId: string, point: number, actual: number, hit: boolean, normalized: number) {
  return {
    target_id: targetId,
    signed_error: actual - point,
    absolute_error: Math.abs(actual - point),
    absolute_percentage_error: Math.abs(actual - point) / point,
    normalized_absolute_error: normalized,
    interval_hit: hit,
    interval_width: 80_000,
    interval_score: hit ? 0.1 : 6.2,
    weighted_interval_score: hit ? 0.1 : 6.2,
    scored_at: iso(5),
  };
}

function parameters(confidence: number, saturation: number) {
  return [
    {
      name: "price_elasticity",
      estimate: -0.72,
      lower_bound: -1.1,
      upper_bound: -0.3,
      confidence,
      unit: "elasticity",
      lag_weeks: 0,
      evidence: [{ kind: "observation", reference: "observation:day-7", observed_day: 7, note: null }],
    },
    {
      name: "marketing_saturation",
      estimate: saturation,
      lower_bound: 0.3,
      upper_bound: 0.9,
      confidence,
      unit: "ratio",
      lag_weeks: 0,
      evidence: [{ kind: "generic_prior", reference: "bootstrap-prior", observed_day: null, note: null }],
    },
  ];
}

const relationships = [
  {
    key: "pricing_conversion",
    cause: "pricing",
    effect: "conversion",
    shape: "linear",
    parameter_names: ["price_elasticity"],
    lag_weeks: 0,
    confidence: 0.7,
    evidence: [],
  },
  {
    key: "marketing_acquisition",
    cause: "marketing_spend",
    effect: "acquisition",
    shape: "saturating",
    parameter_names: ["marketing_saturation"],
    lag_weeks: 0,
    confidence: 0.65,
    evidence: [],
  },
];

const candidates = [
  {
    strategy: "aggressive_growth",
    expected_ending_cash: 1_500_000,
    downside_ending_cash: 180_000,
    bankruptcy_probability: 0.22,
    expected_customer_growth: 0.55,
    robustness: "low",
    robust_utility: 0.41,
    rollout_count: 1000,
  },
  {
    strategy: "balanced_growth",
    expected_ending_cash: 1_350_000,
    downside_ending_cash: 760_000,
    bankruptcy_probability: 0.03,
    expected_customer_growth: 0.31,
    robustness: "high",
    robust_utility: 0.82,
    rollout_count: 1000,
  },
  {
    strategy: "cash_preservation",
    expected_ending_cash: 1_180_000,
    downside_ending_cash: 920_000,
    bankruptcy_probability: 0.01,
    expected_customer_growth: 0.08,
    robustness: "high",
    robust_utility: 0.68,
    rollout_count: 1000,
  },
];

/**
 * A two-week run at day 14 with a third decision still prepared.
 * `modelError` flips week 1 into an interval miss that recalibrates the model.
 */
export function fixtures(modelError = false) {
  const week1Cash = modelError ? 700_000 : 1_018_000;

  const run = {
    id: runId,
    status: "running",
    workflow_step: "checkpoint",
    benchmark_session_id: "fake-session",
    current_day: 14,
    horizon_days: 500,
    version: 5,
    last_decision_id: week1Id,
    failure_reason: null,
    created_at: iso(0),
    updated_at: iso(5),
  };

  const day0 = observation(0, 1_000_000, 1_200, 80_000, 0.033);
  const day7 = observation(7, 1_010_000, 1_240, 84_000, 0.031);
  const day14 = observation(14, week1Cash, 1_290, 88_000, modelError ? 0.052 : 0.029);

  const model1 = {
    id: model1Id,
    run_id: runId,
    version: 1,
    source_observation_day: 0,
    based_on_version_id: null,
    update_method: "bootstrap_v1",
    schema_version: "1.0",
    parameters: parameters(0.5, 0.72),
    relationships,
    changes: [],
    created_at: iso(0),
  };

  const model2 = {
    id: model2Id,
    run_id: runId,
    version: 2,
    source_observation_day: 7,
    based_on_version_id: model1Id,
    update_method: modelError ? "residual_recalibration_v1" : "bootstrap_v1",
    schema_version: "1.0",
    parameters: parameters(modelError ? 0.38 : 0.76, modelError ? 0.64 : 0.72),
    relationships,
    changes: modelError
      ? [
          {
            parameter_name: "marketing_saturation",
            previous_estimate: 0.72,
            new_estimate: 0.64,
            previous_confidence: 0.61,
            new_confidence: 0.38,
            update_method: "prediction_residual",
            evidence: [
              { kind: "prediction_residual", reference: "outcome:miss-1", observed_day: 14, note: null },
            ],
          },
        ]
      : [],
    created_at: iso(3),
  };

  function decision(
    id: string,
    week: number,
    status: string,
    obs: ReturnType<typeof observation>,
    actual: ReturnType<typeof observation> | null,
    modelVersionId: string | null,
    predictionId: string | null,
  ) {
    return {
      id,
      run_id: runId,
      week,
      status,
      observation: obs,
      actual_outcome: actual,
      action_plan: {
        name: "balanced_growth",
        strategy_family: "balanced_growth",
        rationale: "Protects runway across plausible demand models.",
        commands: [
          { tool: "set_prices", arguments: { A: 45 }, idempotency_key: `price-${week}` },
          { tool: "set_daily_spend", arguments: { marketing: 9000 }, idempotency_key: `spend-${week}` },
        ],
      },
      forecasts: {
        items: [
          { horizon_days: 7, point: 1_020_000, lower: 980_000, upper: 1_060_000 },
          { horizon_days: 28, point: 1_160_000, lower: 1_040_000, upper: 1_280_000 },
          { horizon_days: 84, point: 1_400_000, lower: 1_100_000, upper: 1_700_000 },
          { horizon_days: 182, point: 1_900_000, lower: 1_300_000, upper: 2_500_000 },
        ],
      },
      world_model_version_id: modelVersionId,
      prediction_id: predictionId,
      prompt_version: "gemini-executive-v1",
      assumptions: ["Demand remains inside the learned elasticity interval."],
      evidence_references: [`observation:day-${obs.day}`],
      candidate_evaluations: candidates,
      selection_reason_code: "ROBUST_DOWNSIDE_PROTECTION",
      selection_reason:
        "Balanced growth keeps the downside above the runway floor in most plausible worlds.",
      created_at: iso(obs.day),
      committed_at: actual ? iso(actual.day) : null,
    };
  }

  const decisionWeek0 = decision(week0Id, 0, "committed", day0, day7, model1Id, "pred-0");
  const decisionWeek1 = decision(week1Id, 1, "committed", day7, day14, model2Id, "pred-1");
  const decisionPrepared = decision(preparedId, 2, "prepared", day14, null, null, null);

  function prediction(
    id: string,
    decisionId: string,
    week: number,
    issued: number,
    modelVersionId: string,
    outcomes: unknown[],
  ) {
    return {
      prediction: {
        id,
        run_id: runId,
        decision_id: decisionId,
        decision_week: week,
        issued_day: issued,
        model_version_id: modelVersionId,
        prompt_version: "gemini-executive-v1",
        observation_reference: `observation:day-${issued}`,
        assumptions: ["Demand remains inside the learned elasticity interval."],
        evidence_references: [`observation:day-${issued}`],
        uncertainty_source: "world-model sampling",
        confidence: 0.73,
        cash_sensitivities: [],
        targets: [
          target(`t-${week}-7`, 7, issued, 1_020_000, 40_000),
          target(`t-${week}-28`, 28, issued, 1_160_000, 120_000),
          target(`t-${week}-84`, 84, issued, 1_400_000, 300_000),
          target(`t-${week}-182`, 182, issued, 1_900_000, 600_000),
        ],
        committed_at: iso(issued),
      },
      outcomes,
    };
  }

  const outcome0 = {
    id: "66666666-6666-4666-8666-666666666660",
    run_id: runId,
    ledger_entry_id: "pred-0",
    target_id: "t-0-7",
    actual: {
      target_id: "t-0-7",
      observed_day: 7,
      cash: 1_010_000,
      observation_reference: "observation:day-7",
      observed_at: iso(2),
    },
    score: score("t-0-7", 1_020_000, 1_010_000, true, 0.01),
    recorded_at: iso(2),
  };

  const outcome1 = {
    id: "66666666-6666-4666-8666-666666666661",
    run_id: runId,
    ledger_entry_id: "pred-1",
    target_id: "t-1-7",
    actual: {
      target_id: "t-1-7",
      observed_day: 14,
      cash: week1Cash,
      observation_reference: "observation:day-14",
      observed_at: iso(5),
    },
    score: score("t-1-7", 1_020_000, week1Cash, !modelError, modelError ? 0.31 : 0.002),
    recorded_at: iso(5),
  };

  const prediction0 = prediction("pred-0", week0Id, 0, 0, model1Id, [outcome0]);
  const prediction1 = prediction("pred-1", week1Id, 1, 7, model2Id, [outcome1]);

  function health(id: string, day: number, degraded: boolean, modelVersionId: string) {
    return {
      id,
      run_id: runId,
      model_version_id: modelVersionId,
      evaluated_day: day,
      status: degraded ? "degraded" : "healthy",
      outcome_ids: [degraded ? outcome1.id : outcome0.id],
      horizon_performance: [
        {
          horizon_days: 7,
          outcome_count: degraded ? 3 : 1,
          mean_normalized_absolute_error: degraded ? 0.31 : 0.002,
          interval_coverage: degraded ? 0 : 1,
          mean_weighted_interval_score: degraded ? 6.2 : 0.1,
          signed_bias: degraded ? -320_000 : -10_000,
        },
      ],
      interval_miss_count: degraded ? 3 : 0,
      directional_bias: degraded ? -320_000 : -10_000,
      rebuild_recommended: degraded,
      trigger_codes: degraded ? ["persistent_interval_miss", "high_normalized_error"] : [],
      evaluated_at: iso(day),
    };
  }

  const health0 = health("88888888-8888-4888-8888-888888888880", 7, false, model1Id);
  const health1 = health("88888888-8888-4888-8888-888888888881", 14, modelError, model2Id);

  let sequence = 0;
  const event = (type: string, payload: Record<string, unknown>) => ({
    id: `event-${(sequence += 1)}`,
    run_id: runId,
    type,
    payload,
    sequence,
    created_at: iso(sequence),
  });

  const events = [
    event("run.created", { horizon_days: 500 }),
    event("run.started", { status: "running" }),
    event("world_model.created", { model_version_id: model1Id, version: 1 }),
    event("decision.prepared", {
      decision_id: week0Id,
      week: 0,
      model_version_id: model1Id,
      candidate_count: 3,
      selected_strategy: "balanced_growth",
      selection_reason_code: "ROBUST_DOWNSIDE_PROTECTION",
    }),
    event("prediction.committed", {
      prediction_id: "pred-0",
      decision_id: week0Id,
      model_version_id: model1Id,
      target_days: [7, 28, 84, 182],
      recovered: false,
    }),
    event("action.executed", {
      decision_id: week0Id,
      receipt_id: "receipt-0a",
      tool: "set_prices",
      idempotency_key: "price-0",
    }),
    event("decision.committed", {
      decision_id: week0Id,
      week: 0,
      day: 7,
      cash: 1_010_000,
      model_version_id: model1Id,
      prediction_id: "pred-0",
    }),
    event("prediction.matured", {
      prediction_outcome_id: outcome0.id,
      target_id: "t-0-7",
      observed_day: 7,
      normalized_absolute_error: 0.01,
      interval_hit: true,
    }),
    event("model_health.evaluated", {
      model_health_signal_id: health0.id,
      model_version_id: model1Id,
      status: "healthy",
      rebuild_recommended: false,
      trigger_codes: [],
    }),
    event("world_model.updated", {
      model_version_id: model2Id,
      version: 2,
      parent_model_version_id: model1Id,
      changed_parameters: modelError ? ["marketing_saturation"] : [],
      rebuild_recommended: false,
    }),
    event("decision.prepared", {
      decision_id: week1Id,
      week: 1,
      model_version_id: model2Id,
      candidate_count: 3,
      selected_strategy: "balanced_growth",
      selection_reason_code: "ROBUST_DOWNSIDE_PROTECTION",
    }),
    event("prediction.committed", {
      prediction_id: "pred-1",
      decision_id: week1Id,
      model_version_id: model2Id,
      target_days: [14, 35, 91, 189],
      recovered: false,
    }),
    event("action.executed", {
      decision_id: week1Id,
      receipt_id: "receipt-1a",
      tool: "set_prices",
      idempotency_key: "price-1",
    }),
    event("action.executed", {
      decision_id: week1Id,
      receipt_id: "receipt-1b",
      tool: "set_daily_spend",
      idempotency_key: "spend-1",
    }),
    event("decision.committed", {
      decision_id: week1Id,
      week: 1,
      day: 14,
      cash: week1Cash,
      model_version_id: model2Id,
      prediction_id: "pred-1",
    }),
    event("prediction.matured", {
      prediction_outcome_id: outcome1.id,
      target_id: "t-1-7",
      observed_day: 14,
      normalized_absolute_error: modelError ? 0.31 : 0.002,
      interval_hit: !modelError,
    }),
    event("model_health.evaluated", {
      model_health_signal_id: health1.id,
      model_version_id: model2Id,
      status: health1.status,
      rebuild_recommended: health1.rebuild_recommended,
      trigger_codes: health1.trigger_codes,
    }),
  ];

  return {
    run,
    decisions: [decisionWeek0, decisionWeek1, decisionPrepared],
    predictions: [prediction0, prediction1],
    latestModel: model2,
    events,
    explanations: {
      [week0Id]: {
        decision: decisionWeek0,
        world_model: model1,
        prediction: prediction0,
        model_health_signals: [health0],
      },
      [week1Id]: {
        decision: decisionWeek1,
        world_model: model2,
        prediction: prediction1,
        model_health_signals: [health1],
      },
    } as Record<string, unknown>,
    report: {
      run,
      decision_count: 3,
      prediction_count: 2,
      matured_outcome_count: 2,
      world_model_version: 2,
      latest_model_health: health1,
    },
  };
}

export type Fixture = ReturnType<typeof fixtures>;

/** Routes the hand-rolled fetch client at an in-memory fixture, mirroring the real API's 404s. */
export function installApi(fixture: Fixture) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const path = new URL(String(input)).pathname;
    const json = (body: unknown, status = 200) =>
      new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

    if (path === "/runs" && init?.method === "POST") return json(fixture.run, 201);
    if (path === `/runs/${runId}` || path === `/runs/${runId}/state`) return json(fixture.run);
    if (path === `/runs/${runId}/events`) return json(fixture.events);
    if (path === `/runs/${runId}/decisions`) return json(fixture.decisions);
    if (path === `/runs/${runId}/world-model`) return json(fixture.latestModel);
    if (path === `/runs/${runId}/predictions`) return json(fixture.predictions);
    if (path === `/runs/${runId}/report`) return json(fixture.report);

    const decisionMatch = path.match(new RegExp(`^/runs/${runId}/decisions/([^/]+)$`));
    if (decisionMatch) {
      const explanation = fixture.explanations[decisionMatch[1]];
      // The API answers 404 for decisions that have not committed their artifacts yet.
      return explanation ? json(explanation) : json({ detail: "Decision has no explanation" }, 404);
    }
    return json({ detail: `Unhandled ${path}` }, 404);
  });
}
