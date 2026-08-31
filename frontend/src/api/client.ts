import type { components } from "./generated/schema";

export type RunRecord = components["schemas"]["RunRecord"];
export type EventRecord = components["schemas"]["EventRecord"];
export type StepResult = components["schemas"]["StepResult"];
export type DecisionRecord = components["schemas"]["DecisionRecord"];
export type DecisionExplanation = components["schemas"]["DecisionExplanation"];
export type WorldModelVersion = components["schemas"]["WorldModelVersion"];
export type PredictionView = components["schemas"]["PredictionView"];
export type RunReport = components["schemas"]["RunReport"];
export type ModelHealthSignal = components["schemas"]["ModelHealthSignal"];
export type CandidateEvaluationRecord = components["schemas"]["CandidateEvaluationRecord"];
export type CashPredictionTarget = components["schemas"]["CashPredictionTarget"];
export type PredictionOutcome = components["schemas"]["PredictionOutcome"];
export type WorldModelParameter = components["schemas"]["WorldModelParameter"];

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(body || `Lithops API returned ${response.status}`, response.status);
  }
  return response.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function optionalRequest<T>(path: string): Promise<T | null> {
  try {
    return await request<T>(path);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

export const lithopsApi = {
  createRun: () => request<RunRecord>("/runs", { method: "POST" }),
  getRun: (runId: string) => request<RunRecord>(`/runs/${runId}`),
  getRunState: (runId: string) => request<RunRecord>(`/runs/${runId}/state`),
  startRun: (runId: string) => request<RunRecord>(`/runs/${runId}/start`, { method: "POST" }),
  pauseRun: (runId: string) => request<RunRecord>(`/runs/${runId}/pause`, { method: "POST" }),
  resumeRun: (runId: string) => request<RunRecord>(`/runs/${runId}/resume`, { method: "POST" }),
  listDecisions: (runId: string) => request<DecisionRecord[]>(`/runs/${runId}/decisions`),
  getDecision: (runId: string, decisionId: string) =>
    request<DecisionExplanation>(`/runs/${runId}/decisions/${decisionId}`),
  /** Returns null for decisions still in `prepared`, which the API answers with 404 by design. */
  getDecisionIfExplained: (runId: string, decisionId: string) =>
    optionalRequest<DecisionExplanation>(`/runs/${runId}/decisions/${decisionId}`),
  getWorldModel: (runId: string) =>
    optionalRequest<WorldModelVersion>(`/runs/${runId}/world-model`),
  listPredictions: (runId: string) =>
    request<PredictionView[]>(`/runs/${runId}/predictions`),
  listEvents: (runId: string) => request<EventRecord[]>(`/runs/${runId}/events`),
  getReport: (runId: string) => request<RunReport>(`/runs/${runId}/report`),
};
