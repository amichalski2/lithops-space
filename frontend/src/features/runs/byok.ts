/** Participant credentials intentionally live in process memory only. */
let geminiApiKey: string | null = null;

export function setGeminiApiKey(value: string): void {
  const normalized = value.trim();
  geminiApiKey = normalized.length > 0 ? normalized : null;
}

export function getGeminiApiKey(): string | null {
  return geminiApiKey;
}

export function clearGeminiApiKey(): void {
  geminiApiKey = null;
}
