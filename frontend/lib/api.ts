/**
 * api.ts — HealthGuard XAI API client
 *
 * Reads the backend base URL from NEXT_PUBLIC_API_URL.
 * For local dev set it to http://localhost:8000 in .env.local
 * For production set it to your Render deployment URL.
 */

const BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

async function post<T>(path: string, body: unknown, token?: string | null): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000); // 30s for cold-start

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const apiKey = process.env.NEXT_PUBLIC_HEALTHGUARD_API_KEY;
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: controller.signal,
  }).finally(() => clearTimeout(timeout));

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${path} failed [${res.status}]: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Shared Types ──────────────────────────────────────────────────────────────

export interface Biomarkers {
  Pregnancies: number;
  Glucose: number;
  BloodPressure: number;
  SkinThickness: number;
  Insulin: number;
  BMI: number;
  DiabetesPedigreeFunction: number;
  Age: number;
}

// ── /api/predict ──────────────────────────────────────────────────────────────

export interface PredictResponse {
  risk_probability: number;       // 0-1
  risk_level: 'Low' | 'Medium' | 'High';
  top_features: Record<string, number>;
}

export function predictRisk(data: Biomarkers, token?: string | null): Promise<PredictResponse> {
  return post<PredictResponse>('/api/predict', data, token);
}

// ── /api/explain ──────────────────────────────────────────────────────────────

export interface ExplainResponse {
  feature_importances: Record<string, number>; // SHAP values (can be negative)
  top_positive_risk_factors: string[];          // e.g. "Glucose (138.00)"
  protective_factors: string[];                 // e.g. "BloodPressure (72.00)"
}

export function getExplanation(data: Biomarkers, token?: string | null): Promise<ExplainResponse> {
  return post<ExplainResponse>('/api/explain', data, token);
}

// ── /api/recommend ────────────────────────────────────────────────────────────

export interface RecommendRequest {
  biomarkers: Biomarkers;
  predicted_risks: Record<string, number>;       // { diabetes: 0.72 }
  top_positive_risk_factors: string[];
  protective_factors: string[];
  use_llm?: boolean;
}

export interface Recommendation {
  category: string;
  recommendation: string;
  rationale: string;
  priority: string;
  evidence_level?: string;
}

export interface RecommendResponse {
  patient_context: Record<string, unknown>;
  prioritized_recommendations: Recommendation[];
  general_wellness_advice: string[];
  disclaimer: string;
}

export function getRecommendations(data: RecommendRequest, token?: string | null): Promise<RecommendResponse> {
  return post<RecommendResponse>('/api/recommend', data, token);
}

// ── /api/chat ─────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  patient_data: Record<string, unknown>;
  history: ChatMessage[];
}

export interface ChatResponse {
  assistant_response: string;
  safety_flag: boolean;
  escalation_required: boolean;
}

export function sendChatMessage(data: ChatRequest, token?: string | null): Promise<ChatResponse> {
  return post<ChatResponse>('/api/chat', data, token);
}
