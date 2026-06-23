import { requestJson } from '../lib/http'
import type { ModelSummary, PredictRequest, PredictResponse } from '../types/prediction'

// Fetch model metadata before rendering the inference form. The response tells
// the UI which fields are required, how they should be labeled, and what labels
// the probability distribution can contain.
export async function getModels(): Promise<ModelSummary[]> {
  return requestJson<ModelSummary[]>('/models')
}

// Send one completed student record to the backend. The backend owns all final
// preprocessing and model execution, while the frontend only packages the form
// state into the shared API contract.
export async function runPrediction(payload: PredictRequest): Promise<PredictResponse> {
  return requestJson<PredictResponse>('/predict', {
    method: 'POST',
    body: payload,
  })
}
