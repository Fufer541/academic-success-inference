// Values that can be represented in JSON and accepted by the backend feature
// map. The backend performs final numeric/categorical validation against the
// trained bundle's schema.
export type PrimitiveFeatureValue = string | number | boolean | null

// Flat map of model input feature name -> raw value.
export type PredictionFeatures = Record<string, PrimitiveFeatureValue>

// `unknown` is reserved for forward compatibility if a future bundle contains a
// field not identified as categorical or numeric in its metadata.
export type FeatureKind = 'categorical' | 'numeric' | 'unknown'

// One dynamically rendered form field. The frontend receives this list from
// `/models` instead of hard-coding the trained model's full schema.
export type FeatureSummary = {
  name: string
  kind: FeatureKind
  required: boolean
}

export type PredictRequest = {
  // Backend model id selected by the user. The current app has one model, but
  // the request shape is ready for additional trained artifacts.
  model: string

  // Feature names must match the names returned by `/models`. Values may still
  // be strings from form controls; backend preprocessing coerces them safely.
  features: PredictionFeatures
}


export type PseudoLabelingSummary = {
  // Whether the research notebook added confident `Enrolled` rows back into the
  // terminal-outcome training set before fitting the final binary ensemble.
  enabled?: boolean

  // Confidence cutoff used by the notebook when accepting pseudo-labels.
  threshold?: number

  // Original unresolved label and row counts from the semi-supervised step.
  unresolvedLabel?: string
  unresolvedRows?: number

  // Count of high-confidence pseudo-labels added to the supervised training
  // data, plus the terminal class split among those accepted rows.
  selectedRows?: number
  selectedDropout?: number
  selectedGraduate?: number

  // Optional confidence diagnostics saved by the notebook for quick inspection.
  confidenceMin?: number
  confidenceMean?: number
}

export type PredictResponse = {
  // Echoes the model that served the prediction.
  model: string

  // Winning class label after ensemble probability blending.
  prediction: string

  // Probability assigned to the winning class.
  confidence?: number

  // Full probability distribution, keyed by class label.
  probabilities: Record<string, number>

  // Backend-provided version marker. This currently includes the model id and
  // phase so UI/debug output can distinguish enrollment-phase predictions.
  modelVersion?: string
}

export type ModelSummary = {
  id: string
  name: string
  description?: string

  // Training phase represented by the bundle, for example `enrollment`.
  phase?: string

  // Research architecture details. These fields let the UI confirm it is
  // connected to the binary terminal-outcome bundle rather than an old
  // multiclass artifact where `Enrolled` was a regular class.
  problemType?: string
  unresolvedLabel?: string
  pseudoLabeling?: PseudoLabelingSummary

  // Ordered feature metadata. The order matches the training DataFrame and is
  // used by the UI to present a stable form.
  features: FeatureSummary[]

  // Class labels in the same order used by backend probability alignment.
  labels: string[]

  // Optional metrics table serialized from the research notebook bundle.
  metrics?: Record<string, Record<string, number>>
}