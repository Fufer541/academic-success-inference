import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'

import { getModels, runPrediction } from '../api/predictions'
import { BINARY_LABELS, CATEGORY_LABELS } from '../config/categories'
import type {
  FeatureSummary,
  ModelSummary,
  PredictResponse,
  PredictionFeatures,
  PrimitiveFeatureValue,
} from '../types/prediction'

type FormValues = Record<string, string | number>
type FeatureGroup = { title: string; code: string; names: string[] }

const FEATURE_GROUPS: FeatureGroup[] = [
  { code: '0-01', title: 'application', names: ['Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Admission grade'] },
  { code: '0-02', title: 'prior education', names: ['Previous qualification', 'Previous qualification (grade)'] },
  { code: '0-03', title: 'family background', names: ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"] },
  { code: '0-04', title: 'student profile', names: ['Marital status', 'Nationality', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International'] },
  { code: '0-05', title: 'economic context', names: ['Unemployment rate', 'Inflation rate', 'GDP'] },
]

const BINARY_FEATURES = new Set(['Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'])

const SAMPLE_FEATURES: PredictionFeatures = {
  'Marital status': 1, 'Application mode': 17, 'Application order': 5, Course: 171,
  'Daytime/evening attendance': 1, 'Previous qualification': 1, 'Previous qualification (grade)': 122,
  Nationality: 1, "Mother's qualification": 19, "Father's qualification": 12,
  "Mother's occupation": 5, "Father's occupation": 9, 'Admission grade': 127.3,
  Displaced: 1, 'Educational special needs': 0, Debtor: 0, 'Tuition fees up to date': 1,
  Gender: 1, 'Scholarship holder': 0, 'Age at enrollment': 20, International: 0,
  'Unemployment rate': 10.8, 'Inflation rate': 1.4, GDP: 1.74,
}

function isBinary(name: string) { return BINARY_FEATURES.has(name) }
function emptyForm(features: FeatureSummary[]): FormValues {
  return Object.fromEntries(features.map(f => [f.name, isBinary(f.name) ? 0 : ''])) as FormValues
}
function sampleForm(features: FeatureSummary[]): FormValues {
  return Object.fromEntries(features.map(f => [f.name, SAMPLE_FEATURES[f.name] ?? (isBinary(f.name) ? 0 : '')])) as FormValues
}
function groupFeatures(features: FeatureSummary[]) {
  const used = new Set<string>()
  const groups = FEATURE_GROUPS.map(g => {
    const fields = g.names.map(n => features.find(f => f.name === n)).filter((f): f is FeatureSummary => Boolean(f))
    fields.forEach(f => used.add(f.name))
    return { ...g, fields }
  }).filter(g => g.fields.length > 0)
  const remaining = features.filter(f => !used.has(f.name))
  if (remaining.length > 0) groups.push({ code: '0-06', title: 'additional inputs', names: [], fields: remaining })
  return groups
}
function pct(v?: number) {
  if (typeof v !== 'number') return '--'
  return `${Math.round(v * 100)}%`
}

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body, #root { background: #111; color: #d4d4d4; font-family: 'Inter', sans-serif; font-size: 14px; min-height: 100vh; }

  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes barGrow {
    from { width: 0%; }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  @keyframes resultIn {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .app { max-width: 1280px; margin: 0 auto; padding: 32px 24px; }

  .header {
    display: flex; align-items: flex-start; justify-content: space-between;
    border-bottom: 1px dashed #2a2a2a; padding-bottom: 24px; margin-bottom: 32px;
    gap: 16px; flex-wrap: wrap;
    animation: fadeSlideDown 0.5s ease forwards;
  }
  .header-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #555; margin-bottom: 6px; }
  .header-title { font-size: 28px; font-weight: 300; color: #e8e8e8; letter-spacing: -0.02em; }
  .header-tags { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .tag { font-size: 11px; color: #555; border: 1px dashed #2a2a2a; padding: 4px 10px; letter-spacing: 0.05em; }

  .layout { display: grid; grid-template-columns: 1fr 340px; gap: 16px; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

  .card {
    border: 1px dashed #2a2a2a; background: #141414;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    animation: fadeSlideUp 0.5s ease forwards;
  }
  .card:hover { border-color: #333; transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }

  .card-header { padding: 20px 24px 16px; border-bottom: 1px dashed #2a2a2a; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
  .card-title { font-size: 13px; color: #888; font-weight: 400; }
  .card-subtitle { font-size: 11px; color: #444; margin-top: 2px; }
  .actions { display: flex; gap: 8px; }

  .btn {
    background: transparent; border: 1px dashed #333; color: #888;
    font-size: 12px; padding: 6px 14px; cursor: pointer;
    font-family: inherit; letter-spacing: 0.05em;
    transition: all 0.15s; position: relative; overflow: hidden;
  }
  .btn::before { content: '[ '; }
  .btn::after { content: ' ]'; }
  .btn:hover:not(:disabled) { color: #d4d4d4; border-color: #555; }
  .btn:active:not(:disabled) { transform: scale(0.97); }
  .btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .btn-primary { border-style: solid; border-color: #d4d4d4; color: #d4d4d4; }
  .btn-primary:hover:not(:disabled) { background: #d4d4d4; color: #111; }
  .btn-primary:disabled { border-color: #333; color: #444; }
  .btn-loading { animation: pulse 1s ease infinite; }

  .form-body { padding: 24px; }
  .group { margin-bottom: 32px; }
  .group:last-child { margin-bottom: 0; }
  .group-enter {
    opacity: 0;
    animation: fadeSlideUp 0.4s ease forwards;
  }

  .group-label { font-size: 11px; letter-spacing: 0.08em; color: #444; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .group-code { color: #333; }
  .fields { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field-label { font-size: 11px; color: #555; display: flex; justify-content: space-between; align-items: center; }
  .field-kind { font-size: 10px; color: #333; letter-spacing: 0.05em; }

  input[type="number"], select {
    background: #111; border: 1px dashed #2a2a2a; color: #d4d4d4;
    font-family: inherit; font-size: 13px; padding: 8px 10px; width: 100%;
    outline: none; transition: border-color 0.2s, background 0.2s;
    appearance: none; -webkit-appearance: none;
  }
  input[type="number"]:focus, select:focus { border-color: #555; border-style: solid; background: #161616; }
  input[type="number"]:hover, select:hover { border-color: #333; }
  select option { background: #1a1a1a; color: #d4d4d4; }

  .toggle { display: grid; grid-template-columns: 1fr 1fr; border: 1px dashed #2a2a2a; overflow: hidden; }
  .toggle-btn {
    background: transparent; border: none; color: #555;
    font-family: inherit; font-size: 12px; padding: 8px 0;
    cursor: pointer; transition: all 0.2s; text-align: center;
    border-right: 1px dashed #2a2a2a;
  }
  .toggle-btn:last-child { border-right: none; }
  .toggle-btn.active { background: #d4d4d4; color: #111; font-weight: 500; }
  .toggle-btn:not(.active):hover { color: #d4d4d4; background: #1a1a1a; }

  .side { display: flex; flex-direction: column; gap: 16px; }

  .prediction-empty { padding: 48px 24px; display: flex; flex-direction: column; align-items: center; gap: 12px; }
  .prediction-empty-icon { width: 40px; height: 40px; border: 1px dashed #2a2a2a; display: flex; align-items: center; justify-content: center; color: #333; font-size: 18px; transition: border-color 0.2s, color 0.2s; }
  .prediction-empty-icon:hover { border-color: #444; color: #555; }
  .prediction-empty-text { font-size: 12px; color: #444; letter-spacing: 0.05em; }

  .prediction-result { padding: 24px; animation: resultIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
  .prediction-outcome { border-left: 2px solid #d4d4d4; padding: 16px 20px; margin-bottom: 24px; background: #111; }
  .prediction-label { font-size: 11px; color: #444; margin-bottom: 6px; letter-spacing: 0.05em; }
  .prediction-value { font-size: 32px; font-weight: 300; color: #e8e8e8; letter-spacing: -0.02em; margin-bottom: 4px; }
  .prediction-confidence { font-size: 12px; color: #555; }

  .proba-item { margin-bottom: 16px; }
  .proba-item:last-child { margin-bottom: 0; }
  .proba-row { display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-bottom: 6px; }
  .proba-bar-bg { height: 1px; background: #2a2a2a; position: relative; }
  .proba-bar {
    height: 1px; background: #d4d4d4;
    position: absolute; top: 0; left: 0;
    animation: barGrow 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards;
  }

  .model-body { padding: 24px; }
  .model-row { display: flex; justify-content: space-between; align-items: baseline; padding: 10px 0; border-bottom: 1px dashed #1e1e1e; gap: 12px; transition: background 0.15s; }
  .model-row:hover { background: #161616; padding-left: 4px; padding-right: 4px; }
  .model-row:last-child { border-bottom: none; }
  .model-key { font-size: 12px; color: #444; }
  .model-val { font-size: 12px; color: #888; text-align: right; }
  .model-val.highlight { color: #d4d4d4; font-weight: 500; }

  .error-box { margin: 16px 24px; padding: 12px 16px; border: 1px dashed #444; font-size: 12px; color: #888; animation: fadeSlideUp 0.3s ease forwards; }
  .loading-box { padding: 48px 24px; font-size: 12px; color: #333; text-align: center; letter-spacing: 0.05em; animation: pulse 1.5s ease infinite; }
`

export function AppShell() {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [activeModelId, setActiveModelId] = useState('')
  const [formValues, setFormValues] = useState<FormValues>({})
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [predKey, setPredKey] = useState(0)
  const [loadError, setLoadError] = useState('')
  const [submitError, setSubmitError] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    let mounted = true
    getModels()
      .then(ms => {
        if (!mounted) return
        setModels(ms)
        const m = ms[0]
        if (m) { setActiveModelId(m.id); setFormValues(emptyForm(m.features)) }
      })
      .catch(e => { if (mounted) setLoadError(e instanceof Error ? e.message : 'error') })
      .finally(() => { if (mounted) setIsLoading(false) })
    return () => { mounted = false }
  }, [])

  const activeModel = useMemo(() => models.find(m => m.id === activeModelId) ?? null, [activeModelId, models])
  const grouped = useMemo(() => groupFeatures(activeModel?.features ?? []), [activeModel])

  const missingCount = useMemo(() => {
    if (!activeModel) return 0
    return activeModel.features.filter(f => {
      if (!f.required) return false
      const v = formValues[f.name]
      return v === undefined || (typeof v === 'string' && v.trim() === '')
    }).length
  }, [activeModel, formValues])

  const testMetrics = activeModel?.metrics?.['Soft vote test']
  const pseudo = activeModel?.pseudoLabeling

  function update(name: string, value: string | number) {
    setFormValues(c => ({ ...c, [name]: value }))
    setPrediction(null)
    setSubmitError('')
  }

  function buildPayload(model: ModelSummary): PredictionFeatures {
    return model.features.reduce<PredictionFeatures>((acc, f) => {
      const v = formValues[f.name]
      let val: PrimitiveFeatureValue = v ?? null
      if (typeof v === 'string') {
        const t = v.trim()
        val = f.kind === 'numeric' ? Number(t) : t
      }
      acc[f.name] = val
      return acc
    }, {})
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!activeModel || missingCount > 0) return
    setIsSubmitting(true)
    setSubmitError('')
    try {
      const r = await runPrediction({ model: activeModel.id, features: buildPayload(activeModel) })
      setPrediction(r)
      setPredKey(k => k + 1)
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'prediction failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  function renderField(f: FeatureSummary) {
    const v = formValues[f.name] ?? ''
    const id = `f-${f.name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`
    const labelEl = (
      <div className="field-label">
        <span>{f.name.toLowerCase()}</span>
        <span className="field-kind">{f.kind}</span>
      </div>
    )
    if (isBinary(f.name)) {
      const opts = BINARY_LABELS[f.name] ?? ['0', '1']
      return (
        <div className="field" key={f.name}>
          {labelEl}
          <div className="toggle">
            {[0, 1].map(opt => (
              <button key={opt} type="button" className={`toggle-btn${Number(v) === opt ? ' active' : ''}`} onClick={() => update(f.name, opt)}>
                {opts[opt].toLowerCase()}
              </button>
            ))}
          </div>
        </div>
      )
    }
    const cats = CATEGORY_LABELS[f.name]
    if (cats) {
      return (
        <label key={f.name} className="field" htmlFor={id}>
          {labelEl}
          <select id={id} value={v === '' ? '' : String(v)} onChange={e => update(f.name, e.target.value)}>
            <option disabled value="">select...</option>
            {Object.entries(cats).map(([code, label]) => (
              <option key={code} value={code}>{label.toLowerCase()}</option>
            ))}
          </select>
        </label>
      )
    }
    return (
      <label key={f.name} className="field" htmlFor={id}>
        {labelEl}
        <input id={id} type="number" value={v} step={f.kind === 'numeric' ? '0.01' : '1'} inputMode="decimal" onChange={e => update(f.name, e.target.value)} />
      </label>
    )
  }

  return (
    <>
      <style>{css}</style>
      <div className="app">
        <header className="header">
          <div>
            <div className="header-label">/ binary terminal model /</div>
            <h1 className="header-title">academic success inference</h1>
          </div>
          <div className="header-tags">
            <span className="tag">enrollment phase</span>
            <span className="tag">24 features</span>
            <span className="tag">dropout / graduate</span>
          </div>
        </header>

        <div className="layout">
          <div className="card" style={{ animationDelay: '0.1s' }}>
            <div className="card-header">
              <div>
                <div className="card-title">student inputs</div>
                <div className="card-subtitle">{activeModel?.name?.toLowerCase() ?? 'loading...'}</div>
              </div>
              <div className="actions">
                <button className="btn" type="button" disabled={!activeModel || isSubmitting}
                  onClick={() => { if (activeModel) { setFormValues(sampleForm(activeModel.features)); setPrediction(null) } }}>
                  sample
                </button>
                <button className="btn" type="button" disabled={!activeModel || isSubmitting}
                  onClick={() => { if (activeModel) { setFormValues(emptyForm(activeModel.features)); setPrediction(null) } }}>
                  reset
                </button>
                <button
                  className={`btn btn-primary${isSubmitting ? ' btn-loading' : ''}`}
                  type="submit" form="predict-form"
                  disabled={!activeModel || missingCount > 0 || isSubmitting}
                >
                  {isSubmitting ? 'running...' : 'predict'}
                </button>
              </div>
            </div>
            {isLoading ? (
              <div className="loading-box">loading model...</div>
            ) : loadError ? (
              <div className="error-box">{loadError}</div>
            ) : (
              <form id="predict-form" className="form-body" onSubmit={handleSubmit}>
                {grouped.map((g, i) => (
                  <div className="group group-enter" key={g.code} style={{ animationDelay: `${i * 0.08}s` }}>
                    <div className="group-label">
                      <span className="group-code">/ {g.code} /</span>
                      <span>{g.title}</span>
                    </div>
                    <div className="fields">
                      {g.fields.map(renderField)}
                    </div>
                  </div>
                ))}
                {submitError && <div className="error-box">{submitError}</div>}
              </form>
            )}
          </div>

          <div className="side">
            <div className="card" style={{ animationDelay: '0.2s' }}>
              <div className="card-header">
                <div className="card-title">prediction</div>
              </div>
              {prediction ? (
                <div className="prediction-result" key={predKey}>
                  <div className="prediction-outcome">
                    <div className="prediction-label">/ estimated terminal outcome /</div>
                    <div className="prediction-value">{prediction.prediction.toLowerCase()}</div>
                    <div className="prediction-confidence">confidence {pct(prediction.confidence)}</div>
                  </div>
                  <div>
                    {Object.entries(prediction.probabilities).map(([label, prob], i) => (
                      <div className="proba-item" key={label} style={{ animationDelay: `${i * 0.1}s` }}>
                        <div className="proba-row">
                          <span>{label.toLowerCase()}</span>
                          <span>{pct(prob)}</span>
                        </div>
                        <div className="proba-bar-bg">
                          <div className="proba-bar" style={{ width: pct(prob), animationDelay: `${0.2 + i * 0.1}s` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="prediction-empty">
                  <div className="prediction-empty-icon">?</div>
                  <div className="prediction-empty-text">
                    {missingCount > 0 ? `/ ${missingCount} fields remaining /` : '/ ready /'}
                  </div>
                </div>
              )}
            </div>

            <div className="card" style={{ animationDelay: '0.3s' }}>
              <div className="card-header">
                <div className="card-title">model</div>
              </div>
              <div className="model-body">
                <div className="model-row"><span className="model-key">features</span><span className="model-val highlight">{activeModel?.features.length ?? 0}</span></div>
                <div className="model-row"><span className="model-key">labels</span><span className="model-val highlight">{activeModel?.labels.join(' / ').toLowerCase() ?? '--'}</span></div>
                <div className="model-row"><span className="model-key">pseudo labels</span><span className="model-val">{pseudo?.enabled ? `${pseudo.selectedRows ?? 0}/${pseudo.unresolvedRows ?? 0} @ ${pct(pseudo.threshold)}` : 'off'}</span></div>
                <div className="model-row"><span className="model-key">test accuracy</span><span className="model-val highlight">{pct(testMetrics?.accuracy)}</span></div>
                <div className="model-row"><span className="model-key">roc auc</span><span className="model-val highlight">{pct(testMetrics?.roc_auc)}</span></div>
                <div className="model-row"><span className="model-key">macro f1</span><span className="model-val highlight">{pct(testMetrics?.f1_macro)}</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}