// dashboard/src/api.js
// All fetch calls go through the Vite /api proxy → FastAPI :8000

const BASE = '/api'

export async function fetchHealth() {
  const r = await fetch(`${BASE}/health`)
  if (!r.ok) throw new Error('API offline')
  return r.json()
}

export async function fetchMetrics() {
  const r = await fetch(`${BASE}/metrics`)
  if (!r.ok) throw new Error('metrics fetch failed')
  return r.json()
}

export async function fetchModelInfo() {
  const r = await fetch(`${BASE}/model/info`)
  if (!r.ok) throw new Error('model/info fetch failed')
  return r.json()
}

export async function fetchTimeseries(limit = 500) {
  const r = await fetch(`${BASE}/timeseries?limit=${limit}`)
  if (!r.ok) throw new Error('timeseries fetch failed')
  return r.json()
}

export async function postPredict(features) {
  const r = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(err.detail || 'prediction failed')
  }
  return r.json()
}
