import { env } from '../config/env'

// Narrow RequestInit so callers can pass plain JavaScript objects for JSON
// bodies. The helper owns JSON.stringify, which keeps API modules small and
// prevents each call site from repeating the same boilerplate.
type RequestOptions = Omit<RequestInit, 'body'> & {
  body?: unknown
}

// Shared JSON fetch wrapper for all backend calls.
//
// `path` is intentionally relative to `VITE_API_BASE_URL`; this keeps browser
// code portable between local Vite development, Docker Compose, and any future
// hosted backend.
export async function requestJson<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  // Prefix relative paths with the configured API base URL.
  const response = await fetch(`${env.apiBaseUrl}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  })

  if (!response.ok) {
    // FastAPI returns useful error details in a JSON `{ detail: ... }` shape.
    // Pulling that text into the thrown Error lets the form show actionable
    // validation messages instead of a generic HTTP status.
    const body = await response.text()
    let detail = response.statusText
    if (body) {
      try {
        const parsed = JSON.parse(body) as { detail?: unknown }
        detail = typeof parsed.detail === 'string' ? parsed.detail : body
      } catch {
        detail = body
      }
    }
    throw new Error(`HTTP ${response.status}: ${detail}`)
  }

  // The generic return type ties each API wrapper to its expected contract while
  // keeping this utility framework-agnostic.
  return (await response.json()) as T
}