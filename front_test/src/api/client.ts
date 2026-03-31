// ─── Service URLs ──────────────────────────────────────────────────────────
export const AUTH_URL    = import.meta.env.VITE_AUTH_URL    || "http://localhost:8006";
export const SESSION_URL = import.meta.env.VITE_SESSION_URL || "http://localhost:8005";
export const MESSAGE_URL = import.meta.env.VITE_MESSAGE_URL || "http://localhost:8003";
export const GATEWAY_WS  = import.meta.env.VITE_GATEWAY_WS  || "ws://localhost:8090";

// ─── Token Management ──────────────────────────────────────────────────────
let authToken: string | null = localStorage.getItem("auth_token");

export function setAuthToken(token: string | null) {
  authToken = token;
  if (token) {
    localStorage.setItem("auth_token", token);
  } else {
    localStorage.removeItem("auth_token");
  }
}

export function getAuthToken(): string | null {
  return authToken;
}

// ─── Base Request Helper ───────────────────────────────────────────────────
export async function request<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }

  const res = await fetch(url, { ...options, headers });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    const err = new Error(body.detail || body.error || body.message || `Request failed (${res.status})`);
    (err as any).status = res.status;
    (err as any).body = body;
    throw err;
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}
