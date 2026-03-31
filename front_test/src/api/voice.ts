import { getAuthToken, GATEWAY_WS } from "./client";

export function getVoiceWebSocketUrl(sessionId: string): string {
  const token = getAuthToken();
  const params = new URLSearchParams();
  if (token) params.set("token", token);
  if (sessionId) params.set("session_id", sessionId);
  return `${GATEWAY_WS}/ws?${params.toString()}`;
}
