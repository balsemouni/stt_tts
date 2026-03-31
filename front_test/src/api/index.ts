// ─── API Layer ──────────────────────────────────────────────────────────────
// Barrel export — import everything from "api/" in one line.

export { authApi } from "./auth";
export { sessionsApi } from "./sessions";
export { messagesApi } from "./messages";
export { chatApi } from "./chat";
export { metricsApi } from "./metrics";
export { getVoiceWebSocketUrl } from "./voice";
export { setAuthToken, getAuthToken, GATEWAY_WS } from "./client";
