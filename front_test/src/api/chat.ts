import type { ChatResponse } from "../types";
import { request, getAuthToken, GATEWAY_WS } from "./client";

// NOTE: Text chat is now handled via the gateway WebSocket (inject_query).
// This module is kept for potential REST fallback.

export const chatApi = {
  async sendMessage(sessionId: string, content: string): Promise<ChatResponse> {
    const base = GATEWAY_WS.replace(/^ws/, "http");
    return request(`${base}/chat`, {
      method: "POST",
      body: JSON.stringify({ message: content }),
    });
  },

  streamMessage(
    sessionId: string,
    content: string,
    onChunk: (text: string) => void,
    onDone: () => void,
    onError: (err: Error) => void
  ): AbortController {
    const controller = new AbortController();
    // Not used — voice pipeline handles all AI interaction
    onDone();
    return controller;
  },
};
