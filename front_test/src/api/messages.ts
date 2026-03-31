import type { Message } from "../types";
import { request, MESSAGE_URL } from "./client";

export const messagesApi = {
  async list(sessionId: string): Promise<Message[]> {
    return request(`${MESSAGE_URL}/sessions/${sessionId}/messages`);
  },

  async send(sessionId: string, role: string, content: string): Promise<Message> {
    return request(`${MESSAGE_URL}/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ role, content }),
    });
  },
};
