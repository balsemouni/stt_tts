import type { Session } from "../types";
import { request, SESSION_URL } from "./client";

export const sessionsApi = {
  async list(userId: string): Promise<Session[]> {
    return request(`${SESSION_URL}/users/${userId}/sessions`);
  },

  async create(userId: string, title?: string): Promise<Session> {
    return request(`${SESSION_URL}/sessions`, {
      method: "POST",
      body: JSON.stringify({ user_id: userId, title }),
    });
  },

  async delete(id: string): Promise<void> {
    await request(`${SESSION_URL}/sessions/${id}`, { method: "DELETE" });
  },

  async rename(id: string, title: string): Promise<Session> {
    return request(`${SESSION_URL}/sessions/${id}/title`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
  },
};
