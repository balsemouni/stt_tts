import type { User } from "../types";
import { request, setAuthToken, AUTH_URL } from "./client";

export const authApi = {
  async login(email: string, password: string): Promise<User> {
    const tokens = await request<{ access_token: string; refresh_token: string }>(
      `${AUTH_URL}/auth/login`,
      { method: "POST", body: JSON.stringify({ email, password }) }
    );
    setAuthToken(tokens.access_token);
    localStorage.setItem("refresh_token", tokens.refresh_token);
    return request<User>(`${AUTH_URL}/auth/me`);
  },

  async signup(email: string, username: string, password: string): Promise<User> {
    return request(`${AUTH_URL}/auth/register`, {
      method: "POST",
      body: JSON.stringify({ email, username, password }),
    });
  },

  async logout(): Promise<void> {
    const refreshToken = localStorage.getItem("refresh_token");
    try {
      await request(`${AUTH_URL}/auth/logout`, {
        method: "POST",
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    } finally {
      setAuthToken(null);
      localStorage.removeItem("refresh_token");
    }
  },

  async me(): Promise<User> {
    return request(`${AUTH_URL}/auth/me`);
  },
};
