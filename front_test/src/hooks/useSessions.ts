import { useState, useEffect, useCallback } from "react";
import { sessionsApi } from "../api";
import type { Session } from "../types";

export function useSessions(userId: string | null) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  const fetchSessions = useCallback(async () => {
    if (!userId) return;
    try {
      const data = await sessionsApi.list(userId);
      setSessions(data);
    } catch {
      console.error("Failed to fetch sessions");
    }
  }, [userId]);

  useEffect(() => {
    if (userId) fetchSessions();
  }, [userId, fetchSessions]);

  const createSession = useCallback(async (title: string) => {
    if (!userId) return;
    const newSession = await sessionsApi.create(userId, title);
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    return newSession;
  }, [userId]);

  const deleteSession = useCallback(async (id: string) => {
    await sessionsApi.delete(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
    setCurrentSessionId((prev) => (prev === id ? null : prev));
  }, []);

  const renameSession = useCallback(async (id: string, title: string) => {
    const updated = await sessionsApi.rename(id, title);
    setSessions((prev) => prev.map((s) => (s.id === id ? updated : s)));
    return updated;
  }, []);

  const reset = useCallback(() => {
    setSessions([]);
    setCurrentSessionId(null);
  }, []);

  return {
    sessions,
    currentSessionId,
    setCurrentSessionId,
    createSession,
    deleteSession,
    renameSession,
    refreshSessions: fetchSessions,
    reset,
  };
}
