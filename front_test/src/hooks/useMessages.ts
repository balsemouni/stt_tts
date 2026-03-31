import { useState, useEffect, useCallback } from "react";
import { messagesApi } from "../api";
import type { Message } from "../types";

export function useMessages(sessionId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);

  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }
    messagesApi
      .list(sessionId)
      .then(setMessages)
      .catch(() => console.error("Failed to fetch messages"));
  }, [sessionId]);

  const addMessage = useCallback((msg: Message) => {
    setMessages((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].role === msg.role) {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: updated[updated.length - 1].content + " " + msg.content,
        };
        return updated;
      }
      return [...prev, msg];
    });
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, addMessage, clearMessages };
}
