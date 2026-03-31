import { useState, useCallback, useRef } from "react";
import { chatApi } from "../api";

export function useChat(sessionId: string | null) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState("");
  const controllerRef = useRef<AbortController | null>(null);

  // Standard single-response chat
  const sendChat = useCallback(
    async (content: string) => {
      if (!sessionId) return null;
      const data = await chatApi.sendMessage(sessionId, content);
      return data.reply;
    },
    [sessionId]
  );

  // Streaming chat
  const sendChatStream = useCallback(
    (content: string, onChunk?: (text: string) => void, onDone?: () => void) => {
      if (!sessionId) return;
      setIsStreaming(true);
      setStreamedText("");

      controllerRef.current = chatApi.streamMessage(
        sessionId,
        content,
        (text) => {
          setStreamedText((prev) => prev + text);
          onChunk?.(text);
        },
        () => {
          setIsStreaming(false);
          onDone?.();
        },
        (err) => {
          console.error("Stream error:", err);
          setIsStreaming(false);
        }
      );
    },
    [sessionId]
  );

  const cancelStream = useCallback(() => {
    controllerRef.current?.abort();
    setIsStreaming(false);
  }, []);

  return { sendChat, sendChatStream, cancelStream, isStreaming, streamedText };
}
