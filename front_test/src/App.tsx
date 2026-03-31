import React, { useState } from "react";
import { Auth } from "./components/Auth";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { SplashScreen } from "./components/SplashScreen";
import { useAuth, useSessions, useMessages } from "./hooks";

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const { user, setUser, loading, logout } = useAuth();
  const { sessions, currentSessionId, setCurrentSessionId, createSession, reset } = useSessions(user?.id || null);
  const { messages, addMessage } = useMessages(currentSessionId);

  const handleNewSession = async () => {
    try {
      await createSession(`Chat ${sessions.length + 1}`);
    } catch {
      console.error("Failed to create session");
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      reset();
    } catch {
      console.error("Logout failed");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-2 border-white/10 border-t-neon-400 rounded-full animate-spin" />
          <p className="text-dark-300 text-sm font-medium">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      {!user ? (
        <Auth onLogin={setUser} splashActive={showSplash} />
      ) : (
        <div className="flex h-screen overflow-hidden">
          <Sidebar
            username={user.username}
            sessions={sessions}
            currentSessionId={currentSessionId}
            onSelectSession={setCurrentSessionId}
            onNewSession={handleNewSession}
            onLogout={handleLogout}
            splashActive={showSplash}
          />
          <main className="flex-1 flex flex-col min-w-0">
            <ChatView
              sessionId={currentSessionId}
              messages={messages}
              onAddMessage={addMessage}
            />
          </main>
        </div>
      )}
      {showSplash && <SplashScreen isLoggedIn={!!user} onComplete={() => setShowSplash(false)} />}
    </>
  );
}
