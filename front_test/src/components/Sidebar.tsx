import React, { useState } from "react";
import { Plus, LogOut, History, ChevronLeft, ChevronRight, Sparkles, Crown, MessageCircle } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import type { Session } from "../types";

interface SidebarProps {
  sessions: Session[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onLogout: () => void;
  username: string;
  splashActive?: boolean;
}

/* ── AskNova "AN" Logo ─────────────────────────────────────────────── */
const ANLogo = ({ size = 36 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sidebarGrad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
        <stop stopColor="#22d3ee" />
        <stop offset="0.5" stopColor="#a78bfa" />
        <stop offset="1" stopColor="#22d3ee" />
      </linearGradient>
      <filter id="sidebarGlow">
        <feGaussianBlur stdDeviation="2" result="blur" />
        <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
      </filter>
    </defs>
    {/* Hexagon */}
    <polygon points="24,3 44,14.5 44,33.5 24,45 4,33.5 4,14.5" fill="none" stroke="url(#sidebarGrad)" strokeWidth="1.2" filter="url(#sidebarGlow)" />
    <polygon points="24,9 38,17.5 38,30.5 24,39 10,30.5 10,17.5" fill="url(#sidebarGrad)" fillOpacity="0.06" />
    {/* Neural lines with data flow animation */}
    <line x1="24" y1="15" x2="16" y2="22" stroke="#22d3ee" strokeWidth="0.5" opacity="0.5" strokeDasharray="2 2" style={{ animation: 'dataFlow 2s linear infinite' }} />
    <line x1="24" y1="15" x2="32" y2="22" stroke="#a78bfa" strokeWidth="0.5" opacity="0.5" strokeDasharray="2 2" style={{ animation: 'dataFlow 2s linear infinite 0.3s' }} />
    <line x1="16" y1="22" x2="24" y2="30" stroke="#a78bfa" strokeWidth="0.5" opacity="0.4" strokeDasharray="2 2" style={{ animation: 'dataFlow 2s linear infinite 0.6s' }} />
    <line x1="32" y1="22" x2="24" y2="30" stroke="#22d3ee" strokeWidth="0.5" opacity="0.4" strokeDasharray="2 2" style={{ animation: 'dataFlow 2s linear infinite 0.9s' }} />
    <line x1="16" y1="22" x2="32" y2="22" stroke="#22d3ee" strokeWidth="0.3" opacity="0.2" />
    {/* Neural nodes */}
    <circle cx="24" cy="15" r="2" fill="#22d3ee" opacity="0.8" style={{ animation: 'brainPulse 2s ease-in-out infinite' }} />
    <circle cx="16" cy="22" r="1.5" fill="#a78bfa" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.3s' }} />
    <circle cx="32" cy="22" r="1.5" fill="#22d3ee" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.6s' }} />
    <circle cx="24" cy="30" r="2" fill="#a78bfa" opacity="0.8" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.9s' }} />
    <circle cx="24" cy="22" r="2.5" fill="white" opacity="0.85" style={{ animation: 'brainPulse 3s ease-in-out infinite' }} />
  </svg>
);

export const Sidebar: React.FC<SidebarProps> = ({
  sessions, currentSessionId, onSelectSession, onNewSession, onLogout, username, splashActive = false
}) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.div
      animate={{ width: collapsed ? 68 : 280 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
      className="h-screen flex flex-col relative z-50 shrink-0 overflow-hidden"
      style={{
        background: 'linear-gradient(180deg, rgba(8,8,20,0.95) 0%, rgba(4,4,10,0.98) 100%)',
        borderRight: '1px solid rgba(255,255,255,0.04)',
      }}
    >
      {/* Ambient glow at top */}
      <div className="absolute top-0 left-0 right-0 h-40 pointer-events-none"
        style={{ background: 'radial-gradient(ellipse at 50% -20%, rgba(139,92,246,0.06) 0%, transparent 70%)' }} />
      {/* Neural dot grid overlay */}
      <div className="absolute inset-0 pointer-events-none neural-grid opacity-[0.02]" />

      {/* ── Logo Area ── */}
      <div className="p-4 pt-5 relative z-10">
        <div className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3 px-1'} mb-5`}>
          <motion.div 
            className="shrink-0"
            whileHover={{ scale: 1.05 }}
            initial={{ opacity: 0, scale: 0.3 }}
            animate={{ opacity: splashActive ? 0 : 1, scale: splashActive ? 0.3 : 1, rotateY: splashActive ? 0 : [0, 3, -3, 0] }}
            transition={{ opacity: { duration: 0.4, ease: 'easeOut' }, scale: { duration: 0.4, ease: 'easeOut' }, rotateY: { duration: 6, repeat: Infinity, ease: "easeInOut" } }}
            style={{ transformStyle: 'preserve-3d' }}
          >
            <ANLogo size={collapsed ? 32 : 38} />
          </motion.div>
          <AnimatePresence>
            {!collapsed && !splashActive && (
              <motion.div
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.3 }}
              >
                <h1 className="font-bold text-[15px] font-display tracking-tight gradient-text-bright leading-tight">AskNova</h1>
                <p className="text-[9px] text-dark-300 font-medium tracking-[0.2em] uppercase mt-0.5">AI Intelligence</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* New Chat Button */}
        <motion.button
          onClick={onNewSession}
          whileHover={{ scale: 1.02, y: -1 }}
          whileTap={{ scale: 0.98 }}
          className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-semibold transition-all relative overflow-hidden group ${collapsed ? 'px-2' : ''}`}
          style={{
            background: 'linear-gradient(135deg, rgba(6,182,212,0.15) 0%, rgba(139,92,246,0.15) 100%)',
            border: '1px solid rgba(6,182,212,0.15)',
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-neon-400/10 to-violet-400/10 opacity-0 group-hover:opacity-100 transition-opacity" />
          <Plus size={16} className="text-neon-400 relative z-10" />
          {!collapsed && <span className="text-neon-300 relative z-10">New Chat</span>}
        </motion.button>
      </div>

      {/* ── Sessions ── */}
      <div className="flex-1 overflow-y-auto px-2 py-1 relative z-10">
        {!collapsed && (
          <div className="flex items-center gap-1.5 px-3 py-2 mb-1">
            <History size={11} className="text-dark-400" />
            <span className="text-[9px] font-bold text-dark-400 uppercase tracking-[0.2em]">History</span>
          </div>
        )}
        <div className="space-y-0.5">
          {sessions.map((session) => (
            <motion.button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              whileHover={{ x: collapsed ? 0 : 2 }}
              className={`w-full text-left px-3 py-2.5 rounded-xl transition-all relative group ${
                currentSessionId === session.id
                  ? ""
                  : "hover:bg-white/[0.02]"
              }`}
              style={currentSessionId === session.id ? {
                background: 'linear-gradient(135deg, rgba(6,182,212,0.08) 0%, rgba(139,92,246,0.06) 100%)',
                border: '1px solid rgba(6,182,212,0.1)',
              } : { border: '1px solid transparent' }}
            >
              {collapsed ? (
                <div className="flex items-center justify-center">
                  <MessageCircle size={15} className={currentSessionId === session.id ? "text-neon-400" : "text-dark-400"} />
                </div>
              ) : (
                <>
                  <div className={`text-[13px] font-medium truncate pr-4 ${
                    currentSessionId === session.id ? "text-white" : "text-dark-200 group-hover:text-dark-100"
                  }`}>
                    {session.title}
                  </div>
                  <div className={`text-[10px] mt-0.5 ${
                    currentSessionId === session.id ? "text-neon-400/50" : "text-dark-400"
                  }`}>
                    {new Date(session.created_at).toLocaleDateString()}
                  </div>
                </>
              )}
              {/* Active indicator */}
              {currentSessionId === session.id && (
                <motion.div
                  layoutId="activeSess"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-[2px] h-5 rounded-full"
                  style={{ background: 'linear-gradient(to bottom, #22d3ee, #a78bfa)' }}
                />
              )}
            </motion.button>
          ))}
        </div>
        {sessions.length === 0 && !collapsed && (
          <div className="text-center py-10">
            <div className="w-10 h-10 mx-auto mb-3 rounded-xl flex items-center justify-center" style={{ background: 'rgba(139,92,246,0.08)' }}>
              <Sparkles size={18} className="text-violet-400/50" />
            </div>
            <p className="text-dark-400 text-xs">No conversations yet</p>
            <p className="text-dark-500 text-[10px] mt-1">Start a new chat above</p>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div className="p-3 relative z-10" style={{ borderTop: '1px solid rgba(255,255,255,0.03)' }}>
        {/* User info */}
        <div className={`flex items-center ${collapsed ? 'justify-center' : 'gap-2.5 px-2'} mb-2.5`}>
          <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-[11px] shrink-0"
            style={{
              background: 'linear-gradient(135deg, rgba(6,182,212,0.2), rgba(139,92,246,0.2))',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            {username[0]?.toUpperCase()}
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-[13px] font-semibold text-white truncate">{username}</p>
              <div className="flex items-center gap-1">
                <Crown size={9} className="text-violet-400" />
                <p className="text-[9px] text-dark-300 font-medium">Free Tier</p>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex gap-1.5">
          <button
            onClick={onLogout}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 text-dark-400 hover:text-red-400 hover:bg-red-500/5 rounded-lg transition-all text-xs font-medium ${collapsed ? '' : ''}`}
          >
            <LogOut size={14} />
            {!collapsed && "Sign Out"}
          </button>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-2 text-dark-400 hover:text-neon-400 hover:bg-white/[0.02] rounded-lg transition-all"
          >
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>
      </div>
    </motion.div>
  );
};
