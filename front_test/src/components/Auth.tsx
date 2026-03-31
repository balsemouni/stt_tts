import React, { useState } from "react";
import { User, Lock, Sparkles, Mail } from "lucide-react";
import { motion } from "motion/react";
import { authApi } from "../api";
import type { User as UserType } from "../types";

interface AuthProps {
  onLogin: (user: UserType) => void;
  splashActive?: boolean;
}

export const Auth: React.FC<AuthProps> = ({ onLogin, splashActive = false }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (isLogin) {
        const user = await authApi.login(email, password);
        onLogin(user);
      } else {
        await authApi.signup(email, username, password);
        setIsLogin(true);
        alert("Account created! Please log in.");
      }
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/3 w-[400px] h-[400px] bg-violet-500/6 rounded-full blur-[150px]" />
        <div className="absolute bottom-1/3 right-1/3 w-[300px] h-[300px] bg-neon-500/4 rounded-full blur-[120px]" />
        {/* Neural dot grid */}
        <div className="absolute inset-0 neural-grid opacity-[0.02]" />
        {/* Perspective grid */}
        <div className="absolute bottom-0 left-0 right-0 h-60 opacity-[0.05]"
          style={{ backgroundImage: 'linear-gradient(rgba(139,92,246,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.4) 1px, transparent 1px)', backgroundSize: '60px 60px', transform: 'perspective(400px) rotateX(60deg)', transformOrigin: 'bottom' }} />
        {/* Floating brain nodes */}
        <svg className="absolute top-[20%] left-[15%] w-32 h-32 opacity-[0.06]" viewBox="0 0 100 100">
          <circle cx="50" cy="20" r="3" fill="#22d3ee" style={{ animation: 'brainPulse 3s ease-in-out infinite' }} />
          <circle cx="25" cy="50" r="2.5" fill="#a78bfa" style={{ animation: 'brainPulse 3s ease-in-out infinite 0.5s' }} />
          <circle cx="75" cy="50" r="2.5" fill="#22d3ee" style={{ animation: 'brainPulse 3s ease-in-out infinite 1s' }} />
          <circle cx="50" cy="80" r="3" fill="#a78bfa" style={{ animation: 'brainPulse 3s ease-in-out infinite 1.5s' }} />
          <line x1="50" y1="20" x2="25" y2="50" stroke="#22d3ee" strokeWidth="0.5" opacity="0.4" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite' }} />
          <line x1="50" y1="20" x2="75" y2="50" stroke="#a78bfa" strokeWidth="0.5" opacity="0.4" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 0.5s' }} />
          <line x1="25" y1="50" x2="50" y2="80" stroke="#a78bfa" strokeWidth="0.5" opacity="0.3" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 1s' }} />
          <line x1="75" y1="50" x2="50" y2="80" stroke="#22d3ee" strokeWidth="0.5" opacity="0.3" strokeDasharray="3 3" style={{ animation: 'dataFlow 2s linear infinite 1.5s' }} />
        </svg>
        <svg className="absolute bottom-[25%] right-[12%] w-24 h-24 opacity-[0.04]" viewBox="0 0 100 100">
          <circle cx="50" cy="30" r="3" fill="#a78bfa" style={{ animation: 'brainPulse 4s ease-in-out infinite' }} />
          <circle cx="30" cy="60" r="2" fill="#22d3ee" style={{ animation: 'brainPulse 4s ease-in-out infinite 0.7s' }} />
          <circle cx="70" cy="60" r="2" fill="#a78bfa" style={{ animation: 'brainPulse 4s ease-in-out infinite 1.4s' }} />
          <line x1="50" y1="30" x2="30" y2="60" stroke="#22d3ee" strokeWidth="0.5" opacity="0.3" strokeDasharray="2 4" style={{ animation: 'dataFlow 3s linear infinite' }} />
          <line x1="50" y1="30" x2="70" y2="60" stroke="#a78bfa" strokeWidth="0.5" opacity="0.3" strokeDasharray="2 4" style={{ animation: 'dataFlow 3s linear infinite 0.5s' }} />
        </svg>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 25, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="max-w-md w-full relative"
      >
        <div className="glass-card rounded-2xl p-8 relative overflow-hidden">
          <div className="absolute inset-0 shimmer rounded-2xl pointer-events-none" />
          
          <div className="relative z-10">
            {/* Logo */}
            <div className="text-center mb-8">
              <motion.div 
                className="inline-flex items-center justify-center w-20 h-20 rounded-2xl mb-5 relative"
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: splashActive ? 0 : 1, scale: splashActive ? 0.5 : 1, rotateY: splashActive ? 0 : [0, 360] }}
                transition={{ opacity: { duration: 0.5, ease: 'easeOut' }, scale: { duration: 0.5, ease: 'easeOut' }, rotateY: { duration: 12, repeat: Infinity, ease: 'linear' } }}
                style={{ transformStyle: 'preserve-3d' }}
              >
                <svg width="80" height="80" viewBox="0 0 100 100" fill="none">
                  <defs>
                    <linearGradient id="authHexG" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.15" />
                      <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.15" />
                    </linearGradient>
                    <linearGradient id="authHexS" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.6" />
                      <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.6" />
                    </linearGradient>
                    <filter id="authGlow">
                      <feGaussianBlur stdDeviation="3" result="blur" />
                      <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                    </filter>
                  </defs>
                  <polygon points="50,5 92,27 92,73 50,95 8,73 8,27" fill="url(#authHexG)" stroke="url(#authHexS)" strokeWidth="1" filter="url(#authGlow)" />
                  <polygon points="50,20 78,35 78,65 50,80 22,65 22,35" fill="none" stroke="url(#authHexS)" strokeWidth="0.4" opacity="0.2" />
                  {/* Neural brain */}
                  <circle cx="50" cy="32" r="3" fill="#22d3ee" opacity="0.8" style={{ animation: 'brainPulse 2s ease-in-out infinite' }} />
                  <circle cx="35" cy="48" r="2.5" fill="#a78bfa" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.3s' }} />
                  <circle cx="65" cy="48" r="2.5" fill="#22d3ee" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.6s' }} />
                  <circle cx="50" cy="50" r="3.5" fill="white" opacity="0.85" style={{ animation: 'brainPulse 3s ease-in-out infinite' }} />
                  <circle cx="38" cy="65" r="2.5" fill="#22d3ee" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite 0.9s' }} />
                  <circle cx="62" cy="65" r="2.5" fill="#a78bfa" opacity="0.7" style={{ animation: 'brainPulse 2s ease-in-out infinite 1.2s' }} />
                  <line x1="50" y1="32" x2="35" y2="48" stroke="#22d3ee" strokeWidth="0.6" opacity="0.4" strokeDasharray="2 3" style={{ animation: 'dataFlow 2s linear infinite' }} />
                  <line x1="50" y1="32" x2="65" y2="48" stroke="#a78bfa" strokeWidth="0.6" opacity="0.4" strokeDasharray="2 3" style={{ animation: 'dataFlow 2s linear infinite 0.3s' }} />
                  <line x1="35" y1="48" x2="50" y2="50" stroke="#a78bfa" strokeWidth="0.4" opacity="0.3" />
                  <line x1="65" y1="48" x2="50" y2="50" stroke="#22d3ee" strokeWidth="0.4" opacity="0.3" />
                  <line x1="50" y1="50" x2="38" y2="65" stroke="#22d3ee" strokeWidth="0.5" opacity="0.3" strokeDasharray="2 3" style={{ animation: 'dataFlow 2s linear infinite 0.6s' }} />
                  <line x1="50" y1="50" x2="62" y2="65" stroke="#a78bfa" strokeWidth="0.5" opacity="0.3" strokeDasharray="2 3" style={{ animation: 'dataFlow 2s linear infinite 0.9s' }} />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xl font-black font-display gradient-text-bright relative z-10" style={{ filter: 'drop-shadow(0 0 10px rgba(6,182,212,0.3))' }}>AN</span>
                </div>
                <div className="absolute inset-0 rounded-2xl glow-cyan opacity-20" />
              </motion.div>
              
              <motion.h1
                className="text-2xl font-bold font-display gradient-text-bright mb-1.5 tracking-tight"
                initial={{ opacity: 0 }}
                animate={{ opacity: splashActive ? 0 : 1 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              >AskNova</motion.h1>
              <h2 className="text-lg font-semibold text-white/90 mb-1">{isLogin ? "Welcome Back" : "Create Account"}</h2>
              <p className="text-dark-200 text-sm">{isLogin ? "Sign in to your AI workspace" : "Join the future of AI"}</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                  className="p-3 bg-red-500/10 text-red-400 text-sm rounded-xl border border-red-500/15">
                  {error}
                </motion.div>
              )}
              
              <div>
                <label className="block text-[10px] font-semibold text-dark-200 mb-1.5 uppercase tracking-[0.15em]">Email</label>
                <div className="relative group">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 text-dark-400 group-focus-within:text-neon-400 transition-colors" size={16} />
                  <input type="email" required
                    className="w-full pl-11 pr-4 py-3 rounded-xl text-sm text-white outline-none transition-all placeholder:text-dark-400"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
                    onFocus={(e) => { e.target.style.borderColor = 'rgba(6,182,212,0.25)'; e.target.style.boxShadow = '0 0 0 3px rgba(6,182,212,0.06)'; }}
                    onBlur={(e) => { e.target.style.borderColor = 'rgba(255,255,255,0.07)'; e.target.style.boxShadow = 'none'; }}
                    placeholder="Enter your email" value={email} onChange={(e) => setEmail(e.target.value)} />
                </div>
              </div>

              {!isLogin && (
                <div>
                  <label className="block text-[10px] font-semibold text-dark-200 mb-1.5 uppercase tracking-[0.15em]">Username</label>
                  <div className="relative group">
                    <User className="absolute left-3.5 top-1/2 -translate-y-1/2 text-dark-400 group-focus-within:text-neon-400 transition-colors" size={16} />
                    <input type="text" required
                      className="w-full pl-11 pr-4 py-3 rounded-xl text-sm text-white outline-none transition-all placeholder:text-dark-400"
                      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
                      onFocus={(e) => { e.target.style.borderColor = 'rgba(6,182,212,0.25)'; e.target.style.boxShadow = '0 0 0 3px rgba(6,182,212,0.06)'; }}
                      onBlur={(e) => { e.target.style.borderColor = 'rgba(255,255,255,0.07)'; e.target.style.boxShadow = 'none'; }}
                      placeholder="Choose a username" value={username} onChange={(e) => setUsername(e.target.value)} />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-[10px] font-semibold text-dark-200 mb-1.5 uppercase tracking-[0.15em]">Password</label>
                <div className="relative group">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-dark-400 group-focus-within:text-neon-400 transition-colors" size={16} />
                  <input type="password" required
                    className="w-full pl-11 pr-4 py-3 rounded-xl text-sm text-white outline-none transition-all placeholder:text-dark-400"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
                    onFocus={(e) => { e.target.style.borderColor = 'rgba(6,182,212,0.25)'; e.target.style.boxShadow = '0 0 0 3px rgba(6,182,212,0.06)'; }}
                    onBlur={(e) => { e.target.style.borderColor = 'rgba(255,255,255,0.07)'; e.target.style.boxShadow = 'none'; }}
                    placeholder="Enter your password" value={password} onChange={(e) => setPassword(e.target.value)} />
                </div>
              </div>

              <motion.button type="submit" whileHover={{ scale: 1.02, y: -1 }} whileTap={{ scale: 0.98 }}
                className="w-full py-3 text-white font-semibold rounded-xl transition-all flex items-center justify-center gap-2 text-sm"
                style={{ background: 'linear-gradient(135deg, #06b6d4, #7c3aed)', boxShadow: '0 4px 20px rgba(6,182,212,0.15)' }}>
                <Sparkles size={15} />
                {isLogin ? "Sign In" : "Create Account"}
              </motion.button>
            </form>

            <div className="mt-5 text-center">
              <button onClick={() => setIsLogin(!isLogin)} className="text-neon-400 hover:text-neon-300 text-sm font-medium transition-colors">
                {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
